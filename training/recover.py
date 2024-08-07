"""
Continue pretraining / finetuning after something went wrong.
"""

import logging
import torch
import os

import config
from data import prepare_dataset
from train import setup_model_optim_sched_scaler, setup_criteria_mixup, _train, setup_tracking_and_logging
from utils import ddp_setup, prep_kwargs, ddp_cleanup
from models import load_pretrained


def continue_training(model, **kwargs):
    model_path = model
    save_state = torch.load(model)

    # state is of the form
    #
    # state = {'epoch': epochs,
    #          'model_state': model.state_dict(),
    #          'optimizer_state': optimizer.state_dict(),
    #          'scheduler_state': scheduler.state_dict(),
    #          'args': dict(args),
    #          'run_name': run_name,
    #          'stats': metrics}

    args = prep_kwargs(save_state["args"])

    args.distributed, device, world_size, rank, gpu_id = ddp_setup()
    torch.cuda.set_device(device)

    if "world_size" in args and args.world_size is not None:
        global_bs = args.batch_size * args.world_size
    else:
        # assume global bs is given in kwargs
        global_bs = kwargs["batch_size"]
    args.batch_size = int(global_bs / world_size)
    args.world_size = world_size

    if "dataset" in args and args.dataset is not None:
        dataset = args.dataset
    else:
        # get default dataset for the task
        if args.task == "pre-train":
            dataset = "ImageNet21k"
        else:
            dataset = "ImageNet"
        args.dataset = dataset

    start_epoch = save_state["epoch"]
    if "epochs" in args and args.epochs is not None and args.epochs != start_epoch:
        epochs = args.epochs
    else:
        epochs = kwargs["epochs"]

    full_run_name, logging_file_name = setup_tracking_and_logging(args, rank, append_model_path=model_path)
    logging.info(f"Logging run {full_run_name} to file {logging_file_name}")

    save_state = torch.load(model_path, map_location="cpu")
    old_args = prep_kwargs(save_state["args"])

    for key in config.default_kwargs.keys():
        args[key] = old_args[key]

    args["run_name"] = old_args["run_name"]

    # get the datasets & dataloaders
    train_loader, args.n_classes, args.ignore_index = prepare_dataset(dataset, old_args, rank=rank)
    val_loader, _, __ = prepare_dataset(dataset, old_args, train=False, rank=rank)

    # model_name = args.model

    model, args, _, __ = load_pretrained(model_path, args)

    model, optimizer, scheduler, scaler = setup_model_optim_sched_scaler(model, device, epochs, args)

    optimizer.load_state_dict(save_state["optimizer_state"])
    scheduler.load_state_dict(save_state["scheduler_state"])

    # log all devices
    logging.info(f"training on {device} -> {torch.cuda.get_device_name(device) if args.device != 'cpu' else ''}")
    if rank == 0:
        logging.info(f"torch version {torch.__version__}")
        logging.info(f"full set of arguments: {args}")

    if args.seed:
        torch.manual_seed(args.seed)

    criterion, val_criterion, mixup = setup_criteria_mixup(args)

    model_folder = f"{args.results_folder}/models/{full_run_name}/"
    os.makedirs(model_folder, exist_ok=True)
    if rank == 0:
        logging.info(f"start training at epoch {start_epoch}")
        logging.info(f"Run name: '{full_run_name}'")
        logging.info(f"Logging file name: '{logging_file_name}'")

    res = _train(
        model,
        train_loader,
        optimizer,
        rank,
        epochs,
        device,
        mixup,
        criterion,
        world_size,
        scheduler,
        args,
        val_loader,
        val_criterion,
        model_folder,
        scaler=scaler,
        do_metrics_calculation=True,
        start_epoch=start_epoch,
        show_tqdm=args.tqdm,
    )

    if rank == 0:
        logging.info(
            f"Run '{full_run_name}' is done. Top-1 validation accuracy (since epoch {start_epoch}): {res['best_val_acc']*100:.2f}%"
        )
    ddp_cleanup(args=args)
