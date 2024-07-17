"""
Module to evaluate trained models.
"""

import torch
import logging
from timm.loss import LabelSmoothingCrossEntropy
from data import prepare_dataset
from metrics import calculate_metrics
from models import load_pretrained
from train import _evaluate, setup_tracking_and_logging
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import set_filter_warnings, prep_kwargs, ddp_setup, get_cpu_name, ddp_cleanup


def evaluate_metrics(model, dataset, **kwargs):
    """Evaluate efficiency metrics for a given model

    Parameters
    ----------
    model : str
        path to model .tar
    dataset : str
        name of the dataset to evaluate on
    kwargs
        further arguments
    """
    set_filter_warnings()
    model_path = model
    args = prep_kwargs(kwargs)
    if args.cuda:
        args.distributed, device, world_size, rank, _ = ddp_setup()
        torch.cuda.set_device(device)
    else:
        args.distributed = False
        device = torch.device("cpu")
        rank = 0
        args.compile_model = False

    save_state = torch.load(model_path, map_location="cpu")
    old_args = prep_kwargs(save_state["args"])
    args.model = old_args.model
    args.dataset = dataset
    args.run_name = old_args.run_name
    args.experiment_name = old_args.experiment_name
    setup_tracking_and_logging(args if args.new_log else old_args, rank, append_model_path=model_path)

    train_loader, args.n_classes, args.ignore_index = prepare_dataset(dataset, args)

    model, args, old_args, save_state = load_pretrained(model, args, new_dataset_params=True)
    model = model.to(device)
    old_args["eval_imsize"] = args.imsize
    args.model = model_name = old_args.model
    args.dataset = dataset

    if args.compile_model:
        model = torch.compile(model)

    if rank == 0:
        logging.info(
            f"Evaluate metrics for model {model_name} on {dataset}. "
            f"It was {old_args.task.replace('-','')}d on {old_args.dataset} for {save_state['epoch']} "
            f"epochs."
        )
        # logging.info(f"full set of arguments: {args}")
        logging.info(f"full set of arguments: {old_args}")

    logging.info(
        f"evaluating on {device} -> {torch.cuda.get_device_name(device) if device.type != 'cpu' else get_cpu_name()}"
    )

    inp = next(iter(train_loader))[0]
    print(f"input shape: {inp.shape}")
    metrics = calculate_metrics(args, model, rank=rank, input=inp.to(device), device=device)
    if rank == 0:
        logging.info(f"Metrics: {metrics}")


def evaluate(model, dataset, **kwargs):
    """Evaluate model accuracy

    Parameters
    ----------
    model : str
        path to model state .tar
    dataset : str
        name of the dataset to evaluate on
    kwargs
        further arguments
    """
    set_filter_warnings()
    model_path = model
    args = prep_kwargs(kwargs)
    args.dataset = dataset
    if args.cuda:
        args.distributed, device, world_size, rank, _ = ddp_setup()
        torch.cuda.set_device(device)
    else:
        args.distributed = False
        device = torch.device("cpu")
        world_size = 1
        rank = 0
        args.compile_model = False
    args.batch_size = int(args.batch_size / world_size)

    save_state = torch.load(model_path, map_location="cpu")
    old_args = prep_kwargs(save_state["args"])
    args.model = old_args.model
    args.dataset = dataset
    args.run_name = old_args.run_name
    args.experiment_name = old_args.experiment_name
    full_run_name, logging_file_name = setup_tracking_and_logging(
        args if args.new_log else old_args, rank, append_model_path=model_path
    )

    val_loader, args.n_classes, args.ignore_index = prepare_dataset(dataset, args, train=False)

    model, args, old_args, save_state = load_pretrained(model, args, new_dataset_params=True)
    model = model.to(device)
    args.model = model_name = old_args.model
    args.dataset = dataset

    if rank == 0:
        logging.info(
            f"Evaluate model {model_name} on {dataset}. "
            f"It was pretrained on {old_args.dataset} for {save_state['epoch']} epochs."
        )

    if args.distributed:
        model = DDP(model)

    if args.compile_model:
        model = torch.compile(model)

    # log all devices
    logging.info(
        f"evaluating on {device} -> {torch.cuda.get_device_name(device) if device.type != 'cpu' else get_cpu_name()}"
    )
    if rank == 0:
        logging.info(f"torch version {torch.__version__}")
        logging.info(f"full set of arguments: {args}")
        logging.info(f"full set of old arguments: {old_args}")

    if args.seed:
        torch.manual_seed(args.seed)

    val_criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    if rank == 0:
        logging.info("start evaluation")
        logging.info(f"Run name: '{full_run_name}'")
        logging.info(f"Logging file name: '{logging_file_name}'")

    if rank == 0:
        val_time, val_loss, val_accs = _evaluate(
            model.to(device),
            val_loader,
            epoch=save_state["epoch"] - 1,
            rank=rank,
            device=device,
            world_size=world_size,
            val_criterion=val_criterion,
            args=args,
        )
        log_s = f"Evaluation done in {val_time}s: loss={val_loss}"
        for key, val in val_accs.items():
            log_s += f", {key}={val}%"
        logging.info(log_s)
    else:
        _evaluate(
            model.to(device),
            val_loader,
            epoch=save_state["epoch"] - 1,
            rank=rank,
            device=device,
            world_size=world_size,
            val_criterion=val_criterion,
            args=args,
        )

    ddp_cleanup(args=args)
