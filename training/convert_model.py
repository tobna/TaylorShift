import torch

import main
import utils
from models import prepare_model


def convert_model(state_dict_path, new_model, new_args, use_new_params=False):
    """Load a model from a state dict and convert it to a new model.

    Parameters
    ----------
    state_dict_path : str
        path to the state dict file
    new_model : str
        new model
    new_args : DotDict
        new model parameters
    use_new_params : bool
        change model parameters (imsize, n_classes) to the ones from state dict.

    Returns
    -------
    nn.Module
        the new model
    """
    save_state = torch.load(state_dict_path, map_location="cpu")
    old_args = utils.prep_kwargs(save_state["args"])
    # old_model = old_args.model

    # load the new model
    new_args.model = new_model

    goal_param_keys = ["imsize", "n_classes", "patch_size", "input_dim", "max_seq_len"]
    goal_params = {key: new_args.pop(key) for key in goal_param_keys if key in new_args and new_args[key] is not None}

    # first use the old values, then adapt to the new ones
    for key in goal_params.keys():
        new_args[key] = old_args[key]

    model = prepare_model(new_model, new_args)

    file_save_state = utils.remove_prefix(save_state["model_state"], prefix="_orig_mod.")
    file_save_state = utils.remove_prefix(file_save_state)

    model_keys = set(model.state_dict().keys())
    file_keys = set(file_save_state.keys())

    model_minus_file = model_keys.difference(file_keys)
    file_minus_model = file_keys.difference(model_keys)

    model.load_state_dict(file_save_state, strict=False)

    if len(model_minus_file) > 0:
        print(f"save state did not include: {model_minus_file}")
    if len(file_minus_model) > 0:
        print(f"key file had extra: {file_minus_model}")

    old_args.backbone = old_args.model

    if use_new_params:
        model._set_input_strand(patch_size=goal_params.get("patch_size"), res=goal_params["imsize"])
        model.set_num_classes(goal_params["n_classes"])
        if "max_seq_len" in goal_params:
            model.set_max_seq_len(goal_params["max_seq_len"])
        if "input_dim" in goal_params:
            model.set_input_dim(goal_params["input_dim"])

        for key, val in goal_params.items():
            old_args[key] = val

    return new_model, old_args, save_state


if __name__ == "__main__":
    parser = main.base_parser(
        "Converting models which have the same set of parameters. For example, eff-TaylorShift <-> dir-TaylorShift."
    )
    parser.add_argument(
        "-dict", "--state_dict_path", type=str, default="state_dict.pth", help="path to the state dict file"
    )
    parser.add_argument("--use_new_params", action="store_true", help="use new parameters for the model")
    parser.add_argument("-f", "--model_folder", type=str, default="models", help="folder to store new model in")

    args = parser.parse_args(args=dict(vars(parser.parse_args())))

    new_model, model_args, old_save_state = convert_model(
        args.state_dict_path, args.model, args, use_new_params=args.use_new_params
    )

    kwargs = {key: old_save_state[key] for key in ["val_accs", "epoch_accs", "stats"] if key in old_save_state}

    utils.save_model_state(
        args.model_folder,
        model_args.epoch,
        model_args,
        new_model.state_dict(),
        regular_save=False,
        additional_reason=f"convert_{model_args.backbone}_to_{model_args.model}",
        **kwargs,
    )
