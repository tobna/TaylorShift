import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import argparse
import traceback

from attention import LinearTaylorAttention, SquaredTaylorAttention, BaselineAttention, SepLinearTaylorAttention
import metrics
import torch
import utils
import logging


def run_test(model_name, model, device, B, d, N, compiled, H=1):
    try:
        qkv = torch.randn(3, B, H, N, d, device=device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        logging.warning(f"Error creating qkv (B={B: >4d} d={d: >3d} N={N: >6d}): {e}")
        traceback.print_exc()
        return

    if device.type != "cpu":
        # Throughput
        try:
            tp = metrics.measure_throughput(model, qkv)
            tp = tp / 3 * B
            logging.info(f"B={B: >4d} d={d: >3d} N={N: >6d} Throughput: {model_name} implementation {tp}")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logging.warning(f"Error computing throughput (B={B: >4d} d={d: >3d} N={N: >6d}): {e}")
            traceback.print_exc()

        # Memory
        try:
            mem = metrics.inference_memory(utils.defaults, model, qkv, device=device, batch_sizes=[3])[3]
            mem = mem / B
            logging.info(f"B={B: >4d} d={d: >3d} N={N: >6d} Inference Memory: {model_name} implementation {mem}")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logging.warning(f"Error computing inference memory (B={B: >4d} d={d: >3d} N={N: >6d}): {e}")
        except KeyError as e:
            logging.warning(
                f"KeyError computing inference memory (B={B: >4d} d={d: >3d} N={N: >6d}): "
                f"No key {e}, since there was another error during the calculation."
            )

    # FLOPS
    try:
        flops = metrics.flops(utils.defaults, model._orig_mod if compiled else model, qkv, n_ims=3)
        logging.info(f"B={B: >4d} d={d: >3d} N={N: >6d} FLOPS: {model_name} implementation {flops}")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        logging.warning(f"Error computing FLOPS (B={B: >4d} d={d: >3d} N={N: >6d}): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=int, nargs="?", required=True, help="Head dimensionality to test.")
    parser.add_argument(
        "-m",
        "--model",
        choices=["efficient", "trivial", "baseline", "separate"],
        nargs="?",
        required=True,
        help="Model to test.",
    )
    parser.add_argument("-c", "--compile", action="store_true", help="Use torch 2.0 compilation.")
    parser.add_argument("-B", type=int, nargs="?", required=True, help="Batch size to test.")
    parser.add_argument("-N", type=int, nargs="?", required=True, help="Sequence length to test.")
    parser.add_argument("-H", type=int, nargs="?", default=1, help="Number of attention heads to test.")
    args = parser.parse_args()

    if args.model == "efficient":
        model = LinearTaylorAttention()
    elif args.model == "trivial":
        model = SquaredTaylorAttention()
    elif args.model == "baseline":
        model = BaselineAttention()
    elif args.model == "separate":
        model = SepLinearTaylorAttention()
    else:
        raise ValueError(f"Unknown model {args.model}")
    if args.compile:
        model = torch.compile(model)

    try:
        torch._dynamo.config.log_level = logging.WARNING
    except:  # noqa: E722
        pass

    device = torch.cuda.set_device(
        0 if torch.cuda.is_available() else -1
    )  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = torch.cuda.get_device_name(device) if device.type != "cpu" else utils.get_cpuinfo()
    model = model.to(device)

    full_run_name = f"cutoff_experiment@{device_info.replace(' ', '_')}_b={args.B}_d={args.d}_model={args.model}"
    logging_file_name = f"{full_run_name}.logdb".replace("/", "_")
    logging.basicConfig(
        format=f"%(asctime)s [{device_info}] %(levelname)s: %(message)s",
        datefmt="%d.%m.%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler("logs/" + logging_file_name), logging.StreamHandler()],
    )

    run_test(
        f"{args.model}_{'comp' if args.compile else 'no_comp'}",
        model,
        device,
        args.B,
        args.d,
        args.N,
        compiled=args.compile,
        H=args.H,
    )
