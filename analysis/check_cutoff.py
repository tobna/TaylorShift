import argparse
import math
import utils
import logging
import torch
import subprocess


def run_test(model, d, B, compiled, H=1, interm_points=15, N_0=None):
    if N_0 is None:
        N_0 = utils.theoretical_cutoff(d)
        logging.info(f"B={B: >4d} d={d: >3d} |-> N_0={N_0: >6d}")
    else:
        logging.info(f"B={B: >4d} d={d: >3d} manual critical point={N_0: >6d}")

    exponential_range_start = 6
    exponential_range_end = int(math.log2(N_0))
    exponential_interm_points = 4
    exponential_sample_points = (exponential_range_end - exponential_range_start) * 4
    exponential_range_diff = 1 / exponential_interm_points
    exponential_range = [
        round(2 ** (exponential_range_start + i * exponential_range_diff)) for i in range(exponential_sample_points)
    ]

    linear_range_min = 0.8 * N_0
    linear_range_max = 2 * N_0 + 300
    linear_del = (linear_range_max - linear_range_min) / interm_points
    linear_range = [round(linear_range_min + i * linear_del) for i in range(interm_points)]

    Ns = exponential_range + linear_range
    Ns = {N - (N % 16) for N in Ns}
    Ns = sorted(list(Ns))
    for N in Ns:
        p_args = ["python3", "check_one.py", "-d", str(d), "-m", model, "-B", str(B), "-N", str(N), "-H", str(H)]
        if compiled:
            p_args.append("-c")
        subprocess.Popen(p_args).wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=int, nargs="?", required=True, help="Head dimensionality to test.")
    parser.add_argument("-b", type=int, nargs="?", required=True, help="Batch size to test.")
    parser.add_argument(
        "-m",
        "--model",
        choices=["efficient", "trivial", "baseline", "separate"],
        nargs="?",
        required=True,
        help="Model to test.",
    )
    parser.add_argument("-c", "--compile", action="store_true", help="Use torch 2.0 compilation.")
    parser.add_argument(
        "-P", type=int, nargs="?", default=None, help="Critical point to go into linear sample mode from."
    )
    args = parser.parse_args()

    H = 1
    interm_points = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = torch.cuda.get_device_name(device) if device.type != "cpu" else utils.get_cpuinfo()
    del device

    full_run_name = f"cutoff_experiment@{device_info.replace(' ', '_')}_b={args.b}_d={args.d}_model={args.model}"
    logging_file_name = f"{full_run_name}.logdb".replace("/", "_")
    logging.basicConfig(
        format=f"%(asctime)s [{device_info}] %(levelname)s: %(message)s",
        datefmt="%d.%m.%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler("logs/" + logging_file_name), logging.StreamHandler()],
    )

    logging.info(f"Logging to file: {logging_file_name}")
    logging.info(f"Device: {device_info}")

    run_test(args.model, args.d, args.b, compiled=args.compile, H=H, interm_points=interm_points, N_0=args.P)
