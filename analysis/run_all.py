import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--partition", type=str, nargs="?", required=True)
parser.add_argument(
    "-m",
    "--models",
    choices=["efficient", "trivial", "baseline"],
    nargs="*",
    default=["efficient", "trivial", "baseline"],
    help="Models to test.",
)
args = parser.parse_args()

ds = [16, 32, 64, 128]
models = args.models
print(f"testing models {models}")
compile = [True, False]

Bs = [1, 16, 32, 64, 128, 1024]

processes = []
for d in ds:
    for model in models:
        for comp in compile:
            for B in Bs:
                process_args = [f"runscripts/check_cutoff_{args.partition}", "-d", str(d), "-m", model, "-b", str(B)]
                if comp:
                    process_args.append("-c")
                processes.append(subprocess.Popen(process_args))

for process in processes:
    process.wait()
