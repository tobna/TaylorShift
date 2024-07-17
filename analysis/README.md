# TaylorShift Analysis
This is the code for analyzing the TaylorShift module.
Namely, this is the basis for our Figure 2 and for the scaling behavior of intermediate results.


# Usage
## Individual Experiments
You can run individual experiments on a slurm cluster using one of the [runscripts](runscripts).
To check the empirical efficiency transition points given a dimension and head size, pass the corresponding args to the script together with the attention version you want to use.
For example
```bash
./runscripts/check_cutoff_A100 -b 32 -d 16 -m baseline
```
calculates the throughput and memory requirements for the baseline attention mechanisms with a per-head dimension of 16 and a batch_size of 32 for all the sequence length needed.

## All Combinations
To automatically run all combinations, use [this](run_all.py) script.

## Scaling Behavior
Check the scaling behavior of intermediate results using [scaling_behavior.py](scaling_behavior.py) or use [this](runscripts/scaling_behavior_A100) to run on a slurm cluster.

