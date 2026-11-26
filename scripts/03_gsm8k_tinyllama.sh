#!/bin/bash
# TinyLlama-1.1B on GSM8K: fixed HyC-LoRA vs AdaHyC, seeds 11-14 (report Tables 1 and 5, Figure 5).
# Each run trains for 2,802 steps, then evaluates sampled and greedy on the 1,319 test questions.
# On an A100 40GB one run takes about 1.5-2 h; we ran three at a time on one GPU.
# Outputs: external/HyC-LoRA-release/exp_results_da/<tag>/ and <tag>.log; ours are in results/gsm8k/.
#
# Usage: bash scripts/03_gsm8k_tinyllama.sh            (all seeds)
#        SEEDS="11" bash scripts/03_gsm8k_tinyllama.sh  (one seed)
set -e
cd "$(dirname "$0")/../external/HyC-LoRA-release"
M=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T

for s in ${SEEDS:-11 12 13 14}; do
    TASK=gsm8k MODEL=$M LAYER_TYPE=intra_inter FIXED=True SEED=$s bash run_da.sh
    TASK=gsm8k MODEL=$M LAYER_TYPE=adahyc FIXED=True BIT_PLAN=../../probe/out_tinyllama/plan.json SEED=$s bash run_da.sh
done

# HyC-LoRA exactly as released (all five defects present), for comparison with the paper's 12.96:
#   TASK=gsm8k MODEL=$M LAYER_TYPE=intra_inter SEED=11 bash run_da.sh
