#!/bin/bash
# TinyLlama-1.1B on WikiText-2, corrected HyC-LoRA vs AdaHyC, seed 11 (report Appendix C.2).
# Sequence length 1,024, 3 epochs; about 40 min per run on an A100 40GB.
# Outputs: external/HyC-LoRA-release/exp_results_da/wikitext2-*; ours are in results/wikitext2/.
set -e
cd "$(dirname "$0")/../external/HyC-LoRA-release"
M=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T

TASK=wikitext2 MODEL=$M LAYER_TYPE=intra_inter FIXED=True SEED=11 bash run_da.sh
TASK=wikitext2 MODEL=$M LAYER_TYPE=adahyc FIXED=True BIT_PLAN=../../probe/out_tinyllama/plan.json SEED=11 bash run_da.sh
