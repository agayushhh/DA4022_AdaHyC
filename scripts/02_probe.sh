#!/bin/bash
# Sensitivity probe plus greedy allocator (report Sections 3.3-3.4, Figures 2 and 4, Table 2).
#
# The training scripts use the plans we actually trained with, probe/out_tinyllama/plan.json and
# probe/out_llama2_7b/plan.json. To keep those intact, this script writes to probe/rerun_<model>/ instead.
# A re-probed plan can differ slightly from ours (GPU nondeterminism in the 50 LoRA warm-up steps); to train with it,
# pass its path as BIT_PLAN to run_da.sh.
#
# Usage: bash scripts/02_probe.sh [tinyllama|llama2_7b|all]   (default: all)
set -e
cd "$(dirname "$0")/.."
TL=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
L2=meta-llama/Llama-2-7b-hf
WHICH=${1:-all}

if [ "$WHICH" = tinyllama ] || [ "$WHICH" = all ]; then
    python probe/probe.py --model $TL --out_dir probe/rerun_tinyllama
    python probe/allocate.py --sens probe/rerun_tinyllama/sensitivity.json --model $TL --out probe/rerun_tinyllama/plan.json
fi
if [ "$WHICH" = llama2_7b ] || [ "$WHICH" = all ]; then
    # 7B: batch 2 and only 2/4-bit measurements (8-bit is extrapolated as s(4)/16) to fit memory and time
    python probe/probe.py --model $L2 --batch_size 2 --bits 2,4 --out_dir probe/rerun_llama2_7b
    python probe/allocate.py --sens probe/rerun_llama2_7b/sensitivity.json --model $L2 --out probe/rerun_llama2_7b/plan.json
fi
