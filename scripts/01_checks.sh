#!/bin/bash
# Compare gradient accuracy against uncompressed QLoRA (Table 6) and record activation memory (Table 3).
# check_da.py verifies the fixed kernels track true QLoRA better than the released ones and exits nonzero if not.
# check_mem.py saves results to external/HyC-LoRA-release/exp_results_da/check_mem-<model>.json.
# Outputs land in results/logs/check_da.log and results/memory/.
set -e
cd "$(dirname "$0")/../external/HyC-LoRA-release"
TL=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T

python check_da.py $TL
python check_mem.py $TL --plan ../../probe/out_tinyllama/plan.json
python check_mem.py meta-llama/Llama-2-7b-hf --plan ../../probe/out_llama2_7b/plan.json
