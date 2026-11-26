#!/bin/bash
# Llama-2-7B on GSM8K with AdaHyC, seed 11 (report Table 4). Needs Hugging Face access to meta-llama/Llama-2-7b-hf.
# About 3 h of training on an A100 40GB plus about 1 h of evaluation. A checkpoint is saved every 300 steps and an
# interrupted run resumes automatically when the same command is run again.
# Peak training memory is 6.5 GiB; on a 24GB card, evaluate with EVAL_BS=16 (32 needs more memory).
#
# Our run was interrupted at step ~1,323 and resumed from step 1,200 on a different GPU; a run started here goes
# straight through, so its result can differ from ours by the usual seed/hardware noise (about 1.3 points).
set -e
cd "$(dirname "$0")/../external/HyC-LoRA-release"

TASK=gsm8k MODEL=meta-llama/Llama-2-7b-hf LAYER_TYPE=adahyc FIXED=True BIT_PLAN=../../probe/out_llama2_7b/plan.json \
    SAVE_STEPS=300 SEED=11 EVAL_BS=${EVAL_BS:-32} bash run_da.sh
