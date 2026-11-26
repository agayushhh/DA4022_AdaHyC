#!/bin/bash
# DA4022 paper-config runner (HyC-LoRA Table 8 settings). Everything is set through env vars, e.g.
#   TASK=gsm8k MODEL=/models/llama-2-7b-hf Q_BIT=2 FIXED=True bash run_da.sh
#   TASK=wikitext2 LAYER_TYPE=adahyc BIT_PLAN=plans/p.json REBUILD_TAU=0.2 SEED=12 bash run_da.sh
#   TASK=gsm8k LAYER_TYPE=none SAVE_STEPS=200 bash run_da.sh   # true QLoRA, checkpoint every 200 steps
# Note: the shipped run_gsm8k.sh is NOT the paper config (layer_type=intra, q_bit=8, layernorm_outlier_ratio=0).
# Output (flat): exp_results_da/<tag>/{adapter/, metrics.json, trainer_state.json, predictions_*.json} + exp_results_da/<tag>.log
# Re-running the same tag resumes from its latest checkpoint iff SAVE_STEPS is set; otherwise it retrains from scratch.
set -ex -o pipefail

TASK=${TASK:-gsm8k}                     # gsm8k | wikitext2
MODEL=${MODEL:-meta-llama/Llama-2-7b-hf} # path must contain "llama" (any case)
LAYER_TYPE=${LAYER_TYPE:-intra_inter}   # none (plain QLoRA, HF+PEFT path) | baseline | intra | intra_inter | intra_inter_full_fuse | adahyc
Q_BIT=${Q_BIT:-2}
SEED=${SEED:-11}
FIXED=${FIXED:-False}                   # True: all upstream bug fixes (--hyc_fixes, implies FIX_BUG=True ZERO_POINT=True)
FIX_BUG=${FIX_BUG:-False}               # intra_inter(_full_fuse) only; adahyc always has the fix
ZERO_POINT=${ZERO_POINT:-False}         # intra_inter / adahyc
BIT_PLAN=${BIT_PLAN:-}                  # adahyc: JSON {"<layer_idx>": {"<buffer>": bits}}
REBUILD_TAU=${REBUILD_TAU:-0.1}         # adahyc: < 0 disables rebuilding
SAVE_STEPS=${SAVE_STEPS:-}              # checkpoint every N optimizer steps (latest kept) + auto-resume; empty = no checkpoints
EVAL_BS=${EVAL_BS:-}                    # eval batch size; default 32 (gsm8k generation), 4 (wikitext2)
EVAL_ONLY=${EVAL_ONLY:-False}           # gsm8k: re-evaluate exp_results_da/<tag>/adapter, no training
if [ "$FIXED" = True ]; then FIX_BUG=True; ZERO_POINT=True; fi

# HyC-LoRA paper settings
iteration_threshold=5
softmax_outlier_ratio=0.05
layernorm_outlier_ratio=0.005
lr=3e-4
rank=16
lora_alpha=16

# short, Windows-safe tag, e.g. gsm8k-TinyLlama-1.1B-intra_inter-q2-fxT-fbT-zpT-pnone-t0.1-s11
model_short=$(basename "$MODEL" | sed -E 's/-intermediate-step.*//; s/-hf$//')
plan_tag=$([ -n "$BIT_PLAN" ] && basename "$BIT_PLAN" .json || echo none)
tag=${TASK}-${model_short}-${LAYER_TYPE}-q${Q_BIT}-fx${FIXED:0:1}-fb${FIX_BUG:0:1}-zp${ZERO_POINT:0:1}-p${plan_tag}-t${REBUILD_TAU}-s${SEED}
[ ${#tag} -le 90 ] || { echo "run tag too long (${#tag} > 90): $tag"; exit 1; }
out=exp_results_da/$tag
mkdir -p $out

if [ "$LAYER_TYPE" = none ]; then
    hyclora_args="--use_hyclora False --layer_type baseline"  # decoder layers dispatch to forward_old (standard HF+PEFT)
else
    hyclora_args="--use_hyclora True --layer_type $LAYER_TYPE --q_bit $Q_BIT \
    --iteration_threshold $iteration_threshold --softmax_outlier_ratio $softmax_outlier_ratio \
    --layernorm_outlier_ratio $layernorm_outlier_ratio \
    --hyc_fixes $FIXED --fix_outlier_bug $FIX_BUG --use_zero_point $ZERO_POINT --rebuild_tau $REBUILD_TAU \
    ${BIT_PLAN:+--bit_plan_path $BIT_PLAN}"
fi
if [ -n "$SAVE_STEPS" ]; then
    save_args="--save_strategy steps --save_steps $SAVE_STEPS --save_total_limit 1"
else
    save_args="--save_strategy no"
fi

if [ "$TASK" = gsm8k ]; then
    # seq 512 (run_gsm8k.py's model_max_length default), bs 4 x grad-accum 4, 6 epochs
    eval_args=$([ "$EVAL_ONLY" = True ] && echo "--is_train False --trained_adapter_path $out/adapter" || true)
    python -u run_gsm8k.py \
        --lora_init --init_lora_weights qlora \
        --model_name_or_path $MODEL --data_name gsm8k \
        --learning_rate $lr --rank $rank --lora_alpha $lora_alpha --seed $SEED \
        --output_dir $out --eval_batch_size ${EVAL_BS:-32} $eval_args \
        --num_train_epochs 6 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
        --evaluation_strategy no $save_args \
        --weight_decay 0.1 --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 10 \
        --do_train True \
        $hyclora_args \
        --report_to none 2>&1 | tee -a $out.log
elif [ "$TASK" = wikitext2 ]; then
    # paper Table 8: seq 1024, bs 4 x grad-accum 4, 3 epochs (run_wikitext2.sh ships bs 2 x 128 for llama-3-8b)
    python -u run_wikitext2.py \
        --model_name_or_path $MODEL \
        --lora_init True --lora_init_type qlora --attn_implementation eager \
        --rank $rank --lora_alpha $lora_alpha \
        $hyclora_args \
        --output_dir $out --learning_rate $lr --seed $SEED \
        --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
        --num_train_epochs 3 --per_device_train_batch_size 4 --per_device_eval_batch_size ${EVAL_BS:-4} \
        --gradient_accumulation_steps 4 $save_args \
        --weight_decay 0.1 --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 \
        --do_train True --do_eval True --block_size 1024 \
        --report_to none 2>&1 | tee -a $out.log
else
    echo "unknown TASK=$TASK (gsm8k | wikitext2)"; exit 1
fi
