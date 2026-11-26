#!/bin/bash
# Peak GPU memory for C-DIC training on 5 MSC episodes (Table 7, Figure 6): one hop as in the paper, three hops,
# and three hops with 4-bit activation compression. Needs a 40GB+ GPU (uncompressed runs peak around 26 GB),
# Hugging Face access to meta-llama/Llama-2-7b-chat-hf, and scripts/setup_cdic_env.sh run first.
# The MSC data (ParlAI msc_v0.1) downloads into data/ on first use.
# Outputs: runs/mem_*/train_log.jsonl (peak_gb per episode); our results are in results/cdic_memory/.
set -e
cd "$(dirname "$0")/.."
PY=${CDIC_PY:-${CDIC_VENV:-$HOME/cdic_venv}/bin/python}
CK=ckpt/llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt

$PY cdic/train.py --icae_ckpt $CK --out runs/mem_h1a0 --limit 5 --epochs 1 --save_every 5
$PY cdic/train.py --icae_ckpt $CK --out runs/mem_h3a0 --limit 5 --epochs 1 --save_every 5 --hops 3
$PY cdic/train.py --icae_ckpt $CK --out runs/mem_h3a4 --limit 5 --epochs 1 --save_every 5 --hops 3 --act_bits 4
