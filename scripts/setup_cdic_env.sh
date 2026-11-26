#!/bin/bash
# C-DIC needs a newer transformers/peft than HyC-LoRA, so it gets its own venv.
# transformers 4.52 breaks with torch 2.3, hence the 4.51.3 pin. System torch is reused.
# Fetches the public ICAE checkpoint (Llama-2-7B-Chat, 271 MB) into ckpt/ at repo root.
#
# Usage: bash scripts/setup_cdic_env.sh        (venv location: $CDIC_VENV, default ~/cdic_venv)
set -e
cd "$(dirname "$0")/.."
V=${CDIC_VENV:-$HOME/cdic_venv}
[ -x $V/bin/python ] || python -m venv --system-site-packages $V
$V/bin/pip install "transformers==4.51.3" "peft>=0.17" sacrebleu rouge-score
$V/bin/python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('sggetao/icae', 'llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt', local_dir='ckpt'))"
