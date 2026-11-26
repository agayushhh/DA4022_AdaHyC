#!/bin/bash
# Python environment for HyC-LoRA / AdaHyC: GSM8K and WikiText-2 training, the sensitivity probe and the checks.
#
# Tested on the docker image pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel, which already provides torch 2.3.0,
# triton 2.3.0, CUDA 12.1 and gcc, on an A100 SXM4 40GB and an RTX 4090 24GB. The HyC-LoRA Triton kernels need an
# Ampere or newer GPU (bf16). Llama-2 is gated: run `huggingface-cli login` with a token that has access to
# meta-llama/Llama-2-7b-hf before the 7B experiments.
#
# Usage (from anywhere): bash scripts/setup_env.sh
set -e
pip install accelerate==0.33.0 bitsandbytes==0.43.2 datasets==2.16.1 evaluate==0.4.2 peft==0.12.0 \
    scikit_learn==1.4.0 tqdm==4.66.1 transformers==4.41.0 sentencepiece protobuf "numpy<2" scipy
# models/utils/compute_utils.py imports flash_attn.bert_padding
pip install packaging ninja
pip install flash_attn==2.5.8 --no-build-isolation
python -c "import flash_attn.bert_padding, torch, triton; assert torch.__version__.startswith('2.3.0') and torch.cuda.is_available(); print(torch.__version__, triton.__version__, torch.cuda.get_device_name(0))"
