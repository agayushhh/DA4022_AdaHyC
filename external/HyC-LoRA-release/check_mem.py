"""Measured activation memory held by the decoder layers for backward, per HyC-LoRA / AdaHyC variant (GPU).
  python check_mem.py /path/to/TinyLlama [--plan p.json] [--out exp_results_da/check_mem-<model>.json]
Per variant: 5 calibration iterations (+1), then on a fresh iteration record memory_allocated() just before layer 0
and just after the last layer's forward. The difference = everything the layers keep alive for backward (plus one
hidden-state tensor, identical for all variants). B=4, S=512 (run_gsm8k.py's training shape). Also prints the
avg-stored-bits accounting formula (adahyc_utils.avg_stored_bits), which omits scales/rstd/lora_a/attention map."""
import json
import os
import sys

import torch

from check_da import load_model
from models.llama.adahyc_utils import buffer_numel, avg_stored_bits, plan_avg_bits

argv = sys.argv[1:]
opt = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d
path, plan, B, S = argv[0], opt('--plan'), 4, 512
model_short = os.path.basename(path.rstrip('/')).split('-intermediate-step')[0]
out_path = opt('--out', f'exp_results_da/check_mem-{model_short}.json')

from run_gsm8k import HyCLoRAArguments
model = load_model(path)
llama = model.base_model.model
cfg, layers = llama.config, llama.model.layers
ids = torch.randint(100, cfg.vocab_size, (B, S), device='cuda')

mark = {}
layers[0].register_forward_pre_hook(lambda m, a: mark.__setitem__('a', torch.cuda.memory_allocated()))
layers[-1].register_forward_hook(lambda m, a, o: mark.__setitem__('b', torch.cuda.memory_allocated()))

variants = [
    ('baseline (bf16, no compression)', dict(layer_type='baseline')),
    ('intra_inter released q2', dict(layer_type='intra_inter', q_bit=2)),
    ('intra_inter hyc_fixes q2', dict(layer_type='intra_inter', q_bit=2, hyc_fixes=True)),
    ('intra_inter hyc_fixes q4', dict(layer_type='intra_inter', q_bit=4, hyc_fixes=True)),
    ('adahyc q2 tau0.1', dict(layer_type='adahyc', q_bit=2)),
] + ([(f'adahyc q2 plan {os.path.basename(plan)}', dict(layer_type='adahyc', q_bit=2, bit_plan_path=plan))] if plan else [])
numel = buffer_numel(cfg.hidden_size, cfg.num_key_value_heads * cfg.hidden_size // cfg.num_attention_heads, cfg.intermediate_size)
rows = []
for name, ov in variants:
    args = HyCLoRAArguments(**ov)
    llama.set_fused_llama_layer(args)
    for _ in range(args.iteration_threshold + 1):
        model.zero_grad(set_to_none=True)
        model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    out = model(input_ids=ids, labels=ids, use_cache=False)
    held = mark['b'] - mark['a']
    out.loss.backward()
    del out
    fl = [l.fused_llama_layer for l in layers]
    if ov['layer_type'] == 'adahyc':
        acct = avg_stored_bits(fl, args.layernorm_outlier_ratio)
    elif ov['layer_type'] == 'baseline':
        acct = 16.0
    else:
        acct = plan_avg_bits(None, len(fl), numel, args.q_bit, args.layernorm_outlier_ratio)
    rows.append(dict(variant=name, config=ov, held_mib=held / 2 ** 20, formula_avg_bits=acct))
    print(f'{name:38s} held {held / 2 ** 20:9.1f} MiB   formula avg bits {acct:.4f}', flush=True)

base = rows[0]['held_mib']
print(f'\n=== measured activation memory held by decoder layers ({model_short}, B={B}, S={S}) ===')
for r in rows:
    r['pct_of_bf16'] = 100 * r['held_mib'] / base
    print(f"{r['variant']:38s} {r['held_mib']:9.1f} MiB  ({r['pct_of_bf16']:5.1f}% of bf16)   "
          f"formula {r['formula_avg_bits']:.4f} bits ({100 * r['formula_avg_bits'] / 16:5.1f}% of 16)")
os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
json.dump(dict(model=path, B=B, S=S, rows=rows), open(out_path, 'w'), indent=1)
print(f'wrote {out_path}')
