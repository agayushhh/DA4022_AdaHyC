# Pure-torch helpers for the adahyc layer (DA4022). Kept free of triton/bnb imports so check_da.py --cpu can test them.
from types import SimpleNamespace

import torch

# quantized buffers per layer (the static_value keys, minus the sparse attention map 'a')
BUFFERS = ('x', 'x_norm_1', 'q', 'k', 'v', 'o', 'x_medium', 'x_norm_2', 'gate', 'up')
OUTLIER_BUFFERS = ('x', 'x_medium')
REBUILDABLE = ('x_norm_1', 'x_medium', 'x_norm_2')


def buffer_numel(hidden, kv_dim, inter):
    """elements per token of every buffer"""
    return {'x': hidden, 'x_norm_1': hidden, 'q': hidden, 'k': kv_dim, 'v': kv_dim, 'o': hidden,
            'x_medium': hidden, 'x_norm_2': hidden, 'gate': inter, 'up': inter}


def layer_bits(plan, layer_idx, q_bit):
    """plan: parsed JSON {"<layer_idx>": {"<buffer>": bits, ..., "rebuild": [...]}} or None; missing entries
    fall back to q_bit. The optional "rebuild" list is validated here and read by layer_rebuild."""
    entry = dict((plan or {}).get(str(layer_idx), {}))
    layer_rebuild(plan, layer_idx)
    entry.pop('rebuild', None)
    assert set(entry) <= set(BUFFERS), f"unknown buffers in bit plan (layer {layer_idx}): {set(entry) - set(BUFFERS)}"
    bits = {b: int(entry.get(b, q_bit)) for b in BUFFERS}
    assert all(v in (2, 4, 8) for v in bits.values()), f"bits must be in {{2,4,8}} (layer {layer_idx}): {bits}"
    return bits


def layer_rebuild(plan, layer_idx):
    """explicit rebuild set from the plan's "rebuild" list, or None (no key: the layer decides with the tau rule)"""
    rebuild = (plan or {}).get(str(layer_idx), {}).get('rebuild')
    if rebuild is None:
        return None
    assert isinstance(rebuild, list) and set(rebuild) <= set(REBUILDABLE),         f"plan 'rebuild' must be a list within {REBUILDABLE} (layer {layer_idx}): {rebuild}"
    return frozenset(rebuild)


def rmsnorm_rebuild(x, rstd, weight):
    """same math as rmsnorm_forward: x * rstd * w in fp32; rstd is the kernel's flat (B*S,) fp32 output"""
    return (x.float() * rstd.view(*x.shape[:-1], 1) * weight.float()).to(torch.bfloat16)


def rebuild_x_medium(x, o, w_o_dequant, b_o, o_lora_a, w_o_lora_b):
    """x_medium = x + o @ W_o (+ b_o) + o_lora_a @ B_o, as lora_forward computes it; w_o_dequant is (in, out)"""
    dtype = w_o_dequant.dtype
    x_medium = x.to(dtype) + o.to(dtype) @ w_o_dequant + o_lora_a.to(dtype) @ w_o_lora_b.to(dtype)
    return x_medium if b_o is None else x_medium + b_o.to(dtype)


def rel_err(approx, true):
    true = true.float()
    return ((approx.float() - true).norm() / true.norm()).item()


def add_err(calib_err, key, rebuilt, stored, true):
    """accumulate [rebuild error, store error]; the sums share a denominator, so comparing sums == comparing means"""
    e = calib_err.setdefault(key, [0.0, 0.0])
    e[0] += rel_err(rebuilt, true)
    e[1] += rel_err(stored, true)


def decide_rebuild(calib_err, tau):
    """rebuild iff mean rebuild error <= (1 + tau) * mean store error; tau < 0 disables rebuilding"""
    if tau < 0:
        return set()
    ok = lambda key: calib_err[key][0] <= (1 + tau) * calib_err[key][1]
    rebuild = {b for b in ('x_norm_1', 'x_medium') if ok(b)}
    # x_norm_2 is rebuilt from whichever x_medium backward will see
    if ok('x_norm_2@rebuilt_x_medium' if 'x_medium' in rebuild else 'x_norm_2'):
        rebuild.add('x_norm_2')
    return rebuild


def avg_stored_bits(layers, outlier_ratio):
    """average stored bits per buffered element over the model. layers: objects with .numel, .bits, .rebuild.
    A rebuilt buffer costs 0. Outlier buffers cost bits * all channels (the packed tensor keeps the zeroed
    outlier channels) + 16 bits per outlier channel. Scales/zero points/rstd/lora_a buffers are not counted."""
    bits = elems = 0
    for layer in layers:
        for buf, n in layer.numel.items():
            elems += n
            if buf in (layer.rebuild or ()):
                continue
            bits += layer.bits[buf] * n + (16 * int(n * outlier_ratio) if buf in OUTLIER_BUFFERS else 0)
    return bits / elems


def plan_avg_bits(plan, n_layers, numel, q_bit, outlier_ratio):
    """avg_stored_bits of a plan file without a model; layers without a "rebuild" key count as rebuilding nothing
    (the tau rule decides those at run time, so the layer's printed number is the one to report)"""
    return avg_stored_bits([SimpleNamespace(numel=numel, bits=layer_bits(plan, i, q_bit), rebuild=layer_rebuild(plan, i))
                            for i in range(n_layers)], outlier_ratio)
