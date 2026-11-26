"""Probe that runs forward stats and gradient sensitivity for HyC-LoRA activation buffers (Llama).

A100 commands:
  pip install peft datasets
  python probe/probe.py --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --out_dir probe/out_tinyllama
  python probe/probe.py --model meta-llama/Llama-2-7b-hf --batch_size 2 --out_dir probe/out_llama2_7b
  # then probe/allocate.py (see its docstring). CPU self-check: python probe/probe.py --selftest

Part 1 (probe_stats.json, table on stdout): for each (layer, buffer) pair in forward mode, reports channel offset severity,
  2/4-bit relative error over sym (released) and zero-point, scale taken from sample 0 (released) or whole batch,
  overlap of top-0.5% outlier channels between x and x_medium plus the index bug error, and Track-2 rebuild error.
Part 2 (sensitivity.json = mean over --sens_batches, sensitivity_se.json = its standard error):
  s_m(b) = ||g - g_{m,b}||^2 / ||g||^2 over LoRA grads (r=16, all 7 linears), only probe group m of one
  layer perturbed in the saved-for-backward tensors. Quantizer defaults match the runtime (--quant zp,
  --scale batch): per-channel zero-point min-max over the whole batch; x and x_medium keep their top-0.5%
  L2-norm channels in bf16.
  LoRA-B is zero at init (=> zero LoRA-A grads), so by default LoRA is first trained for
  --warmup_steps=50 (AdamW lr 3e-4) on GSM8K batches disjoint from the probe batches;
  --random_lora_b N(0, 0.02) init is the cheap alternative.
Stored groups (b = bits of the buffer): x (input_layernorm), x_norm_1 (q/k/v_proj inputs), qkv_attn
  (q,k pre-RoPE + v), o (o_proj input), x_medium (post_attention_layernorm), x_norm_2 (gate/up_proj inputs),
  gate_up (gate/up outputs; silu(gate) and down_proj's input are recomputed from them as HyC-LoRA does).
  For q/k/v/gate/up only the frozen-weight part W x (HyC's *_main) is quantized, the LoRA part stays exact.
Rebuild groups (AdaHyC; b = bits of the SOURCES, the rebuilt buffer replaces the stored one):
  x_norm_1@rb   = rmsnorm(x_hat)                       (rstd from the exact x, as adahyc_utils.rmsnorm_rebuild)
  x_medium@rb   = x_hat + o_hat W_o^T + exact LoRA(o)  (x and o both at b bits)
  x_norm_2@rb   = rmsnorm(x_medium stored at b bits)
  x_norm_2@rbxm = rmsnorm(x_medium@rb at source bits b)
"""
import os
os.environ.setdefault("USE_TF", "0")
import argparse
import json
import statistics
import tempfile
from collections import defaultdict

import torch
import transformers.models.llama.modeling_llama as ml

from allocate import BUFFERS, OUTLIER_RATIO, REBUILDABLE, allocate, avg_bits, baseline, buffer_sizes, clean, model_config
from allocate import main as allocate_main

QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"
ANSWER_PROMPT = "The final answer is: "
BUFS = ["x", "x_norm_1", "q", "k", "v", "o", "x_medium", "x_norm_2", "gate", "up"]
GROUPS = ["x", "x_norm_1", "qkv_attn", "o", "x_medium", "x_norm_2", "gate_up"]
RB_GROUPS = ["x_norm_1@rb", "x_medium@rb", "x_norm_2@rb", "x_norm_2@rbxm"]
MAIN = {"self_attn.q_proj": "qkv_attn", "self_attn.k_proj": "qkv_attn", "self_attn.v_proj": "qkv_attn",
        "mlp.gate_proj": "gate_up", "mlp.up_proj": "gate_up"}  # HyC quantizes only the W x part of these


# ---------------------------------------------------------------- quantization
def fake_quant(x, bits, zp=False, sample0=True, idx=None):
    """Takes x: (B, S, C) float. Runs per-channel min-max quant then dequant like HyC-LoRA.
    sym (released): s=(max-min)/(2^b-1), q=clamp(round(x/s)). zp: adds zero point z=-round(min/s)-2^(b-1).
    idx: zeroes outlier channels before quant, restores them in bf16."""
    orig = x
    if idx is not None and len(idx):
        x = x.clone()
        x[..., idx] = 0
    ref = x[0] if sample0 else x.flatten(0, 1)
    mn, mx = ref.amin(0), ref.amax(0)
    s = ((mx - mn) / (2 ** bits - 1)).clamp_min(1e-8)
    z = -torch.round(mn / s) - 2 ** (bits - 1) if zp else 0
    y = (torch.clamp(torch.round(x / s + z), -2 ** (bits - 1), 2 ** (bits - 1) - 1) - z) * s
    if idx is not None and len(idx):
        y[..., idx] = orig[..., idx].bfloat16().float()
    return y


def topk_idx(x, sample0=True):
    """Outlier channels for HyC-LoRA: top 0.5% by channel L2 norm from sample 0 (or whole batch), sorted by norm."""
    return (x[0] if sample0 else x.flatten(0, 1)).norm(dim=0).topk(int(x.shape[-1] * OUTLIER_RATIO)).indices


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


# ---------------------------------------------------------------- data / model
def batches(args, tok, vocab):
    """Yields endless (input_ids, labels) batches shaped (B, S)."""
    B, S = args.batch_size, args.seq_len
    if tok is None:
        g = torch.Generator().manual_seed(args.seed)
        while True:
            ids = torch.randint(3, vocab, (B, S), generator=g)
            yield ids, ids.clone()
    from datasets import load_dataset
    ids, labs, rows = [], [], []
    for ex in load_dataset("openai/gsm8k", "main", split="train"):  # Prompt from run_gsm8k.py, packed to length S
        src = f"{ex['question']}{QUESTION_PROMPT}"
        tgt = f"{ex['answer']}{tok.eos_token}".replace("####", ANSWER_PROMPT)
        s, f = tok(src).input_ids, tok(src + tgt).input_ids
        ids, labs = ids + f, labs + [-100] * len(s) + f[len(s):]
        while len(ids) >= S:
            rows.append((ids[:S], labs[:S]))
            ids, labs = ids[S:], labs[S:]
            if len(rows) == B:
                yield torch.tensor([r[0] for r in rows]), torch.tensor([r[1] for r in rows])
                rows = []
    raise RuntimeError("ran out of GSM8K data")


def load(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    if args.model == "tiny":
        torch.manual_seed(args.seed)
        return ml.LlamaForCausalLM(model_config("tiny")).to(dev, dtype), None
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation="eager").to(dev)
    return model, (None if args.random_data else AutoTokenizer.from_pretrained(args.model))


# ---------------------------------------------------------------- part 1: forward stats
@torch.no_grad()
def layer_stats(layer, cap):
    f = {k: v.float() for k, v in cap.items()}
    idx = {k: topk_idx(f[k]) for k in ("x", "x_medium")}
    out = {}
    for name, t in f.items():
        flat = t.flatten(0, 1)
        r = flat.mean(0).abs() / flat.std(0).clamp_min(1e-12)
        d = dict(off_med=r.median().item(), off_p90=r.quantile(0.9).item(), off_max=r.max().item(),
                 off_frac_gt1=(r > 1).float().mean().item())
        for b in (2, 4):
            for zp in (False, True):
                for s0 in (True, False):
                    d[f"err{b}_{'zp' if zp else 'sym'}_{'s0' if s0 else 'batch'}"] = rel(fake_quant(t, b, zp, s0, idx.get(name)), t)
        out[name] = d
    ix, im = idx["x"], idx["x_medium"]
    if len(ix):  # (c) In released code, x outliers get restored at x_medium channel indices
        sx, sm = set(ix.tolist()), set(im.tolist())
        out["x"]["outlier_jaccard_vs_xmed"] = len(sx & sm) / len(sx | sm)
        out["x"]["outlier_order_agree_vs_xmed"] = (ix == im).float().mean().item()
        xb = fake_quant(f["x"], 2, False, True, ix)
        xb[..., ix] = 0
        xb[..., im] = f["x"][..., ix].bfloat16().float()
        out["x"]["err2_sym_s0_idxbug"] = rel(xb, f["x"])
    dt = cap["x"].dtype
    for zp in (False, True):  # Track 2 option (d): rebuild rather than store
        k = "zp" if zp else "sym"
        xh = fake_quant(f["x"], 2, zp, True, ix)
        n1 = layer.input_layernorm(xh.to(dt)).float()
        xm = xh + layer.self_attn.o_proj(fake_quant(f["o"], 2, zp, True).to(dt)).float()
        n2 = layer.post_attention_layernorm(xm.to(dt)).float()
        for name, rb in (("x_norm_1", n1), ("x_medium", xm), ("x_norm_2", n2)):
            out[name][f"rebuild2_{k}_s0"] = rel(rb, f[name])
    return out


@torch.no_grad()
def forward_stats(model, calib):
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    cap, hooks = {}, []

    def grab(name, output):
        return (lambda m, a, o: cap.__setitem__(name, o.detach())) if output else \
               (lambda m, a: cap.__setitem__(name, a[0].detach()))

    for i, L in enumerate(model.model.layers):
        A, M = L.self_attn, L.mlp
        for mod, name, output in [(L.input_layernorm, "x", 0), (A.q_proj, "x_norm_1", 0), (A.q_proj, "q", 1),
                                  (A.k_proj, "k", 1), (A.v_proj, "v", 1), (A.o_proj, "o", 0),
                                  (L.post_attention_layernorm, "x_medium", 0), (M.gate_proj, "x_norm_2", 0),
                                  (M.gate_proj, "gate", 1), (M.up_proj, "up", 1)]:
            hooks.append(mod.register_forward_hook(grab(name, 1)) if output else mod.register_forward_pre_hook(grab(name, 0)))

        def done(m, a, o, i=i):
            for buf, d in layer_stats(m, cap).items():
                for k, v in d.items():
                    acc[i][buf][k].append(v)
            cap.clear()
        hooks.append(L.register_forward_hook(done))
    dev = next(model.parameters()).device
    for ids, _ in calib:
        model(input_ids=ids.to(dev))
    for h in hooks:
        h.remove()
    return {str(i): {b: {k: statistics.mean(v) for k, v in d.items()} for b, d in L.items()} for i, L in acc.items()}


def print_stats(st):
    cols = ["off_med", "off_p90", "off_frac_gt1"] + [f"err{b}_{q}_{s}" for b in (2, 4) for q in ("sym", "zp") for s in ("s0", "batch")]
    print("\n| layer | buf | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 2))
    for l, d in st.items():
        for b in BUFS:
            print(f"| {l} | {b} | " + " | ".join(f"{d[b][c]:.3g}" for c in cols) + " |")
    print("\n| layer | jaccard(x,x_med top0.5%) | order agree | x err2 sym/s0 | x err2 with idx bug |"
          " xn1 direct/rebuild (sym) | x_med direct/rebuild (sym) | xn2 direct/rebuild (sym) |"
          " xn1 d/r (zp) | x_med d/r (zp) | xn2 d/r (zp) |\n|" + "---|" * 11)
    for l, d in st.items():
        x = d["x"]
        row = [x.get("outlier_jaccard_vs_xmed", float("nan")), x.get("outlier_order_agree_vs_xmed", float("nan")),
               x["err2_sym_s0"], x.get("err2_sym_s0_idxbug", float("nan"))]
        cells = [f"{v:.3g}" for v in row] + [f"{d[b][f'err2_{q}_s0']:.3g} / {d[b][f'rebuild2_{q}_s0']:.3g}"
                                             for q in ("sym", "zp") for b in ("x_norm_1", "x_medium", "x_norm_2")]
        print(f"| {l} | " + " | ".join(cells) + " |")


# ---------------------------------------------------------------- part 2: gradient sensitivity
class Q:
    """State that the saved tensor pack hook reads."""
    stack = [None]   # Current (layer, group) tag set by module pre/post hooks
    target = None    # Perturbed (layer, group), None means exact
    bits, zp, sample0 = 2, True, False
    B = S = 0
    sub = {}         # Map storage ptr to (X, R): saved tensor X gets replaced with R (recomputed from quantized buffers)
    cache = {}       # Target layer: quantized proj outputs, x / o inputs, rebuilt "rb" buffer
    proj_in = None   # Input to current LoRA projection (for peft dtype cast copy)
    param_ptrs = set()  # Parameter storages: saved weights and weight.t() views are not quantized


def fq3(t, idx_outlier=False):
    """Fake-quantizes saved tensors shaped (B,S,C) or (B*S,C); returns all others unchanged."""
    C = t.shape[-1]
    if C == 1 or t.numel() != Q.B * Q.S * C:
        return t
    x = t.reshape(Q.B, Q.S, C).float()
    idx = topk_idx(x, Q.sample0) if idx_outlier else None
    return fake_quant(x, Q.bits, Q.zp, Q.sample0, idx).to(t.dtype).reshape(t.shape)


def rms_rebuild(norm, x_hat, x):
    """adahyc_utils.rmsnorm_rebuild computes x_hat * rstd(x) * w in fp32; kernel keeps rstd of exact x."""
    rstd = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    return (x_hat.float() * rstd * norm.weight.float()).to(x.dtype)


def reg(X, R):
    """Saved tensors viewing X's storage get swapped to the same view of R."""
    Rs = torch.empty_strided(X.size(), X.stride(), dtype=X.dtype, device=X.device)
    Rs.copy_(R)
    Q.sub[X.untyped_storage().data_ptr()] = (X, Rs)


def lookup(t):
    e = Q.sub.get(t.untyped_storage().data_ptr()) if t.device.type != "meta" else None
    if e is None:
        return None
    X, R = e
    return R.as_strided(t.size(), t.stride(), t.storage_offset() - X.storage_offset())


def pack(t):
    if Q.target is None or t.untyped_storage().data_ptr() in Q.param_ptrs:
        return t
    r = lookup(t)
    if r is not None:
        return r
    l, (g, _, rb) = Q.target[0], Q.target[1].partition("@")
    if Q.stack[-1] != (l, g):
        return t
    if rb:  # Rebuilt buffer replaces the saved copy
        r = Q.cache["rb"]
        return r.to(t.dtype).reshape(t.shape) if t.numel() == r.numel() else t
    if g in ("qkv_attn", "gate_up"):  # Substitutions (sub_main) handle this fully; attention map stays exact
        return t
    return fq3(t, idx_outlier=g in ("x", "x_medium"))


def install_tags(model):
    """Module hooks that maintain Q.stack and apply HyC-LoRA recomputation substitutions."""
    hooks = []
    layers = model.get_base_model().model.layers
    paths = {"input_layernorm": "x", "self_attn": "qkv_attn", "self_attn.q_proj": "x_norm_1",
             "self_attn.k_proj": "x_norm_1", "self_attn.v_proj": "x_norm_1", "self_attn.o_proj": "o",
             "post_attention_layernorm": "x_medium", "mlp": "gate_up", "mlp.act_fn": "gate_up",
             "mlp.gate_proj": "x_norm_2", "mlp.up_proj": "x_norm_2", "mlp.down_proj": "hadamard"}
    pop = lambda m, a, o: Q.stack.pop() and None  # Hook return value would replace the output.
    for i, L in enumerate(layers):
        for path, g in paths.items():
            m = L.get_submodule(path)
            hooks += [m.register_forward_pre_hook(lambda m, a, t=(i, g): Q.stack.append(t)), m.register_forward_hook(pop)]
        for name, m in L.named_modules():
            if name.endswith("lora_B.default"):  # HyC-LoRA keeps the rank-r LoRA intermediate exact.
                hooks += [m.register_forward_pre_hook(lambda m, a: Q.stack.append(None)), m.register_forward_hook(pop)]
            if name.endswith("lora_A.default"):  # peft casts input to adapter dtype; carry SUB along.
                hooks.append(m.register_forward_pre_hook(carry_sub))
            if name.endswith("_proj"):
                hooks.append(m.register_forward_pre_hook(lambda m, a: setattr(Q, "proj_in", a[0])))
        for path, g in MAIN.items():  # Only quantize W x: capture base layer output, then substitute.
            m = L.get_submodule(path)
            hooks.append(m.base_layer.register_forward_hook(
                lambda m, a, o, t=(i, g): Q.cache.__setitem__("main", o.detach().clone()) if Q.target == t else None))
            hooks.append(m.register_forward_hook(lambda m, a, o, t=(i, g), k=path: sub_main(k, o) if Q.target == t else None))
        M = L.mlp
        hooks.append(M.act_fn.register_forward_hook(lambda m, a, o, i=i: sub_act(m, o) if Q.target == (i, "gate_up") else None))
        hooks.append(M.down_proj.register_forward_pre_hook(lambda m, a, i=i, M=M: sub_had(M.act_fn, a[0]) if Q.target == (i, "gate_up") else None))
        is_rb = lambda i: Q.target is not None and Q.target[0] == i and "@" in Q.target[1]
        O = L.self_attn.o_proj
        hooks.append(L.input_layernorm.register_forward_pre_hook(lambda m, a, i=i: rb_x(m, a[0]) if is_rb(i) else None))
        hooks.append(O.register_forward_pre_hook(lambda m, a, i=i: Q.cache.__setitem__("o", a[0]) if is_rb(i) else None))
        hooks.append(L.post_attention_layernorm.register_forward_pre_hook(
            lambda m, a, i=i, O=O: rb_xm(m, O, a[0]) if is_rb(i) else None))
    return hooks


@torch.no_grad()
def sub_main(name, out):  # Saved projection output becomes Q(W x) + exact LoRA part, like the fused layer stores.
    main = Q.cache.pop("main")
    hat = (out.float() - main.float() + fq3(main).float()).to(out.dtype)
    reg(out, hat)
    Q.cache[name] = hat


@torch.no_grad()
def sub_act(act, out):  # Recompute silu(gate) from dequantized gate.
    reg(out, act.forward(Q.cache["mlp.gate_proj"]))


@torch.no_grad()
def sub_had(act, h):  # Recompute down_proj input from dequantized gate/up.
    reg(h, act.forward(Q.cache["mlp.gate_proj"]) * Q.cache["mlp.up_proj"])


@torch.no_grad()
def rb_x(norm, x):
    Q.cache["x"] = x
    if Q.target[1] == "x_norm_1@rb":
        Q.cache["rb"] = rms_rebuild(norm, fq3(x, True), x)


@torch.no_grad()
def rb_xm(norm, o_proj, xm):
    """x_medium@rb = x_hat + o_hat W_o^T + exact LoRA(o) = x_medium + (x_hat - x) + W_o (o_hat - o)."""
    x, o, g = Q.cache["x"], Q.cache["o"], Q.target[1]
    xm_rb = (xm.float() + (fq3(x, True).float() - x.float())
             + (o_proj.base_layer(fq3(o)).float() - o_proj.base_layer(o).float())).to(xm.dtype)
    if g == "x_medium@rb":
        Q.cache["rb"] = xm_rb
    elif g == "x_norm_2@rb":
        Q.cache["rb"] = rms_rebuild(norm, fq3(xm, True), xm)
    elif g == "x_norm_2@rbxm":
        Q.cache["rb"] = rms_rebuild(norm, xm_rb, xm)


@torch.no_grad()
def carry_sub(m, a):
    x, p = a[0], Q.proj_in
    if Q.target is not None and p is not None and x.untyped_storage().data_ptr() != p.untyped_storage().data_ptr():
        r = lookup(p)
        if r is not None:
            reg(x, r.to(x.dtype))


_rope = ml.apply_rotary_pos_emb
_repeat_kv = ml.repeat_kv


def rope(q, k, cos, sin, *a, **kw):
    """HyC-LoRA stores q and k before RoPE, then reapplies RoPE in backward."""
    qe, ke = _rope(q, k, cos, sin, *a, **kw)
    if Q.target is not None and Q.stack[-1] == Q.target and Q.target[1] == "qkv_attn":
        qe, ke = qe.contiguous(), ke.contiguous()  # Thus matmul's saved (B*h,S,d) is a view of qe.
        with torch.no_grad():  # q and k are views of q/k_proj outputs substituted by sub_main.
            qh, kh = _rope(lookup(q), lookup(k), cos, sin, *a, **kw)
            reg(qe, qh)
            reg(ke, kh)
    return qe, ke


def repeat_kv(h, n):
    out = _repeat_kv(h, n)
    r = lookup(h) if Q.target is not None else None
    if r is not None and out.untyped_storage().data_ptr() != h.untyped_storage().data_ptr():
        with torch.no_grad():
            reg(out, _repeat_kv(r, n))
    return out


def lora_grad(model, params, ids, labels):
    model.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        model(input_ids=ids, labels=labels).loss.backward()
    Q.sub.clear()
    Q.cache.clear()
    return torch.cat([p.grad.float().flatten() for p in params])


def sensitivity(model, calib, warm, args):
    from peft import LoraConfig, get_peft_model
    cfg = model.config
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    dev = params[0].device
    if args.random_lora_b:
        for n, p in model.named_parameters():
            if "lora_B" in n:
                torch.nn.init.normal_(p, std=0.02)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr)
        for step in range(args.warmup_steps):
            ids, lab = next(warm)
            model.zero_grad(set_to_none=True)
            loss = model(input_ids=ids.to(dev), labels=lab.to(dev)).loss
            loss.backward()
            opt.step()
            if step % 10 == 0 or step == args.warmup_steps - 1:
                print(f"warmup step {step} loss {loss.item():.4f}")
    Q.zp, Q.sample0 = args.quant == "zp", args.scale == "sample0"
    ml.apply_rotary_pos_emb, ml.repeat_kv = rope, repeat_kv
    hooks = install_tags(model)
    Q.param_ptrs = {p.untyped_storage().data_ptr() for p in model.parameters()}
    bits = [int(b) for b in args.bits.split(",")]
    res = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for bi, (ids, lab) in enumerate(calib[:args.sens_batches]):
        ids, lab = ids.to(dev), lab.to(dev)
        Q.B, Q.S = ids.shape
        Q.target = None
        g = lora_grad(model, params, ids, lab)
        gn = g.pow(2).sum()
        print(f"batch {bi}: exact-rerun noise floor {((g - lora_grad(model, params, ids, lab)).pow(2).sum() / gn).item():.3g}")
        for l in range(cfg.num_hidden_layers):
            for grp in GROUPS + RB_GROUPS:
                for b in bits:
                    Q.target, Q.bits = (l, grp), b
                    res[str(l)][grp][str(b)].append(((g - lora_grad(model, params, ids, lab)).pow(2).sum() / gn).item())
                    Q.target = None
    for h in hooks:
        h.remove()
    ml.apply_rotary_pos_emb, ml.repeat_kv = _rope, _repeat_kv
    se = lambda v: statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan")
    agg = lambda f: {l: {g: {b: f(v) for b, v in d.items()} for g, d in G.items()} for l, G in res.items()}
    return agg(statistics.mean), agg(se)


# ---------------------------------------------------------------- main / selftest
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", help="HF name/path or 'tiny'")
    p.add_argument("--out_dir", default="probe/out")
    p.add_argument("--n_batches", type=int, default=8, help="calibration batches for forward stats")
    p.add_argument("--sens_batches", type=int, default=8, help="first k calibration batches used for sensitivity")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--random_data", action="store_true")
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--random_lora_b", action="store_true", help="N(0,0.02) LoRA-B init instead of warmup")
    p.add_argument("--bits", default="2,4,8", help="buffer bits (stored groups) / source bits (rebuild groups)")
    p.add_argument("--quant", choices=["sym", "zp"], default="zp", help="quantizer for sensitivity (zp = runtime, sym = released)")
    p.add_argument("--scale", choices=["sample0", "batch"], default="batch",
                   help="statistics (scale/zero point/outlier channels) from the whole batch (runtime) or sample 0 (released)")
    p.add_argument("--skip_stats", action="store_true")
    p.add_argument("--skip_sens", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()
    os.makedirs(a.out_dir, exist_ok=True)
    model, tok = load(a)
    model.config.use_cache = False
    model.config._attn_implementation = "eager"
    data = batches(a, tok, model.config.vocab_size)
    calib = [next(data) for _ in range(max(a.n_batches, a.sens_batches))]
    if not a.skip_stats:
        st = forward_stats(model, calib[:a.n_batches])
        json.dump(st, open(os.path.join(a.out_dir, "probe_stats.json"), "w"), indent=1)
        print_stats(st)
    if not a.skip_sens:
        sens, se = sensitivity(model, calib, data, a)
        json.dump(sens, open(os.path.join(a.out_dir, "sensitivity.json"), "w"), indent=1)
        json.dump(se, open(os.path.join(a.out_dir, "sensitivity_se.json"), "w"), indent=1)
        G = GROUPS + RB_GROUPS
        print("\n| layer | " + " | ".join(G) + " |   (mean+-se at " + a.bits + " bits)\n|" + "---|" * (len(G) + 1))
        for l, d in sens.items():
            print(f"| {l} | " + " | ".join("/".join(f"{d[g][b]:.2g}+-{se[l][g][b]:.1g}" for b in a.bits.split(",")) for g in G) + " |")


def selftest():
    t = torch.randn(2, 128, 8)
    t[..., 0] += 5  # Synthetic channel offset.
    e_sym, e_zp = rel(fake_quant(t, 2)[..., 0], t[..., 0]), rel(fake_quant(t, 2, zp=True)[..., 0], t[..., 0])
    print(f"offset channel 2-bit rel err: sym {e_sym:.3f}  zp {e_zp:.3f}")
    assert e_zp <= e_sym
    with tempfile.TemporaryDirectory() as d:
        main(["--model", "tiny", "--random_data", "--n_batches", "2", "--sens_batches", "3", "--batch_size", "2",
              "--seq_len", "128", "--warmup_steps", "20", "--out_dir", d])  # Defaults: zp, batch stats, bits 2,4,8.
        sens = json.load(open(os.path.join(d, "sensitivity.json")))
        se = json.load(open(os.path.join(d, "sensitivity_se.json")))
        assert json.load(open(os.path.join(d, "probe_stats.json")))["1"]["x_norm_1"]["rebuild2_sym_s0"] >= 0
        allocate_main(["--sens", os.path.join(d, "sensitivity.json"), "--model", "tiny", "--out", os.path.join(d, "plan.json")])
        assert json.load(open(os.path.join(d, "plan.json"))) and os.path.exists(os.path.join(d, "plan_summary.txt"))
    assert all(set(L) == set(GROUPS + RB_GROUPS) for L in sens.values()) and set(se) == set(sens)
    s = lambda b: [v[str(b)] for L in sens.values() for v in L.values()]
    print(f"max s(8)={max(s(8)):.2e}  mean s(2)={statistics.mean(s(2)):.3e}  mean s(4)={statistics.mean(s(4)):.3e}")
    assert max(s(8)) < 1e-3
    assert statistics.mean(s(2)) > statistics.mean(s(4))
    # Rebuild from 2-bit sources is worse than from 8-bit sources.
    assert all(L[g]["2"] > L[g]["8"] for L in sens.values() for g in RB_GROUPS)

    # Allocator uses monotone clamp plus 8-bit fill.
    c = clean({"0": {"g": {"2": 1.0, "4": 2.0}}})["0"]["g"]
    assert c == {2: 1.0, 4: 1.0, 8: 1.0 / 16}, c
    cfg = model_config("tiny")
    size, base = buffer_sizes(cfg), baseline(cfg)

    def check(plan, budget):
        assert set(plan) == set(sens)
        for L in plan.values():
            assert set(L["rebuild"]) <= set(REBUILDABLE)
            assert set(L) - {"rebuild"} == set(BUFFERS) - set(L["rebuild"]) and all(L[b] in (2, 4, 8) for b in L if b != "rebuild")
        assert avg_bits(plan, size) <= budget + 1e-9
    for kw in (dict(), dict(budget=2.5), dict(rebuildable=()), dict(forced=REBUILDABLE), dict(budget=2.5, rebuildable=())):
        plan, avg = allocate(sens, cfg, **kw)
        print(f"allocator {kw}: avg {avg:.4f}, rebuild {[L['rebuild'] for L in plan.values()]}")
        check(plan, kw.get("budget") or base)
        if kw.get("forced"):
            assert all(L["rebuild"] == list(REBUILDABLE) for L in plan.values())
    plan, avg = allocate(sens, cfg, rebuildable=())  # No rebuild at HyC budget equals HyC all-2-bit.
    assert all(v == 2 for L in plan.values() for b, v in L.items() if b != "rebuild") and abs(avg - base) < 1e-9
    # Synthetic sensitivities: free exact rebuild means rebuild everywhere and reinvest; useless rebuild means never.
    syn = lambda r: {l: {**{g: {"2": 1.0, "4": 0.1, "8": 0.01} for g in GROUPS}, **{g: {"2": r, "4": r, "8": r} for g in RB_GROUPS}}
                     for l in sens}
    plan, _ = allocate(syn(0.0), cfg)
    assert all(L["rebuild"] == list(REBUILDABLE) for L in plan.values()) and any(v > 2 for L in plan.values() for b, v in L.items() if b != "rebuild")
    plan, _ = allocate(syn(10.0), cfg)
    assert all(not L["rebuild"] for L in plan.values())
    # Coupling: x_norm_2 rebuilt from a rebuilt x_medium uses @rbxm entry.
    S = syn(0.0)
    for l in S:
        S[l]["x_norm_2@rbxm"] = {"2": 50.0, "4": 50.0, "8": 50.0}  # Rebuilding both is bad; x_norm_2 alone is free.
        S[l]["x_medium@rb"] = {"2": 0.0, "4": 0.0, "8": 0.0}
    plan, _ = allocate(S, cfg)
    assert all(len(set(L["rebuild"]) & {"x_medium", "x_norm_2"}) == 1 for L in plan.values()), plan
    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
