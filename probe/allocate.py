"""Greedy allocator that picks bits or rebuild per (layer, buffer) for HyC-LoRA activation buffers (AdaHyC plan).

A100 commands (after probe.py has written sensitivity.json + sensitivity_se.json):
  python probe/allocate.py --sens probe/out_tinyllama/sensitivity.json --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --out probe/out_tinyllama/plan.json
  python probe/allocate.py --sens probe/out_llama2_7b/sensitivity.json --model meta-llama/Llama-2-7b-hf --out probe/out_llama2_7b/plan.json
  # --rebuildable "" = HyC-LoRA-style plan without rebuild; --rebuilt x_medium,... forces those rebuilds

Each buffer is stored at b in {2,4,8} bits or, for x_norm_1, x_medium and x_norm_2, rebuilt in backward at 0 bits.
A rebuilt buffer still has error: its sensitivity is what probe measured at the source bits (x_norm_1@rb at bits[x]; x_medium@rb at min(bits[x], bits[o]); x_norm_2@rb at bits[x_medium] when x_medium is stored, x_norm_2@rbxm at min(bits[x], bits[o]) when x_medium is rebuilt too).
Goal: minimize total measured sensitivity under --budget (avg stored bits via adahyc_utils.avg_stored_bits; default = HyC-LoRA all-2-bit, 2.0133 on TinyLlama).
Greedy: begin from the cheapest plan (every rebuildable buffer rebuilt, rest at 2 bits), then keep applying the single-unit move (rebuilt->stored or a bit bump, jumps allowed) with the largest drop of the layer's total sensitivity per extra stored bit, until no affordable move helps. Layer totals are recomputed after each move, so the x/o -> rebuild and x_medium -> x_norm_2 couplings are handled exactly.
Units sharing one probe measurement move together: q,k,v (qkv_attn) and gate,up (gate_up).
Sensitivities are clamped monotone (s(8) <= s(4) <= s(2), warning on violation); missing 8-bit = s(4)/16.
Output: {"<layer_idx>": {"<buffer>": bits, ..., "rebuild": [...]}}, rebuilt buffers left out of bits; plus <out>_summary.txt.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "HyC-LoRA-release", "models", "llama"))
from adahyc_utils import BUFFERS, REBUILDABLE, avg_stored_bits, buffer_numel  # noqa: E402

OUTLIER_RATIO = 0.005  # HyC-LoRA layernorm_outlier_ratio: bf16 channels of x and x_medium
LEVELS = (2, 4, 8)
UNITS = {"x": ("x",), "x_norm_1": ("x_norm_1",), "qkv_attn": ("q", "k", "v"), "o": ("o",),
         "x_medium": ("x_medium",), "x_norm_2": ("x_norm_2",), "gate_up": ("gate", "up")}
RB_NEEDS = {"x_norm_1": ["x_norm_1@rb"], "x_medium": ["x_medium@rb"], "x_norm_2": ["x_norm_2@rb", "x_norm_2@rbxm"]}


def model_config(name):
    from transformers import AutoConfig, LlamaConfig
    if name == "tiny":
        return LlamaConfig(hidden_size=256, intermediate_size=688, num_hidden_layers=2, num_attention_heads=4,
                           num_key_value_heads=2, vocab_size=1000, max_position_embeddings=512,
                           attn_implementation="eager")
    return AutoConfig.from_pretrained(name)


def buffer_sizes(cfg):
    """Per-token element count for each buffer."""
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return buffer_numel(cfg.hidden_size, cfg.num_key_value_heads * hd, cfg.intermediate_size)


def avg_bits(plan, size):
    """adahyc_utils.avg_stored_bits of a plan {layer: {buffer: bits, ..., "rebuild": [...]}}."""
    return avg_stored_bits([SimpleNamespace(numel=size, bits=L, rebuild=L.get("rebuild", ())) for L in plan.values()],
                           OUTLIER_RATIO)


def baseline(cfg, q_bit=2):
    """Average bits of HyC-LoRA with every buffer stored at q_bit (the default budget)."""
    return avg_bits({0: {b: q_bit for b in BUFFERS}}, buffer_sizes(cfg))


def clean(sens, se=None):
    """{l: {group: {bits: s}}} with int bits, 8-bit filled in and s monotone non-increasing in bits. Idempotent."""
    out = {}
    for l, G in sens.items():
        for g, d in G.items():
            d = {int(b): v for b, v in d.items()}
            for lo, hi in zip(LEVELS, LEVELS[1:]):
                if hi == 8 and 8 not in d:  # assumption: 4 more bits cut s 16x (quant-noise theory: 256x; conservative)
                    d[8] = d[4] / 16
                if d[hi] > d[lo]:
                    noise = f" (se {se[l][g].get(str(hi), float('nan')):.1g} / {se[l][g].get(str(lo), float('nan')):.1g})" if se else ""
                    print(f"warning: layer {l} {g}: s({hi})={d[hi]:.3g} > s({lo})={d[lo]:.3g}{noise}; clamped to s({lo})")
                    d[hi] = d[lo]
            out.setdefault(l, {})[g] = d
    return out


def layer_sens(S, st):
    """Total measured sensitivity of one layer. S: clean(...)[layer]; st: {buffer: bits}, 0 = rebuilt."""
    src = min(st["x"], st["o"])
    tot = S["x"][st["x"]] + S["o"][st["o"]] + S["qkv_attn"][st["q"]] + S["gate_up"][st["gate"]]
    tot += S["x_norm_1"][st["x_norm_1"]] if st["x_norm_1"] else S["x_norm_1@rb"][st["x"]]
    tot += S["x_medium"][st["x_medium"]] if st["x_medium"] else S["x_medium@rb"][src]
    if st["x_norm_2"]:
        tot += S["x_norm_2"][st["x_norm_2"]]
    else:
        tot += S["x_norm_2@rb"][st["x_medium"]] if st["x_medium"] else S["x_norm_2@rbxm"][src]
    return tot


def to_plan(st):
    return {**{b: v for b, v in st.items() if v}, "rebuild": [b for b in REBUILDABLE if not st[b]]}


def allocate(sens, cfg, budget=None, rebuildable=REBUILDABLE, forced=(), se=None):
    """Returns (plan, avg stored bits). budget=None -> the all-2-bit HyC-LoRA baseline."""
    size = buffer_sizes(cfg)
    budget = baseline(cfg) if budget is None else budget
    S = clean(sens, se)
    layers = sorted(S, key=int)
    for b in forced:
        assert all(k in S[l] for l in layers for k in RB_NEEDS[b]), f"--rebuilt {b}: rebuild sensitivity not in {sorted(S[layers[0]])}"
    missing = {b for b in rebuildable if not all(k in S[l] for l in layers for k in RB_NEEDS[b])}
    if missing - set(forced):
        print(f"warning: no rebuild measurements for {sorted(missing)} (old sensitivity.json?) -> stored only")

    def levels(unit):
        b = UNITS[unit][0]
        return (0,) if b in forced else (0,) + LEVELS if b in rebuildable and b not in missing else LEVELS

    def cost(st):  # stored bits per token of one layer
        return avg_bits({0: to_plan(st)}, size) * sum(size.values())

    st = {l: {b: levels(u)[0] for u, bufs in UNITS.items() for b in bufs} for l in layers}
    cap = budget * len(layers) * sum(size.values())
    used = sum(cost(st[l]) for l in layers)
    if used > cap + 1e-6:
        print(f"warning: budget {budget} below the cheapest plan {used / cap * budget:.4f}")
    while True:
        best = None
        for l in layers:
            c0, s0 = cost(st[l]), layer_sens(S[l], st[l])
            for u, bufs in UNITS.items():
                for nb in levels(u):
                    if nb <= st[l][bufs[0]]:
                        continue
                    new = {**st[l], **{b: nb for b in bufs}}
                    dc, ds = cost(new) - c0, s0 - layer_sens(S[l], new)
                    if ds > 0 and used + dc <= cap + 1e-6 and (best is None or ds / dc > best[0]):
                        best = (ds / dc, l, new, dc)
        if best is None:
            break
        _, l, st[l], dc = best
        used += dc
    plan = {l: to_plan(st[l]) for l in layers}
    return plan, avg_bits(plan, size)


def summary(plan, sens, cfg, budget):
    S, base = clean(sens), baseline(cfg)
    state = lambda L: {b: L.get(b, 0) for b in BUFFERS}
    s_plan = sum(layer_sens(S[l], state(L)) for l, L in plan.items())
    s_hyc = sum(layer_sens(S[l], {b: 2 for b in BUFFERS}) for l in plan)
    lines = [f"all-2-bit HyC-LoRA baseline avg bits {base:.4f} | budget {budget:.4f} | plan avg bits {avg_bits(plan, buffer_sizes(cfg)):.4f}",
             f"predicted total sensitivity: plan {s_plan:.4g} vs HyC all-2-bit {s_hyc:.4g}",
             "layers rebuilding x_norm_1/x_medium/x_norm_2: "
             + "/".join(str(sum(b in L["rebuild"] for L in plan.values())) for b in REBUILDABLE) + f" of {len(plan)}"]
    for l, L in plan.items():
        up = {b: v for b, v in L.items() if b != "rebuild" and v > 2}
        lines.append(f"layer {l:>2}: rebuild {','.join(L['rebuild']) or '-':<26} upgraded {up or '-'}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sens", required=True)
    p.add_argument("--se", default=None, help="standard errors (default: sensitivity_se.json next to --sens, if present)")
    p.add_argument("--model", required=True, help="HF name/path for the config, or 'tiny'")
    p.add_argument("--budget", type=float, default=None,
                   help="average stored bits/element; default = all-2-bit HyC-LoRA baseline under the same accounting")
    p.add_argument("--rebuildable", default=",".join(REBUILDABLE), help="buffers the allocator may rebuild ('' = none)")
    p.add_argument("--rebuilt", default="", help="comma list of buffers forced to be rebuilt, e.g. x_norm_1,x_medium,x_norm_2")
    p.add_argument("--out", default="plan.json")
    a = p.parse_args(argv)
    rebuildable = [b for b in a.rebuildable.split(",") if b]
    forced = [b for b in a.rebuilt.split(",") if b]
    assert set(rebuildable) | set(forced) <= set(REBUILDABLE), (rebuildable, forced)
    se_path = a.se or os.path.join(os.path.dirname(a.sens), "sensitivity_se.json")
    se = json.load(open(se_path)) if os.path.exists(se_path) else None
    sens = clean(json.load(open(a.sens)), se)  # warn once; clean() is idempotent
    cfg = model_config(a.model)
    budget = baseline(cfg) if a.budget is None else a.budget
    plan, avg = allocate(sens, cfg, budget, rebuildable, forced, se)
    json.dump(plan, open(a.out, "w"), indent=1)
    text = summary(plan, sens, cfg, budget)
    open(os.path.splitext(a.out)[0] + "_summary.txt", "w").write(text + "\n")
    print(text + f"\n-> {a.out}")


if __name__ == "__main__":
    main()
