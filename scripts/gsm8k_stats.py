"""GSM8K scores and paired McNemar tests (see Tables 1, 4 and 5), computed from the per-question prediction files.

Usage (from the repository root):
  python scripts/gsm8k_stats.py                                            # our results, results/gsm8k
  python scripts/gsm8k_stats.py external/HyC-LoRA-release/exp_results_da   # your own reruns

Pairs each AdaHyC run with the HyC-LoRA-fixed run for the same model and seed. Uses the exact two-sided McNemar
test: if b questions are solved only by AdaHyC and c only by HyC-LoRA fixed, p = P(split at least as uneven as
(b, c) | Binomial(b + c, 1/2)). Pooled rows sum b and c over seeds.
"""
import json
import re
import statistics
import sys
from math import comb
from pathlib import Path

TAG = re.compile(r"gsm8k-(?P<model>.+)-(?P<layer>adahyc|intra_inter)-q2-fxT-.*-s(?P<seed>\d+)$")


def mcnemar_p(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def correct(run_dir, mode):
    f = run_dir / f"predictions_{mode}.json"
    return [p["correct"] for p in json.loads(f.read_text())["predictions"]] if f.exists() else None


def main(root):
    runs = {}
    for d in sorted(Path(root).iterdir()):
        m = TAG.match(d.name)
        if d.is_dir() and m:
            runs[(m["model"], m["layer"], int(m["seed"]))] = d
    for model in sorted({k[0] for k in runs}):
        print(f"\n=== {model}")
        for mode in ("sampled", "greedy"):
            acc = {"intra_inter": [], "adahyc": []}
            pooled_b = pooled_c = 0
            print(f"  {mode}:  seed  HyC-fixed  AdaHyC   only-Ada/only-fixed   p")
            for seed in sorted({k[2] for k in runs if k[0] == model}):
                fx, ad = (correct(runs[(model, l, seed)], mode) if (model, l, seed) in runs else None
                          for l in ("intra_inter", "adahyc"))
                row = f"           {seed:>4}"
                for name, c in (("intra_inter", fx), ("adahyc", ad)):
                    if c:
                        acc[name].append(100 * sum(c) / len(c))
                        row += f"  {acc[name][-1]:>8.2f}"
                    else:
                        row += f"  {'-':>8}"
                if fx and ad:
                    b = sum(a and not f for a, f in zip(ad, fx))
                    c = sum(f and not a for a, f in zip(ad, fx))
                    pooled_b, pooled_c = pooled_b + b, pooled_c + c
                    row += f"   {b:>4} / {c:<4}           {mcnemar_p(b, c):.2g}"
                print(row)
            for name, label in (("intra_inter", "HyC-fixed"), ("adahyc", "AdaHyC")):
                v = acc[name]
                if len(v) > 1:
                    print(f"           mean {label}: {statistics.mean(v):.2f} +- {statistics.stdev(v):.2f} (n={len(v)})")
            if pooled_b + pooled_c and len(acc["adahyc"]) > 1:
                print(f"           pooled: {pooled_b} / {pooled_c}, p = {mcnemar_p(pooled_b, pooled_c):.2g}")


if __name__ == "__main__":
    assert abs(mcnemar_p(131, 80) - 0.000546) < 2e-5  # seed-11 sampled-decoding value from the report
    main(sys.argv[1] if len(sys.argv) > 1 else "results/gsm8k")
