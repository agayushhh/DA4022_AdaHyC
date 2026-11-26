"""Runs C-DIC on MSC session-5 episodes, streaming sessions 1-5 through one memory.

Each turn: teacher-forced NLL of gold r_t (gives PPL), greedy decode (gives BLEU, ROUGE), slot count, delta, write decision (insert/replace).
Finished episodes append rows to --out (jsonl); rerunning picks up where it left off.
Summary file (--out minus .jsonl plus _summary.json):
  headline      : mean over sessions 2..5 of per-session results, unweighted (paper Table 1 MSC, see App. O
                  Table 21: C-DIC PPL (8.553+8.418+8.350+8.404)/4 = 8.431). Session s means the session-s turns of
                  session-5 episodes; ParlAI session_s test episodes are prefixes of these (checked: s1 matches for
                  all 501, session s matches 492/486/501 of 501 for s=2/3/4), so this lines up with per-file eval.
  per_session   : same metrics per session; pooled: all turns of --eval_sessions pooled (old way)
  session5 / final_turn : only session-5 turns / last turn per episode
Metrics: PPL = exp(token-weighted mean NLL over r tokens + EOS) within a session (CHOICE).
  ROUGE-1/2/L from rouge-score with stemming: precision, recall, F1 each; paper (Sec. 4.3) calls R-1/R-2
  recall, so rouge{1,2,L} = recall (headline), *_p and *_f also logged.
  BLEU (paper: "up to 4-gram precision", variant not stated): bleu_corpus = sacrebleu corpus BLEU / 100;
  bleu_sent = mean sacrebleu sentence BLEU (exp smoothing, 4-gram) / 100.
Usage: see train.py docstring.
"""
import argparse
import json
import math
import os
import torch
from cdic import episode_stats
from data import msc_episodes, tokenize
from train import common_args, make


def aggregate(rows):
    import sacrebleu
    from rouge_score import rouge_scorer
    if not rows:
        return {}
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rg = [sc.score(r["gold"], r["gen"]) for r in rows]
    n = len(rows)
    res = dict(n_turns=n,
               ppl=math.exp(sum(r["loss"] * r["ntok"] for r in rows) / sum(r["ntok"] for r in rows)),
               bleu_corpus=sacrebleu.corpus_bleu([r["gen"] for r in rows], [[r["gold"] for r in rows]]).score / 100,
               bleu_sent=sum(sacrebleu.sentence_bleu(r["gen"], [r["gold"]]).score for r in rows) / n / 100)
    for k in ("rougeL", "rouge1", "rouge2"):
        res[k] = sum(x[k].recall for x in rg) / n
        res[k + "_p"] = sum(x[k].precision for x in rg) / n
        res[k + "_f"] = sum(x[k].fmeasure for x in rg) / n
    d = [r["delta"] for r in rows if r["delta"] is not None and math.isfinite(r["delta"])]
    res.update(slots=sum(r["n_slots"] for r in rows) / n, retrieved=sum(r["n_ret"] for r in rows) / n,
               delta=sum(d) / max(len(d), 1), insert=sum(r["write"] == "insert" for r in rows),
               replace=sum(r["write"] == "replace" for r in rows))
    return res


def main():
    ap = common_args(argparse.ArgumentParser())
    ap.add_argument("--cdic_ckpt", default=None, help="omit = ICAE-initialised compressor, zero-shot")
    ap.add_argument("--split", default="test", choices=["test", "valid"])
    ap.add_argument("--eval_sessions", default="2,3,4,5")
    ap.add_argument("--compress_with", choices=["gold", "generated"], default="gold")
    ap.add_argument("--max_new_tokens", type=int, default=128)  # CHOICE: greedy, <=128 new tokens
    a = ap.parse_args()
    keep = sorted(int(s) for s in a.eval_sessions.split(","))
    eps = msc_episodes(a.data_dir, 5, a.split)[: a.limit or None]
    rows = []
    if os.path.exists(a.out):  # resume: only keep complete episodes (rows written per finished episode)
        rows = [json.loads(l) for l in open(a.out, encoding="utf-8") if l.strip()]
        done = {r["episode"] for r in rows if r["final"]}
        rows = [r for r in rows if r["episode"] in done]
        with open(a.out + ".tmp", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        os.replace(a.out + ".tmp", a.out)
        print(f"resuming: {len(done)} episodes done")
    done = {r["episode"] for r in rows}
    if len(done) < len(eps):
        m, tok = make(a, cdic_ckpt=a.cdic_ckpt)
        m.eval()
        with open(a.out, "a", encoding="utf-8") as f, torch.no_grad():
            for e, ep in enumerate(eps):
                if e in done:
                    continue
                outs = m.episode(tokenize(tok, ep, a.max_utt_tokens), gen_tokens=a.max_new_tokens,
                                 closed_loop=a.compress_with == "generated")
                new = [dict(episode=e, turn=t, session=x["session"], final=t == len(ep) - 1, loss=o["loss"],
                            ntok=o["ntok"], n_slots=o["n_slots_before"], n_ret=len(o["retrieved"]), write=o["write"],
                            delta=o["delta"] if math.isfinite(o["delta"]) else None, gold=x["r"],
                            gen=tok.decode(o["gen"], skip_special_tokens=True).strip())
                       for t, (x, o) in enumerate(zip(ep, outs))]
                f.writelines(json.dumps(r) + "\n" for r in new)
                f.flush()
                rows += new
                st = episode_stats(outs)
                print(f"episode {e}: final_slots={st['final_slots']} insert={st['n_insert']} "
                      f"replace={st['n_replace']} loss={sum(o['loss'] for o in outs) / len(outs):.3f}", flush=True)
    per = {s: aggregate([r for r in rows if r["session"] == s]) for s in keep}
    metrics = [k for k in next(iter(per.values())) if k != "n_turns"]
    final_slots = [r["n_slots"] + (r["write"] == "insert") for r in rows if r["final"]]
    res = {"headline_mean_over_sessions": {k: sum(per[s][k] for s in keep) / len(keep) for k in metrics},
           "per_session": per,
           "pooled": aggregate([r for r in rows if r["session"] in keep]),
           "session5": aggregate([r for r in rows if r["session"] == 5]),
           "final_turn": aggregate([r for r in rows if r["final"]]),
           "mean_final_slots_per_episode": sum(final_slots) / len(final_slots),
           "args": vars(a)}
    print(json.dumps(res, indent=1))
    if res["mean_final_slots_per_episode"] < 2:
        print("WARNING: mean slots/episode < 2 -- psi collapse risk (retrieval always replaces one slot)")
    json.dump(res, open(a.out.replace(".jsonl", "") + "_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
