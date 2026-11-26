"""How to train C-DIC on MSC. Only LoRA and the compression tokens C get trained; generator stays frozen; one episode per batch.

On an A100 80GB (you need HF access to meta-llama/Llama-2-7b-chat-hf, run `huggingface-cli login` first):
  pip install "transformers==4.52.4" "peft>=0.17" sacrebleu rouge-score
  wget -P ckpt https://huggingface.co/sggetao/icae/resolve/main/llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt
  for s in 42 43 44; do
    python cdic/train.py --icae_ckpt ckpt/llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt --out runs/cdic_s$s --seed $s
    python cdic/eval.py  --icae_ckpt ckpt/llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt --cdic_ckpt runs/cdic_s$s/last.pt --out runs/cdic_s$s/eval_test.jsonl
  done
  # zero-shot baseline from ICAE init: run eval.py without --cdic_ckpt; closed loop: eval.py ... --compress_with generated
  # quick test: add --limit 5 to both commands. Track 1: --hops 2|3 --act_bits 2. Track 3: --psi_drop_outlier_ratio 0.005
To resume, just run the same command again (it loads runs/.../last.pt)."""
import argparse
import json
import os
import random
import time
import torch
from transformers import AutoTokenizer
from cdic import build, episode_stats, trainable_state
from data import msc_episodes, tokenize


def common_args(ap):
    ap.add_argument("--base", default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--icae_ckpt", default=None)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--pool", choices=["mean", "last"], default="mean")
    ap.add_argument("--psi_drop_outlier_ratio", type=float, default=0.0)
    ap.add_argument("--psi_calib", type=int, default=32, help="#compressor outputs used to calibrate channel norms")
    ap.add_argument("--max_ret", type=int, default=0, help="bounded retrieval B_ret; 0 = unbounded (paper main)")
    ap.add_argument("--max_utt_tokens", type=int, default=128)
    ap.add_argument("--act_bits", type=int, default=0, choices=[0, 2, 4, 8], help="Track 1: saved-activation bits; 0=off")
    ap.add_argument("--act_outlier_ratio", type=float, default=0.005, help="Track 1: top-L2 channels kept in bf16")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    return ap


def make(a, **kw):
    torch.manual_seed(a.seed); random.seed(a.seed)
    m = build(a.base, a.icae_ckpt, tau=a.tau, alpha=a.alpha, hops=a.hops, pool=a.pool, max_ret=a.max_ret,
              psi_drop_outlier_ratio=a.psi_drop_outlier_ratio, psi_calib=a.psi_calib,
              act_bits=a.act_bits, act_outlier_ratio=a.act_outlier_ratio, **kw).to(a.device)
    tok = AutoTokenizer.from_pretrained(a.base)
    return m, tok


def main():
    ap = common_args(argparse.ArgumentParser())
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.0)  # CHOICE: paper doesn't say; use constant LR, no warmup
    ap.add_argument("--save_every", type=int, default=50, help="episodes between (atomic) checkpoints")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m, tok = make(a)
    eps = msc_episodes(a.data_dir, 4, "train")[: a.limit or None]
    eps = [tokenize(tok, e, a.max_utt_tokens) for e in eps]
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    last, start = os.path.join(a.out, "last.pt"), (0, 0)
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=True)
        m.model.load_state_dict(ck["lora"], strict=False); m.C.data.copy_(ck["C"])
        opt.load_state_dict(ck["opt"]); start = (ck["epoch"], ck["step"])
        print("resumed at", start)
    m.train()
    log = open(os.path.join(a.out, "train_log.jsonl"), "a")
    cuda = torch.cuda.is_available()
    final_slots = []
    for ep in range(start[0], a.epochs):
        order = list(range(len(eps)))
        random.Random(a.seed + ep).shuffle(order)
        for step in range(start[1] if ep == start[0] else 0, len(order)):
            t0, nrec = time.time(), m.n_recompute
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            opt.zero_grad(set_to_none=True)
            outs = m.episode(eps[order[step]], train=True)
            opt.step()
            m.mem = []  # release all slot graphs
            st = episode_stats(outs)
            final_slots.append(st["final_slots"])
            rec = dict(epoch=ep, step=step, loss=sum(o["loss"] for o in outs) / len(outs), turns=len(outs),
                       sec=round(time.time() - t0, 2), recomputes=m.n_recompute - nrec,
                       peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else 0, **st)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if (step + 1) % a.save_every == 0 or step + 1 == len(order):
                nxt = (ep, step + 1) if step + 1 < len(order) else (ep + 1, 0)
                torch.save({**trainable_state(m), "opt": opt.state_dict(), "epoch": nxt[0], "step": nxt[1],
                            "args": vars(a)}, last + ".tmp")
                os.replace(last + ".tmp", last)  # atomic: if a box is lost, you lose at most save_every episodes
                mean_slots = sum(final_slots) / len(final_slots)
                if mean_slots < 2:
                    print(f"WARNING: mean slots/episode {mean_slots:.2f} < 2 -- psi collapse risk (everything "
                          "replaces one slot); check delta in train_log.jsonl")
                print({k: v for k, v in rec.items() if k not in ("slots_per_turn", "delta")}, f"mean_slots={mean_slots:.2f}")


if __name__ == "__main__":
    main()
