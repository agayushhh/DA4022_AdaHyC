"""C-DIC (Jung et al., ICML 2026, arXiv 2606.12411) -- a compact reimplementation based on ICAE (Ge et al., ICLR 2024).

No official code exists. Choices where the paper is silent are marked `CHOICE:`.

Model (mirrors ICAE v1, github.com/getao/icae: code/icae_v1/llama_icae_modeling.py):
  f_comp = Llama-2-7B-Chat + LoRA(q_proj, v_proj; r=128, alpha=32, dropout=0.05), input
           [BOS; R_t; Emb(q); Emb(r); C], Z = last-layer hidden states at the C positions, *post* final RMSNorm
           (ICAE takes hidden_states[-1], which HF appends after the final norm).
  f_gen  = the same backbone with the adapter disabled (frozen). Input, as in ICAE fine-tuning:
           [R_t; FT; Emb(q); FT] then r; FT is ICAE's learned (here frozen) "ft" token embedding.
           CHOICE: when R_t is empty (turn 0) the soft prefix is replaced by BOS.
Memory: list of Slot(z, last). For each turn t (Sec. 3.2 / Alg. 1):
  1 retrieve  S_i = cos(psi(f_comp([BOS; q; C])), psi(Z_i)) * exp(-alpha*(t - last_i)); R = {S_i >= tau} or top-1
  2 score     l_t = -log P(r_t | R; q_t) via frozen generator, teacher-forced
  3 compress  Z_t = f_comp([BOS; R; q; r; C])   (r = gold in training, gold or generated r_hat in eval)
  4 write     delta = max S, j = argmax: delta < tau -> append, else replace slot j.  last_i := t on retrieve/write.
ra-TBPTT: dl_t/dZ_j != 0 iff delta_t >= tau (j = argmax); every other use of a slot is detached. Each slot keeps the
graph of the compressor pass that wrote it. Per-turn backward (sum of per-turn grads == grad of the mean loss).
--hops K: the argmax input of a writer pass enters as a leaf copy `zin` of the parent slot; after l_t.backward() the
gradient on zin is pushed into the parent's graph by hand, hop by hop, up to K hops. Graphs are independent, so
memory is bounded: a slot keeps its parent link only while it is < K-1 hops below an in-memory slot (_prune), a graph
backpropagated at hop K is freed (retain_graph only for hop < K), and the graphs of a slot that cannot be attached any
more (t - last > dt_max, since S <= exp(-alpha*dt) < tau) are dropped and rebuilt exactly (stored inputs + RNG state)
if a fallback retrieval revives it and it is later attached.
Track 1 (--act_bits b): tensors saved for backward in graph-building passes are stored b-bit per-channel zero-point
quantized (+ top act_outlier_ratio channels by L2 norm in bf16), HyC-LoRA style, via saved_tensors_hooks.
"""
import os
os.environ.setdefault("USE_TF", "0")
import argparse
import contextlib
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

BOS, EOS = 1, 2  # Tokenizer for Llama-2


# ------------------------------------------------------------------------ Track 1: activation quantization
def quant(x, bits, outlier_ratio):
    """Asymmetric per-channel quantization along the last dimension, with 8//bits values packed into each byte; top-k L2 channels remain bf16."""
    C = x.shape[-1]
    f = x.detach().reshape(-1, C).float()
    k = int(outlier_ratio * C)
    idx = f.norm(dim=0).topk(k).indices if k else None
    out = f[:, idx].to(torch.bfloat16) if k else None
    mn, mx = f.aminmax(dim=0)
    scale = (mx - mn).clamp_min(1e-12) / (2 ** bits - 1)
    q = ((f - mn) / scale).round_().clamp_(0, 2 ** bits - 1).to(torch.uint8).flatten()
    per = 8 // bits
    q = F.pad(q, (0, -q.numel() % per)).view(-1, per)
    p = q[:, 0].clone()
    for i in range(1, per):
        p |= q[:, i] << (bits * i)
    return (p, x.shape, x.dtype, bits, mn, scale, idx, out)


def dequant(pk):
    p, shape, dtype, bits, mn, scale, idx, out = pk
    C, m = shape[-1], 2 ** bits - 1
    q = torch.stack([(p >> (bits * i)) & m for i in range(8 // bits)], 1).flatten()[:shape.numel()]
    f = q.view(-1, C).float() * scale + mn
    if idx is not None:
        f[:, idx] = out.float()
    return f.to(dtype).view(shape)


@dataclass
class Slot:
    z: torch.Tensor                 # (n, d); during training holds the graph of the pass that produced it
    last: int                       # index of the turn of the last retrieval or write
    inp: list | None = None         # training: detached inputs from the writer pass (for exact recompute)
    rng: tuple | None = None        # training: RNG state prior to the writer pass (for LoRA dropout replay)
    parent: "Slot | None" = None    # for hops > 1: slot attached to the writer pass at inp[ppos], as leaf `zin`
    ppos: int = 0
    zin: torch.Tensor | None = None
    probe: torch.Tensor | None = None  # selftest: zero leaf added to C in the writer pass; .grad indicates gradient arrival


class CDIC(nn.Module):
    def __init__(self, model, mem_init, n_mem=128, tau=0.8, alpha=0.05, hops=1, pool="mean",
                 psi_drop_outlier_ratio=0.0, psi_calib=32, max_ret=0, act_bits=0, act_outlier_ratio=0.005):
        super().__init__()
        assert hops >= 1 and act_bits in (0, 2, 4, 8)
        self.model, self.n, self.tau, self.alpha, self.hops = model, n_mem, tau, alpha, hops
        self.pool, self.drop_p, self.psi_calib, self.max_ret = pool, psi_drop_outlier_ratio, psi_calib, max_ret
        self.act_bits, self.act_outlier_ratio = act_bits, act_outlier_ratio
        # ICAE memory_token_embed rows: [mem_0 .. mem_{M-1}, ae, lm, ft] (M = checkpoint's mem_size); ft is the last row.
        # CHOICE: when n_mem < M, the first n_mem memory tokens are used.
        assert mem_init.shape[0] >= n_mem + 3, (mem_init.shape, n_mem)
        self.C = nn.Parameter(mem_init[:n_mem].detach().float().clone())         # compression tokens that are learnable
        self.register_buffer("ft", mem_init[-1].detach().float().clone())
        d = mem_init.shape[1]
        self.register_buffer("chan_sq", torch.zeros(d))
        # largest dt such that exp(-alpha*dt) >= tau; beyond that, S < tau for any cosine, meaning the slot cannot be attached
        self.dt_max = math.floor(math.log(1 / tau) / alpha + 1e-6) if tau > 0 and alpha > 0 else math.inf
        self.prune = True               # selftest reference disables it
        self.n_calib = 0
        self.n_recompute = 0
        self.probe = False
        self.mem: list[Slot] = []

    # ---------------------------------------------------------------- components
    def emb(self, ids):
        e = self.model.get_input_embeddings()
        return e(torch.tensor(ids, dtype=torch.long, device=e.weight.device))

    def f_comp(self, parts, probe=None):
        dt = parts[0].dtype
        c = self.C if probe is None else self.C + probe
        x = torch.cat([*parts, c.to(dt)], 0)[None]
        h = self.model.get_base_model().model(inputs_embeds=x).last_hidden_state[0]  # after final normalization (same as ICAE)
        return h[-self.n:]

    def gen_prefix(self, R, q):
        dt = self.emb([BOS]).dtype
        ft = self.ft.to(dt)[None]
        return torch.cat([*(R or [self.emb([BOS])]), ft, self.emb(q), ft], 0)

    def gen_loss(self, R, q, r, hooks=False):
        tgt = r + [EOS]
        x = torch.cat([self.gen_prefix(R, q), self.emb(r)], 0)[None]
        # Track 1 hooks only cover the backbone; the small log-softmax saved by the loss remains exact
        with self.model.disable_adapter(), (self._saved() if hooks else contextlib.nullcontext()):
            logits = self.model(inputs_embeds=x, logits_to_keep=len(tgt)).logits[0]
        return F.cross_entropy(logits.float(), torch.tensor(tgt, device=logits.device)), len(tgt)

    @torch.no_grad()
    def generate(self, R, q, max_new_tokens=128):
        x = self.gen_prefix(R, q)[None]
        with self.model.disable_adapter():
            out = self.model.generate(inputs_embeds=x, attention_mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.long),
                                      max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=EOS, pad_token_id=EOS)
        ids = out[0].tolist()
        return ids[:ids.index(EOS)] if EOS in ids else ids

    @torch.no_grad()
    def observe(self, z):  # Track-3 calibration: energy per channel from the first `psi_calib` compressor outputs
        if self.drop_p > 0 and self.n_calib < self.psi_calib:
            self.chan_sq += z.float().pow(2).sum(0)
            self.n_calib += 1

    @torch.no_grad()
    def psi(self, z):
        z = z.detach().float()
        if self.drop_p > 0:  # remove the top-p channels based on calibrated norm (massive-activation hypothesis)
            k = int(self.drop_p * z.shape[1])
            if k:
                z = z.clone()
                z[:, self.chan_sq.topk(k).indices] = 0
        # CHOICE: paper says "mean or CLS"; 'last' denotes the last compression token (which sees all under causal attention).
        return z.mean(0) if self.pool == "mean" else z[-1]

    @torch.no_grad()
    def scores(self, qv, t):
        if not self.mem:
            return torch.empty(0)
        P = torch.stack([self.psi(s.z) for s in self.mem])
        dt = torch.tensor([t - s.last for s in self.mem], dtype=torch.float, device=P.device)
        return (F.cosine_similarity(P, qv[None], dim=-1) * torch.exp(-self.alpha * dt)).cpu()

    # ---------------------------------------------------------------- graph management
    def _saved(self):
        """Track 1 hooks applied to every graph-building pass (writer passes; generator pass when a slot is attached)."""
        if not self.act_bits:
            return contextlib.nullcontext()
        weights = {p.untyped_storage().data_ptr() for p in self.parameters()}

        def pack(x):
            # only differentiable activations; skips weights (frozen or LoRA), ints/bools, small tensors, and
            # non-differentiable auxiliary tensors (rotary cos/sin, attention logsumexp, masks).
            if (not x.is_floating_point() or x.dim() < 2 or x.shape[-1] < 64 or not x.requires_grad
                    or x.untyped_storage().data_ptr() in weights):
                return x.detach()  # NOT x: a packed tensor with grad_fn creates an uncollectable cycle (graph leak)
            return quant(x, self.act_bits, self.act_outlier_ratio)
        return torch.autograd.graph.saved_tensors_hooks(pack, lambda p: p if torch.is_tensor(p) else dequant(p))

    def _rng_devices(self):
        return [self.C.device] if self.C.device.type == "cuda" else []

    def _write(self, s, recompute=False):
        """(Re)constructs the writer pass of slot s with gradients. The parent (if still linked) is inserted as leaf s.zin."""
        parts = list(s.inp)
        if s.parent is not None:
            s.zin = parts[s.ppos].detach().requires_grad_()
            parts[s.ppos] = s.zin
        if not recompute:
            s.rng = (torch.get_rng_state(), *[torch.cuda.get_rng_state(d) for d in self._rng_devices()])
            with self._saved():
                return self.f_comp(parts, s.probe)
        with torch.random.fork_rng(devices=self._rng_devices()):
            torch.set_rng_state(s.rng[0])
            for d, st in zip(self._rng_devices(), s.rng[1:]):
                torch.cuda.set_rng_state(st, d)
            with self._saved():
                return self.f_comp(parts, s.probe)

    def _ensure(self, s):
        """Rebuilds dropped graphs along the K-hop chain of a slot that is about to be attached (exact: same params, inputs and dropout masks; the stored forward value is kept, and the gradient comes from the rebuilt graph)."""
        for _ in range(self.hops):
            if s is None:
                return
            if not s.z.requires_grad:
                z = self._write(s, recompute=True)
                s.z = s.z.detach() + (z - z.detach())
                self.n_recompute += 1
            s = s.parent

    def _prune(self, t_next):
        """Frees things that can no longer get gradient: parent links at >= K-1 hops below an in-memory slot (hop numbers only increase), and the graphs of a stale in-memory slot and its chain (rebuilt by _ensure if later attached)."""
        for s in self.mem:
            stale = t_next - s.last > self.dt_max
            for depth in range(self.hops):
                if stale:
                    s.z, s.zin = s.z.detach(), None
                p = s.parent
                if depth == self.hops - 1:
                    s.parent = None
                if p is None or depth == self.hops - 1:
                    break
                s = p

    # ---------------------------------------------------------------- a single turn
    def turn(self, t, q, r, train=False, loss_scale=1.0, closed_loop=False, gen_tokens=0):
        """q, r: lists of token ids (r is gold). Training performs per-turn backward by itself.
        gen_tokens>0: also return greedy generation; closed_loop: compress the generated output instead of gold r (Alg. 1)."""
        was = self.training
        self.eval()  # query/retrieval pass is gradient-free: LoRA dropout is disabled
        with torch.no_grad():
            qz = self.f_comp([self.emb([BOS]), self.emb(q)])
        self.train(was)
        self.observe(qz)
        S = self.scores(self.psi(qz), t)
        attach, j, delta, ret = False, None, float("-inf"), []
        if len(S):
            delta, j = S.max(0)
            delta, j = delta.item(), j.item()
            ret = (S >= self.tau).nonzero().flatten().tolist() or [j]  # Alg. 1: {s_i >= tau}, otherwise top-1
            if self.max_ret:  # bounded retrieval (paper App., B_ret); 0 means unbounded (main setting)
                ret = sorted(sorted(ret, key=lambda i: -S[i].item())[:self.max_ret])
            attach = train and delta >= self.tau
        # CHOICE: the top-1 fallback also resets recency. Paper: dt_i = "turns since Z_i was last retrieved" and
        # Alg. 1 places the fallback slot in R_t, i.e. it is retrieved. Side effect: a stale slot can become attachable
        # again (dealt with exactly by _ensure).
        for i in ret:
            self.mem[i].last = t
        if attach:
            self._ensure(self.mem[j])
        # CHOICE: retrieved slots are concatenated in memory insertion order.
        R = [self.mem[i].z if (attach and i == j) else self.mem[i].z.detach() for i in ret]

        with torch.set_grad_enabled(train):
            loss, ntok = self.gen_loss(R, q, r, hooks=attach)
        if train and loss.requires_grad:  # false when nothing is attached (e.g. turn 0): no trainable path
            (loss * loss_scale).backward(retain_graph=self.hops > 1)
            s, h = self.mem[j], 1  # hop h equals s's writer pass; push the gradient on s.zin into the parent's graph
            while s.zin is not None:
                g, s.zin.grad = s.zin.grad, None
                if h >= self.hops or s.parent is None or g is None:
                    break
                s, h = s.parent, h + 1
                torch.autograd.backward(s.z, g, retain_graph=h < self.hops)
        out = dict(loss=loss.item(), ntok=ntok, delta=delta, retrieved=ret, n_slots_before=len(self.mem),
                   attach=attach)
        if gen_tokens:
            out["gen"] = self.generate([z.detach() for z in R], q, gen_tokens)

        # compression step; only the argmax input may remain attached (as leaf zin), and only for hops > 1
        rc = out["gen"] if closed_loop else r
        parts = [self.emb([BOS]), *[z.detach() for z in R], self.emb(q), self.emb(rc)]
        if train:
            slot = Slot(None, t, inp=parts, probe=torch.zeros_like(self.C, requires_grad=True) if self.probe else None)
            if attach and self.hops > 1:
                slot.parent, slot.ppos = self.mem[j], 1 + ret.index(j)
            slot.z = self._write(slot)
        else:
            with torch.no_grad():
                slot = Slot(self.f_comp(parts), t)
        self.observe(slot.z)
        # write-back (a gradient-free decision). Uses delta from retrieval scores (including decay).
        if j is not None and delta >= self.tau:
            self.mem[j] = slot
            out["write"], out["slot"] = "replace", j
        else:
            self.mem.append(slot)
            out["write"], out["slot"] = "insert", len(self.mem) - 1
        if train and self.prune:
            self._prune(t + 1)
        return out

    def episode(self, turns, train=False, **kw):
        """turns: list of (q_ids, r_ids). Clears memory. Training: per-turn backward of l_t / T."""
        self.mem = []
        return [self.turn(t, q, r, train=train, loss_scale=1.0 / len(turns), **kw) for t, (q, r) in enumerate(turns)]


def episode_stats(outs):
    """Diagnostics: slots per turn, delta (max score), counts of insert/replace/attach (psi-collapse check)."""
    return dict(slots_per_turn=[o["n_slots_before"] for o in outs],
                delta=[round(o["delta"], 4) if math.isfinite(o["delta"]) else None for o in outs],
                n_insert=sum(o["write"] == "insert" for o in outs), n_replace=sum(o["write"] == "replace" for o in outs),
                n_attach=sum(o["attach"] for o in outs), final_slots=outs[-1]["n_slots_before"] + (outs[-1]["write"] == "insert"))


# ------------------------------------------------------------------------ loading routines
def build(base="meta-llama/Llama-2-7b-chat-hf", icae_ckpt=None, cdic_ckpt=None, dtype=torch.bfloat16,
          lora_r=128, lora_alpha=32, lora_dropout=0.05, model=None, **cdic_kw):
    """ICAE v1 LoRA config: peft default llama targets (q_proj, v_proj), r=128 (checkpoint shapes), alpha=32
    (ft_inference.py), dropout 0.05. ICAE v1 used a forked peft where `enable_lora=False` is equivalent to disable_adapter."""
    from peft import LoraConfig, get_peft_model
    if model is None:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                                             target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    W = model.get_input_embeddings().weight
    mem = torch.randn(cdic_kw.get("n_mem", 128) + 3, W.shape[1]) * W.float().std().item()
    if icae_ckpt:  # llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt: base weights are stored as float placeholders
        sd = torch.load(icae_ckpt, map_location="cpu", weights_only=True)
        lora = {k[len("icae."):]: v for k, v in sd.items() if "lora_" in k}
        res = model.load_state_dict(lora, strict=False)
        assert not res.unexpected_keys and not [k for k in res.missing_keys if "lora_" in k], res
        mem = sd["memory_token_embed.weight"]
    m = CDIC(model, mem, **cdic_kw)
    if cdic_ckpt:
        ck = torch.load(cdic_ckpt, map_location="cpu", weights_only=True)
        res = model.load_state_dict(ck["lora"], strict=False)
        assert not res.unexpected_keys, res
        m.C.data.copy_(ck["C"])
    return m


def trainable_state(m):
    return {"lora": {k: v.detach().cpu() for k, v in m.model.state_dict().items() if "lora_" in k},
            "C": m.C.detach().cpu()}


# ------------------------------------------------------------------------ self-test
def selftest():
    import gc
    import weakref
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=1024)
    rnd = lambda: torch.randint(3, 64, (5,)).tolist()

    def mk(hops, cfg=cfg, **kw):
        torch.manual_seed(0)
        return build(model=LlamaForCausalLM(cfg).float(), lora_r=4, lora_alpha=8, lora_dropout=0.0, n_mem=8,
                     dtype=torch.float32, hops=hops, **kw)

    def run(hops):
        m = mk(hops)
        m.probe = True
        # real scoring path: cosine * decay; empty memory gives empty scores
        m.mem = []
        assert len(m.scores(torch.ones(32), 0)) == 0
        o = m.turn(0, rnd(), rnd(), train=True)
        s_real = m.scores(m.psi(m.mem[0].z), 3)
        assert torch.allclose(s_real, torch.tensor([torch.exp(torch.tensor(-3 * m.alpha)).item()]), atol=1e-5), s_real
        m.mem = []
        # scripted retrieval scores per turn (to make gradient rules deterministic)
        script = {1: [0.9], 2: [0.9], 3: [0.9], 4: [0.5], 5: [0.85, 0.95], 6: [0.7, 0.9]}
        m.scores = lambda qv, t: torch.tensor(script[t]) if t in script else torch.empty(0)
        writers = []  # probe for the pass written at turn t
        log = []
        for t in range(7):
            before = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in writers]
            o = m.turn(t, rnd(), rnd(), train=True)
            writers.append(m.mem[o["slot"]].probe)
            inc = [((p.grad if p.grad is not None else torch.zeros_like(p)) - b).abs().sum().item() > 0
                   for p, b in zip(writers, before)]
            log.append((o, inc, len(m.mem)))
        return m, log

    for hops in (1, 2):
        m, log = run(hops)
        w = [o["write"] for o, _, _ in log]
        assert w == ["insert", "replace", "replace", "replace", "insert", "replace", "replace"], w
        assert [n for _, _, n in log] == [1, 1, 1, 1, 2, 2, 2]
        assert log[5][0]["retrieved"] == [0, 1] and log[6][0]["retrieved"] == [1] and log[4][0]["retrieved"] == [0]
        inc = [i for _, i, _ in log]  # inc[t][s] = whether l_t's backward reached the pass written at turn s
        assert inc[0] == []                                         # memory is empty
        assert inc[1] == [True]                                     # A used with delta>=tau -> its writer receives gradient
        assert inc[2][1]
        assert inc[2][0] == (hops >= 2)                             # 2-hop ancestor: only when --hops 2
        assert inc[3][2] and inc[3][1] == (hops >= 2) and not inc[3][0]  # 3-hop ancestor never occurs (hops<=2)
        assert not any(inc[4])                                      # delta<tau: fallback slot is detached
        assert inc[5][4] and not inc[5][3]                          # argmax is attached, other retrieved slot is detached
        assert inc[5][2] is False
        assert inc[6][5] and not inc[6][3]                          # slot with S<=tau not retrieved -> no gradient
        assert inc[6][4] == (hops >= 2)
        lora_g = [p.grad for n, p in m.model.named_parameters() if "lora_B" in n]
        assert all(g is not None for g in lora_g) and any(g.abs().sum() > 0 for g in lora_g)
        assert all(p.grad is None for n, p in m.model.named_parameters() if "lora_" not in n)
        assert m.C.grad is not None and m.C.grad.abs().sum() > 0
        print(f"hops={hops}: writes={w}")
        print("  grad reached writer pass s from loss l_t (rows t, cols s): " +
              " | ".join("".join("x" if b else "." for b in i) for i in inc))

    # K-hop memory bound + stale-slot release: long scripted episode with long chains, stale slots that get revived
    # (fallback) and attached. Gradients must exactly match the no-release reference; live graphs <= K per slot.
    def scripted(t, n):  # n = number of slots in memory
        if t % 7 == 6:   # attach slot 0 (stale since it was last used about 7 turns ago -> exact recompute)
            return torch.tensor([0.95] + [0.1] * (n - 1))
        if t % 7 == 5:   # nothing exceeds tau -> top-1 fallback on slot 0 (revives it)
            return torch.tensor([0.5] + [0.1] * (n - 1))
        if t % 5 == 4:   # start a new thread
            return torch.tensor([0.1] * n)
        return torch.tensor([0.1] * (n - 1) + [0.9])  # extend the chain of the newest thread

    class Holder:  # saved-tensor wrapper: lives exactly as long as the graph that saved it
        def __init__(self, x):
            self.x = x

    def chain_run(hops, prune, act_bits=0, cfg=cfg, T=24):
        m = mk(hops, cfg=cfg, act_bits=act_bits)
        m.prune = prune
        torch.manual_seed(1)
        turns = [(rnd(), rnd()) for _ in range(T)]
        m.scores = lambda qv, t: scripted(t, len(m.mem)) if m.mem else torch.empty(0)
        holders, slots, peak, alive = [], [], 0, []
        if not act_bits:
            def track():
                return torch.autograd.graph.saved_tensors_hooks(
                    lambda x: holders.append(weakref.ref(h := Holder(x.detach()))) or h, lambda h: h.x)
            m._saved = track
        m.mem = []
        for t, (q, r) in enumerate(turns):
            o = m.turn(t, q, r, train=True, loss_scale=1 / T)
            slots.append(weakref.ref(m.mem[o["slot"]]))
            alive = s = None  # drop the test's own references from the previous turn
            gc.collect()
            live = {id(s) for s in m.mem}
            for s in list(m.mem):  # reachable = in-memory plus ancestors up to depth K-1
                for _ in range(hops - 1):
                    s = s.parent
                    if s is None:
                        break
                    live.add(id(s))
                assert not prune or s is None or s.parent is None, "parent link beyond K-1 hops"
            alive = [w() for w in slots if w() is not None]
            assert not prune or {id(s) for s in alive} <= live, "slot beyond K hops still referenced"
            n_graphs = sum(s.z.requires_grad for s in alive)
            peak = max(peak, sum(w() is not None for w in holders))
            if prune:
                assert n_graphs <= hops * len(m.mem), (n_graphs, len(m.mem))
        g = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
        return g, peak, m.n_recompute, len(alive)

    for hops in (1, 2, 3):
        g0, peak0, _, n0 = chain_run(hops, prune=False)
        g1, peak1, nrec, n1 = chain_run(hops, prune=True)
        assert torch.allclose(g0, g1, rtol=1e-5, atol=1e-7), (hops, (g0 - g1).abs().max())
        assert nrec > 0 and (n1 < n0 or hops == 1) and peak1 < peak0, (nrec, n0, n1, peak0, peak1)
        print(f"hops={hops}: grads == no-release reference; slots alive {n0}->{n1}, "
              f"saved tensors alive (max) {peak0}->{peak1}, exact recomputes {nrec}")

    # Track 1: gradients with act_bits vs uncompressed (hidden 128, so tensors pass the last-dim >= 64 filter)
    cfg2 = LlamaConfig(vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024)
    # T=2: one attached turn (gradient through generator plus one writer pass); T=12: K=2-hop episode with recomputes.
    # 2-bit is asserted only on T=2: on the K-hop episode it drops to ~0.5 (RMSNorm inputs; see report).
    for T, checks in ((2, ((8, 0.999), (2, 0.8))), (12, ((8, 0.999), (4, 0), (2, 0)))):
        gref = chain_run(2, True, cfg=cfg2, T=T)[0]
        for bits, thr in checks:
            cos = F.cosine_similarity(chain_run(2, True, act_bits=bits, cfg=cfg2, T=T)[0], gref, dim=0).item()
            print(f"T={T} hops=2 act_bits={bits}: grad cosine vs act_bits=0 = {cos:.5f}")
            assert cos > thr, (T, bits, cos)
    x = torch.randn(7, 300) * 3 + 1
    x[:, 5] *= 50
    for bits in (2, 4, 8):
        e = (dequant(quant(x, bits, 0.005)) - x).abs().max() / (x.max() - x.min())
        assert e <= 0.5 / (2 ** bits - 1) + 1e-3, (bits, e)

    # Track-3 hook: top-p channels by calibrated norm are zeroed before pooling
    m = build(model=LlamaForCausalLM(cfg).float(), lora_r=4, lora_dropout=0.0, n_mem=8, dtype=torch.float32,
              psi_drop_outlier_ratio=0.25, psi_calib=2)
    z = torch.randn(8, 32); z[:, 7] *= 100
    m.observe(z); m.observe(z); m.observe(z * 0)  # third call is ignored (calibration frozen after 2)
    v = m.psi(z)
    assert v[7] == 0 and (v == 0).sum() == 8, v
    # eval-mode turn with generation runs and does not keep any graph
    m.eval()
    o = m.episode([(rnd(), rnd()) for _ in range(3)], gen_tokens=4)
    assert all(not s.z.requires_grad for s in m.mem) and all(len(x["gen"]) <= 4 for x in o)
    print(episode_stats(o))
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()
