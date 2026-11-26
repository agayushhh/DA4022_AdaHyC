# AdaHyC: sensitivity-guided store-or-rebuild allocation for two-bit activation compression

Course project for DA4022, IIT Guwahati.
Ayush Agarwal, Nitish Kumar Singh, Kunj Aggarwal, Mithil Modi, Onkar Kawde, Snehil Sai Pawar.

HyC-LoRA (MLSys 2025) cuts the activation memory of QLoRA fine-tuning by storing buffered activations at about two
bits, but it gives every buffer in every layer the same precision. AdaHyC keeps HyC-LoRA's memory budget and changes
how it is spent: three buffers per decoder layer are recomputed in the backward pass instead of stored, the gradient
sensitivity of every remaining buffer is measured once before training, and a greedy allocator moves the freed bits
to the buffers where they reduce gradient error the most. The store-or-rebuild decision is borrowed from C-DIC
(ICML 2026), which decides per dialogue turn whether to write a new memory slot or revise an existing one. In the other
direction, we apply HyC-style compression inside C-DIC training to make deeper truncated backpropagation affordable.

While building on HyC-LoRA we found and fixed five defects in its released code. The corrected method is reported as
a separate baseline ("HyC-LoRA fixed") so that the effect of the fixes is not credited to AdaHyC.

The full write-up is [docs/final_report.pdf](docs/final_report.pdf).

## Results

GSM8K accuracy (%) with two-bit activation compression, sampled decoding as in the HyC-LoRA paper.

| Model | HyC-LoRA (paper) | HyC-LoRA fixed (ours) | AdaHyC (ours) | QLoRA, 16-bit activations (paper) |
|---|---|---|---|---|
| TinyLlama-1.1B, seeds 11–14 | 12.96 | 15.47 ± 0.59 | **17.15 ± 1.37** | 15.85 |
| Llama-2-7B, seed 11 | 35.86 | not run | **36.92** | 38.82 |

- AdaHyC and HyC-LoRA fixed hold the same activation memory, measured in bytes: 650.9 vs 643.6 MiB on TinyLlama and
  1419.6 vs 1426.2 MiB on Llama-2-7B (batch 4 × 512 tokens).
- Over the four TinyLlama seeds, AdaHyC beats the fixed baseline in a paired McNemar test (p = 0.002 sampled,
  p = 0.043 greedy), but not on every seed: seeds 11 and 12 are clear wins, seeds 13 and 14 are ties.
- The Llama-2-7B result is a single run that was resumed from a checkpoint; its margin over the paper is within one
  standard error.
- With 4-bit activation compression, three-hop C-DIC training peaks at 17.5 GB, 31% below the 25.4 GB of C-DIC's
  published one-hop training. This is a memory measurement only; quality was not evaluated.

All limitations are discussed in Section 6 of the report.

## Trained adapters

The trained LoRA adapters of every run can be found in the
[`adapters-v1` release](https://github.com/agayushhh/DA4022_AdaHyC/releases/tag/adapters-v1). They are too
large for the git tree (up to 577 MB each), so each run's full `adapter/` folder is attached there as a zip.

| Task | Model | Method | Seed | Download |
|---|---|---|---|---|
| GSM8K | Llama-2-7B | AdaHyC | 11 | [zip](https://github.com/agayushhh/DA4022_AdaHyC/releases/download/adapters-v1/gsm8k_Llama-2-7b_adahyc_seed11.zip) |
| GSM8K | TinyLlama-1.1B | AdaHyC | 11 | [zip](https://github.com/agayushhh/DA4022_AdaHyC/releases/download/adapters-v1/gsm8k_TinyLlama-1.1B_adahyc_seed11.zip) |
| GSM8K | TinyLlama-1.1B | AdaHyC | 12 | [zip](https://github.com/agayushhh/DA4022_AdaHyC/releases/download/adapters-v1/gsm8k_TinyLlama-1.1B_adahyc_seed12.zip) |
| GSM8K | TinyLlama-1.1B | AdaHyC | 13 | [zip](https://github.com/agayushhh/DA4022_AdaHyC/releases/download/adapters-v1/gsm8k_TinyLlama-1.1B_adahyc_seed13.zip) |
| GSM8K | TinyLlama-1.1B | AdaHyC | 14 | [zip](https://github.com/agayushhh/DA4022_AdaHyC/releases/download/adapters-v1/gsm8k_TinyLlama-1.1B_adahyc_seed14.zip) |
| GSM8K | TinyLlama-1.1B | HyC-LoRA fixed | 11 | [zip](https://github.com/agayushhh/DA4022_AdaHyC/releases/download/adapters-v1/gsm8k_TinyLlama-1.1B_hyclora-fixed_seed11.zip) |
| GSM8K | TinyLlama-1.1B | HyC-LoRA fixed | 12 | [zip](https://github.com/agayushhh/DA4022_AdaHyC/releases/download/adapters-v1/gsm8k_TinyLlama-1.1B_hyclora-fixed_seed12.zip) |
| GSM8K | TinyLlama-1.1B | HyC-LoRA fixed | 13 | [zip](https://github.com/agayushhh/DA4022_AdaHyC/releases/download/adapters-v1/gsm8k_TinyLlama-1.1B_hyclora-fixed_seed13.zip) |
| WikiText-2 | TinyLlama-1.1B | AdaHyC | 11 | [zip](https://github.com/agayushhh/DA4022_AdaHyC/releases/download/adapters-v1/wikitext2_TinyLlama-1.1B_adahyc_seed11.zip) |
| WikiText-2 | TinyLlama-1.1B | HyC-LoRA fixed | 11 | [zip](https://github.com/agayushhh/DA4022_AdaHyC/releases/download/adapters-v1/wikitext2_TinyLlama-1.1B_hyclora-fixed_seed11.zip) |

The HyC-LoRA fixed run for TinyLlama seed 14 has no adapter because its weights were not saved; its metrics and
per-question predictions are complete in `results/gsm8k/`. Each zip unpacks to `<run>/adapter/` and loads with
`peft.PeftModel.from_pretrained(base_model, "<run>/adapter")` on top of the base model named in `adapter_config.json`.

## Repository layout

```
├── README.md
├── docs/                      final_report.pdf (NeurIPS 2026 style), mid-term report, topic finalization
├── external/
│   ├── HyC-LoRA-release/      fork of the official HyC-LoRA code with our fixes and the AdaHyC layer
│   └── hyclora_da4022.patch   every change we made to HyC-LoRA, as a diff against upstream 62ac396
├── probe/                     sensitivity probe and allocator
│   ├── probe.py, allocate.py
│   └── out_tinyllama/, out_llama2_7b/   sensitivities and the plans used in every AdaHyC run
├── cdic/                      C-DIC re-implementation (no official code exists), with K-hop ra-TBPTT and
│                              activation compression
├── scripts/                   environment setup and one script per experiment (see below)
└── results/                   everything we measured
    ├── gsm8k/                 one folder per run: metrics.json, predictions_{sampled,greedy}.json,
    │                          trainer_state.json, adapter/adapter_config.json; plus the run's log
    │                          (adapter weights: see "Trained adapters" above)
    ├── wikitext2/             same, for WikiText-2
    ├── memory/                measured activation memory (check_mem output)
    ├── cdic_memory/           C-DIC peak-memory logs (peak_gb per episode)
    ├── logs/                  logs of the gradient check, both probes and the C-DIC memory test
    └── early_v1/              logs of the first runs, made before the code audit (report Appendix E)
```

### What we changed in HyC-LoRA

The fork lives in `external/HyC-LoRA-release` and is based on upstream commit `62ac396`;
`external/hyclora_da4022.patch` shows every modification. All fixes are behind the flag `--hyc_fixes` (`FIXED=True` in
`run_da.sh`), so the released behaviour can still be reproduced.

| File | Change |
|---|---|
| `models/llama/fused_llama_layer_adahyc.py`, `models/llama/adahyc_utils.py` | new: the AdaHyC decoder layer, rebuild functions, plan reading, bit accounting |
| `operators/rmsnorm_kernels.py`, `operators/rmsnorm_updatable_kernels.py` | fix 1: RMSNorm backward used the LayerNorm formula |
| `operators/compress_function.py` | fixes 2 and 3: calibration merge of outlier index and scales; statistics from all samples |
| `models/llama/fused_llama_layer_intra_inter*.py`, `..._intra.py`, `..._baseline.py` | fix 4 (outlier index), fix 5 (zero point), flag plumbing |
| `models/llama/modeling_llama.py` | registers the `adahyc` layer type and the new options |
| `run_gsm8k.py` | seed set before LoRA creation; greedy evaluation and per-question output; bf16 recast when resuming |
| `run_wikitext2.py` | the same HyC/AdaHyC options; test-set perplexity, peak memory and a metrics file; resumes only when checkpointing is enabled |
| `run_da.sh` | new: launcher with the paper's settings, driven by environment variables |
| `check_da.py`, `check_mem.py` | new: gradient check against true QLoRA, measured activation memory |

## Reproducing the results

### Requirements

- An NVIDIA GPU of the Ampere generation or newer. We used an A100 40GB and an RTX 4090 24GB. TinyLlama runs and the
  Llama-2-7B GSM8K run fit in 24 GB; the C-DIC memory test needs 40 GB.
- The docker image `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel` (torch 2.3.0, triton 2.3.0, CUDA 12.1), or an
  equivalent environment.
- For anything involving Llama-2, a Hugging Face token with access to `meta-llama/Llama-2-7b-hf` and
  `meta-llama/Llama-2-7b-chat-hf`: `huggingface-cli login`.
- Datasets (GSM8K, WikiText-2, MSC) and models download automatically on first use.

### Steps

Run everything from the repository root.

```bash
bash scripts/setup_env.sh               # HyC-LoRA / AdaHyC environment
bash scripts/setup_cdic_env.sh          # only for the C-DIC experiment: separate venv + ICAE checkpoint

bash scripts/01_checks.sh               # gradient fidelity (Table 6) and measured memory (Table 3)
bash scripts/02_probe.sh                # optional: re-run the probe and allocator (writes probe/rerun_*)
bash scripts/03_gsm8k_tinyllama.sh      # Tables 1 and 5: fixed vs AdaHyC, seeds 11-14
bash scripts/04_gsm8k_llama2_7b.sh      # Table 4
bash scripts/05_wikitext2.sh            # Appendix C.2
bash scripts/06_cdic_memory.sh          # Table 7

python scripts/gsm8k_stats.py external/HyC-LoRA-release/exp_results_da   # accuracies and McNemar tests
```

Training runs write to `external/HyC-LoRA-release/exp_results_da/`, which starts empty, so a rerun never overwrites
or resumes from our results. `python scripts/gsm8k_stats.py` without an argument reproduces Tables 1 and 5 from
the predictions we saved in `results/gsm8k/`.

Things to know when comparing a rerun with our numbers:

- The training scripts use the bit plans we trained with (`probe/out_*/plan.json`). Running `allocate.py` on our saved
  sensitivities regenerates these plans exactly; re-running the whole probe can give a slightly different plan, because
  the probe starts with 50 LoRA warm-up steps on the GPU.
- GSM8K results move by about one point between seeds and between GPUs, and sampled decoding adds its own noise. The
  paired, greedy comparison between the two methods is more stable than either accuracy on its own.
- Our Llama-2-7B run was resumed from its step-1,200 checkpoint on a different GPU after a time-out; a fresh run goes
  straight through.
- The trained adapters are in the `adapters-v1` release (see [Trained adapters](#trained-adapters)), not in the git
  tree.
- The first-run logs in `results/early_v1/` were produced before the seeding fix and the full set of code fixes, and
  are not reproducible with the current flags exactly.

## References

- HyC-LoRA: Wang et al., *Memory Efficient LoRA Fine-tuning with Hybrid Activation Compression*, MLSys 2025.
  Code: https://github.com/thu-ee-acts-lab/HyC-LoRA-release (Apache-2.0).
- C-DIC: Jung et al., *Context-Driven Incremental Compression for Multi-Turn Dialogue Generation*, ICML 2026,
  arXiv:2606.12411.
- ICAE: Ge et al., *In-context Autoencoder for Context Compression in a Large Language Model*, ICLR 2024.
  Code: https://github.com/getao/icae.
