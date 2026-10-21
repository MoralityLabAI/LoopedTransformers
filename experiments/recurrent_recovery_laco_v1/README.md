# Recurrent recovery of LaCo-pruned Qwen3

Question: can a *compact shared recurrent block with loop-specific adapters* recover computation that LaCo layer
collapse removed?

    h[t+1] = h[t] + g[t] * F_shared(h[t]; adapter[t], norm[t])        t = 0 .. T-1

where `F_shared(h) = block(h) - h` is the residual update (attention + MLP) of the LaCo-collapsed layer, shared
across loops; `adapter[t]` is a rank-r LoRA on all 7 linear maps of the block, one per loop; `norm[t]` are the 4
RMSNorms (input, post-attention, q-norm, k-norm), either shared or per-loop; `g[t]` is a learnable per-loop gate
(init 1).  Residual pathways are never altered and `F(h).shape == h.shape` is asserted at every step.

## Conditions

| id | what | script |
|----|------|--------|
| A | teacher transformer = Qwen3-0.6B-Base + LoRA tuned on the synthetic tasks (CE) while staying close to the base on text (KL); LoRA merged | `train_student.py --cond A` |
| B | LaCo-pruned student (no training) | `--cond B --steps 0` |
| C | pruned student + ordinary LoRA on every remaining layer | `--cond C` |
| D | pruned student, collapsed layer replaced by the recurrent block above | `--cond D` |
| E | D with untrained (identity) adapters = *naive repeated application* control | `--cond D --steps 0` |

Why a tuned teacher?  The base model cannot do the synthetic algorithmic tasks, so "recovering" algorithmic
computation would be undefined.  A is the best a 28-layer model reaches with the same data; students are pruned from
A and distilled against A.

## Layout

    looprec/qwen3_core.py   own Qwen3 implementation (checked against HF: scripts/verify_core.py, fp32 diff 4e-5)
    looprec/laco.py         LaCo merge arithmetic (last|first|mean|rdsc|rdsc_mean), searches, manifests, student builder
    looprec/adapters.py     LoRA / loop-specific LoRA, bitsandbytes 4/8-bit
    looprec/tasks.py        finite-state (S5 word problem), arithmetic (add, mul), multistep variable chains
    looprec/losses.py       chunked KL + CE (no full-batch logits), residual-state matching loss
    looprec/evaluation.py   accuracy / KL / latency / memory / FLOPs per loop count; mcbench.py = ARC-E/C, PIQA
    scripts/run_laco.py     LaCo search -> manifests/*.json
    scripts/train_student.py  one condition (train + evaluate + manifest)
    scripts/run_suite.py    all conditions + ablations, VRAM-aware retry
    scripts/make_report.py  plots + tables + checkpoint index
    scripts/test_pipeline_cpu.sh  end-to-end smoke test on a tiny random model (no GPU)

## Run

    export LOOPREC_PYTHON=<python with torch>=2.5, transformers (only for verify), safetensors, pandas, matplotlib>
    python scripts/verify_core.py
    python scripts/run_suite.py --phase teacher
    python scripts/run_suite.py --phase main --laco-recipe <path to laco-lab laco_recipe.json>   # or omit: own LaCo search
    python scripts/run_suite.py --phase ablate
    python scripts/make_report.py

Quantisation / 8B: `--quant 4bit` swaps frozen base linears for bitsandbytes NF4 (`pip install --target vendor
bitsandbytes`); `--grad-ckpt` enables per-layer and per-loop-step activation checkpointing.
See `reports/REDUCTIONS.md` for what was scaled down and why.

## Status / results

All conditions and 20 ablations have been run on Qwen3-0.6B-Base (single seed). Read `reports/RESULTS.md` (interpretation),
`reports/tables.md`, `reports/figs/`, `reports/checkpoint_manifest.json` (per-run manifests with sha256 in `runs/*/manifest.json`)
and `reports/REDUCTIONS.md`. Tests: `pytest tests -q` and `scripts/test_pipeline_cpu.sh`.

## Provenance and what is not in git

- **Checkpoints are not committed.** The tuned teacher (`checkpoints/teacher_tuned/model.safetensors`, 1.2 GB) and each run's
  `trainable.pt` / `resume.pt` stayed in the Claude session folder that produced them; they are identified by hash in
  `reports/checkpoint_manifest.json` and `runs/<run>/manifest.json` (`teacher_sha256`, `checkpoint_files[].sha256`). Every run can be
  regenerated with `scripts/run_suite.py` (teacher tuning ~25 min, whole suite a few hours on a free 16 GB GPU).
  Per-run `results.json` and `train_log.jsonl` are committed, so `scripts/make_report.py` reproduces the tables and figures.
- **LaCo recipe source: our own search** (`scripts/run_laco.py`, single collapsed range of 7 layers, `last` merge, calibration on
  pile-10k train chunks; `manifests/laco_r{7,6,4}.json`). The separate LaCo reproduction ("laco-lab") was not finished when these runs were made;
  its recipe lives in the jev-qwen repo on branch `add/laco-lab-layer-compression`. `scripts/train_student.py --laco` /
  `run_suite.py --laco-recipe` accept a laco-lab `laco_recipe.json` directly, so the study can be re-run against it.
- Dependencies: torch >= 2.5, safetensors, tokenizers, pandas, matplotlib (transformers only for `scripts/verify_core.py`);
  optional bitsandbytes in `vendor/` for `--quant`. Tests: `python -m pytest tests -q` from this directory.
- Headline: extra recurrent steps did not recover knowledge or algorithmic ability beyond ordinary adaptation; real layers beat loops at
  matched FLOPs (see `reports/RESULTS.md`).
