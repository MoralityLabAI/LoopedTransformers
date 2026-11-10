# 400M Silico handoff

## Decision

**Do not launch the 400M campaign on the current evidence.** No Silico compute was used for any of this.

From the local campaign ([`LOCAL_SCALING_RESULTS.md`](LOCAL_SCALING_RESULTS.md)):
1. The two-clock HRM does not reliably beat a parameter-matched flat graph transformer on the only discriminating holdout (`test_mirror`). Over three seeds at 5M, HRM is 46.3 ± 17.5% vs flat 65.3 ± 18.3% at the tight budget, and level at 2× budget.
2. There is no size trend from 5M to 30M for either architecture.
3. CGRT branching never beat single-context best-first at matched Lean calls, and costs about 4× the model compute.
4. Seed variance (sd about 18 points) is larger than any architecture effect measured.

So a 400M run would not tell the arms apart, and none of the local evidence says scale is the bottleneck. The bottlenecks are benchmark validity (shortcut-free problems) and real proof data.

What would change the decision: **Gate 0** below, which is local and free.

**Gate 0 failed on 2026-11-09** ([result](LOCAL_SCALING_RESULTS.md#c2-and-gate-0-result-2026-11-09)). Flat-S solves 99.5% of `test_mirror_deep`, so no arm can clear +10 points on `lcgrt-mirror`. **c3 on Mathlib agrees (2026-11-10)** ([result](LOCAL_SCALING_RESULTS.md#c3-result-2026-11-10-no-recurrent-advantage-on-mathlib-the-400m-no-go-stands)). At S scale neither recurrent encoder beats flat at kernel-certified completion: HRM −2.7 and loop −0.9 points at close@8, both CIs include 0. Its XS replicate, c3-cpu, found the same ([result](LOCAL_SCALING_RESULTS.md#c3-cpu-result-v2-2026-11-10-no-architecture-effect-and-no-model-adds-to-automation)). The decision stands, and no local gate is left open.

## Local findings, estimates, and what needs a GPU to confirm

| kind | item | source |
|---|---|---|
| **local, measured** | parameter counts of every local and 400M preset | `count_parameters`, meta device |
| **local, measured** | throughput, peak VRAM, pause behaviour at 5M/15M/30M on an RTX 5080 Laptop | [`results/hw_diag.json`](results/hw_diag.json), [`results/c1/train_receipts.json`](results/c1/train_receipts.json) |
| **local, measured** | all certified success rates | [`results/c1/evals/`](results/c1/evals) |
| **estimate** | 400M activation memory, H100 samples/s, H100-hours | [`results/scale_400m.json`](results/scale_400m.json): memory scaled from measured bytes per (token × width × block application); throughput assumes the RTX 5080's achieved fraction of its matmul ceiling (23.5%) carries over to an H100 ceiling **assumed** at 650 TFLOP/s |
| **needs GPU confirmation** | 400M memory and throughput; whether any 400M arm beats its baseline | Gates 1–2 |

## Ready-to-run configurations

All three run with the existing trainer: `python -m lcgrt.train --preset <name>`.

| preset | architecture | width | heads | exact parameters | fwd GFLOP/decision (42 tokens) | est. memory per 20,480-token microbatch, grad ckpt | est. H100-h per 1M training states |
|---|---|---:|---:|---:|---:|---:|---:|
| `HRM-400M` | two-clock HRM, 1 fast + 1 slow block, T = 8 | 3,376 | 16 | **399,131,139** | 133.6 | 14.6 GiB | 0.97 |
| `loop-400M` | single-clock tied loop, T = 8 (the only variant whose test-time loops helped locally) | 5,152 | 16 | **398,332,194** | 241.1 | 13.7 GiB | 1.75 |
| `flat-400M` | 6 untied graph blocks | 2,304 | 16 | **398,364,866** | 33.5 | 9.4 GiB | 0.24 |

Deeper per-clock stacks are counted in `results/scale_400m.json` but not wired into the trainer:
- HRM 4+4 blocks at width 1,744: 398,692,995 parameters;
- HRM 3+3 blocks at width 2,008: 399,411,523;
- flat 12 × 1,648: 399,534,802.

All counts fall within the plan's 398M–402M window.

**Optimizer** (local values that worked, extrapolated):
- AdamW, β = (0.9, 0.95), weight decay 0.05, gradient clip 1.0, bf16 autocast;
- peak learning rate 1e-4 (local: 3e-4 at 5M, 2e-4 at 15M, 1.5e-4 at 30M; scaled by width^-½), warmup 300 steps, cosine to 10% over 3,000 steps;
- gradient checkpointing per fast step (`--grad-ckpt`);
- batch 256 states per step with a padded-token budget of 20,480 (`--token-budget`), no accumulation.

At about 15 GiB estimated, one microbatch fits an 80 GB card with room for a 3× larger budget. Keep the local sample stream for comparability.

**Checkpoint format** ([`lcgrt/train.py`](lcgrt/train.py)):
- `latest.pt` is a torch dict {model, opt, step, identity, identity_hash, config, torch_rng}, with a `latest.json` sha256 sidecar;
- each save rotates the previous checkpoint to `lkg.pt`/`lkg.json`;
- `complete.json` is written at the end;
- exit code 75 means "paused, resume me";
- resume refuses a changed identity hash.

At 400M a checkpoint is about 4.8 GB (fp32 weights + Adam state). Write checkpoints to `$SILICO_EXPERIMENT_ARTIFACTS_DIR`.

**Data manifests** (shards are deterministic; rebuild them on the job host and compare the `plan_sha256` recorded in the manifest):

| dataset | manifest | manifest sha256 |
|---|---|---|
| lcgrt-v1 (26,800 theorems) | [`results/data/lcgrt-v1_manifest.json`](results/data/lcgrt-v1_manifest.json) | `f98c911d6d4e55d84fa940249fc0f629ac6df585f563e4d8e5fb5da62988db30` |
| lcgrt-v1 + test_reason | [`results/data/lcgrt-v1_manifest_reason.json`](results/data/lcgrt-v1_manifest_reason.json) | `bfcb48b68094bc72169d7542a36f2848a602dbbf1570aa630434f98cf8f98f7b` |
| lcgrt-v1 + test_mirror | [`results/data/lcgrt-v1_manifest_mirror.json`](results/data/lcgrt-v1_manifest_mirror.json) | `4871aae2a2997ec14956da00ef5a8fc3d12e4be87c2f7167adf98d32ff309c66` |
| lcgrt-mirror (13,000 theorems) | [`results/data/lcgrt-mirror_manifest.json`](results/data/lcgrt-mirror_manifest.json) | `fbf3ae5e21e54b3e707c2aa69affbc69a470c4158c807d20c5ba743b9057e9e5` |

**Job host setup** (Linux):

```bash
cd research/lean_cgrt_hrm400
elan toolchain install leanprover/lean4:v4.19.0
export LCGRT_LEAN_HOME=~/.elan/toolchains/leanprover--lean4---v4.19.0
export LCGRT_WORK_ROOT=$SILICO_EXPERIMENT_ARTIFACTS_DIR
(cd lean && lake build)
python -m lcgrt.dataset build --name lcgrt-mirror --workers 8
python -m unittest discover -s tests
```

## Gates

| gate | work | compute | proceed when |
|---|---|---|---|
| **0 (local, required first)** | finish c2 (queued); then a 10-seed sweep at 5M on `lcgrt-mirror`: flat-S, S-loop, HRM-S with randomised depth (30 runs, about 8 RTX 5080 hours with thermal pauses) | 0 H100-h | a recurrent arm beats flat-S on `test_mirror_deep` at b1 by ≥ 10 points, with the seed-level 95% interval excluding 0 |
| 1 | H100 smoke: `HRM-400M`, `loop-400M`, `flat-400M`, 50 steps each with `--grad-ckpt`; record samples/s and peak memory against the estimates above | ≤ 0.33 H100-h | measured memory ≤ 72 GiB and throughput within 2× of the estimate |
| 2 (smallest distinguishing job) | the Gate-0 winner at 400M vs `flat-400M`, 3 seeds each, 3,000 steps × 256 states on `lcgrt-mirror` train (768k states). Evaluation is CPU/Lean-bound and can run locally on the downloaded checkpoints | est. 4.6 H100-h (loop arm 3 × 1.34 h + flat 3 × 0.18 h); cap 7 H100-h | the arm difference on `test_mirror_deep` exceeds the Gate-0 seed spread |

Three seeds per arm detect only large effects at the local seed sd (about 18 points), which is why the seed sweep comes first and runs locally, where seeds are cheap.

A from-scratch 400M campaign on synthetic Horn data alone is not justified even if Gate 2 passes:
- a 5M model already solves every split except the mirrors;
- 163k training states is far below what 400M parameters can use.

The next data stage, Mathlib extraction, is built (2026-11-06/07; see [`LEAN_DATA_PIPELINE.md`](LEAN_DATA_PIPELINE.md#mathlib-real-proofs-traced-by-lean-added-2026-11-06-after-the-ram-upgrade)):
- Mathlib v4.19.0 is installed, and its source proofs are traced with per-declaration certification;
- a byte-level goal→tactic model has flat, two-clock and single-clock encoders;
- a kernel-certified one-step completion benchmark runs on held-out Mathlib states.

The local experiment on it, c3 in [`LOCAL_SCALING_RESULTS.md`](LOCAL_SCALING_RESULTS.md#c3-mathlib-registered-before-results), ran on 2026-11-10. H1 failed, so the no-go stands. H2 held for flat-S and loop-S: a learned model interleaved with automation adds about 2 points over automation alone at 8 Lean calls. Premise retrieval is still not built.
