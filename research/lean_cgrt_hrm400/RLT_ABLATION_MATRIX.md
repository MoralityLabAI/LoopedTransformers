# RLT-01: Q5 / HRM400 feedback ablation

Registered 2026-11-14 before any RLT-01 training. Exploratory local experiment; the historical Q5 outcomes are already known. This amendment does not change the original Q5 rule or the 400M scaling gates. Results belong in a separate receipt, including failures.

The six-arm pilot configuration was fixed before training. The other rows are a planning matrix and need their own dated registration before execution; the scale-only diagnostic row was added while this pilot was running.

## Source and scope

Li et al., [Rectified Looped Transformer, arXiv:2611.16265v1](https://arxiv.org/abs/2611.16265v1), submitted 2026-11-12; supplied local PDF checked at pp. 4-5. Equation (4) sets `eta[k] = 1/(K-k)` for zero-based `k=0..K-1`:

```
h[k+1] = F(x[k], conditioning)
x[k+1] = x[k] + eta[k] * (h[k+1] - x[k])
```

The final coefficient is one. The feedback introduces no trainable parameters or extra block evaluations, but elementwise arithmetic still costs time. The magnitude guarantee assumes a residual bound `||F(x,t)-x|| <= M*(1-t)^alpha`, `0 < alpha <= 1`. Perturbation bounds require Lipschitz assumptions and common conditioning. These statements do not establish contraction or length generalization of our hybrids. The paper's time/step conditioning and endpoint supervision differ from the toy's hop supervision; this pilot transfers only the feedback schedule.

Context: the Claude chat **Lean-CGRT HRM audit and setup** (local session `local_94d819de-0711-40b2-8204-60307275d6b7`) and [the local scaling record](LOCAL_SCALING_RESULTS.md). Graph HRM in [model.py](lcgrt/model.py) already applies a post-norm to each fast state; the current Mathlib encoders in [tactic_lm.py](lcgrt/tactic_lm.py) use pre-norm blocks with input re-injection. Record these separately: a result on the Mathlib encoder is not a result on the graph HRM or a 400M model.

## Matrix

All feedback variants predict one candidate endpoint per fast step. Input re-injection, readout, slow-clock frequency, data, optimizer, training seed, batch and number of block calls are held fixed within each pair. The HRM slow block reads the interpolated fast state; its own feedback is unchanged.

| cohort | arms | training / evaluation | status |
|---|---|---|---|
| Mathlib CPU/GPU pilot | single-clock loop and two-clock HRM, each with direct / RLT / matched damping: **6 arms** | XS presets, fixed K=8, seed 131, 1,000 steps, batch 16; AdamW, lr 5e-4, 100-step warmup; final checkpoint evaluated | implemented by `lcgrt.rlt_pilot`; results pending |
| Q5 historical diagnostic | frozen Q4/Q5 checkpoints, direct feedback plus inference-only RLT and damping | original calibration items and outcomes, raw F1/F8 and original 0.05 threshold | planned; inference edits are distribution shifts, not RLT-trained models |
| Scale-only diagnostic control | direct feedback with a fixed valid-token norm cap, preserving direction | choose one radius per architecture/seed from the 95th percentile across training-calibration tokens and loops; freeze it before held-out scoring; record clipping frequency | planned; tests radial scale control without a remaining-time schedule |
| Q5 fresh transfer | direct / RLT / damping, lambda in {0, 0.01}, fresh seeds 10-15 | original depth-8 architecture, stream, learning rate and 30,000-step budget | planned; toy trainer not changed in this amendment |
| Q5 supervision control | same feedback arms and fresh seeds, same weights, but intermediate targets are the final answer instead of each loop's hop | needed before interpreting an endpoint-supervised variant as equivalent to Q5 | planned; at lambda=0 the two objectives coincide, so deduplicate |
| Graph HRM transfer | graph S-loop / HRM-S, direct / RLT / damping, post-norm retained | paired local runs on `lcgrt-mirror`; fixed and randomized-depth recipes reported separately | planned; graph trainer not changed |
| 400M | only after the existing local and scaling gates pass | same parameter and compute receipts as the current handoff | deferred; no new 400M run authorized by this matrix |

`matched_damping` uses a constant nonterminal coefficient
`c[K] = (sum(j=2..K, 1/j))/(K-1)`, and terminal coefficient one. At K=1 it is direct feedback. It has the same coefficient sum and terminal replacement as RLT at each budget. This controls average damping strength; it does not match the product of Jacobians or the realized displacement.

The pilot uses the existing approximately 1.5M-parameter XS presets. Direct and hybrid variants have exactly the same parameter tensors within each architecture. Across loop and HRM, report the actual counts and costs rather than attributing differences entirely to feedback. No time embedding, extra loss head, parameter reset, or slow-state interpolation is added.

## Recorded measurements

`diagnostics.json` per trained arm contains K in {4, 8, 16, 32}, each run from its own initial state. Model weights are fixed. The dataset is `mathlib-c3/train`; all monitoring and pilot evaluation uses `val`, with no registered test split read.

| measurement | definition / receipt |
|---|---|
| hidden magnitude | per-loop valid-token Euclidean norm of feedback state `x`; mean, p95, max; endpoint `h` mean norm recorded separately |
| update magnitude | `||x[k+1]-x[k]||`, endpoint displacement `||h[k+1]-x[k]||`, and whole-state update/input norm ratio; initial zero-state ratio is null |
| perturbation gain | paired clean/perturbed trajectories; initial fast-state noise at relative RMS {1e-4, 1e-3, 1e-2}; gain `||delta x[k]||/||delta x[0]||`; a second gain includes HRM slow-state differences |
| inference-depth robustness | bits per tactic byte and teacher-forced tactic exact match on the same fixed 128 val states, at every budget; neither is a proof-completion claim |
| Lean completion | existing kernel-certified evaluator, same 24 terminal val states and beam 8 for all arms; model, automation and interleaved model+automation close@{1,2,4,8}, validity, reached/certifiable counts, and failed files; primary encoder K=8, secondary K=32 |
| cost / provenance | trainer configuration and identity, split hashes, checkpoint SHA256, step throughput and wall time; process-tree peak commit, actual parameter counts, device and precision |

Perturbation probes use four fixed val goals and mask padded tokens. They are finite-direction, finite-size gains, not worst-case Lipschitz estimates. Slow slots evolve under each trajectory, so the HRM measurement does not inherit the paper's fixed-conditioning guarantee. Diagnostics run in fp32 (CPU by default; optional CUDA with TF32 disabled); CUDA training uses bf16 and a bounded allocator. Each CUDA diagnostic includes a one-goal CPU/CUDA logit parity check at K=8, rejecting differences outside atol 1e-3 / rtol 1e-4. The pilot keeps one final validation event, so best-on-val and final weights coincide.

For Q5, distinguish **task-depth accuracy** at depths 9-16 / 17-32 from **changing compute budget on the same task**. The original direct model's prefix at loop j is budget independent. RLT's prefix is not: K changes every preceding coefficient. Evaluate a new trajectory for every requested K; do not read loop j from a K=32 run and call it the K=j result. Keep task targets explicit. For the old one-hop-per-loop protocol, the RLT endpoint interpretation and hop targets may conflict; report that failure rather than silently changing the target.

## Interpretation fixed before results

- Retain Q5's depth-8 validity gate and `F1/F8 < 0.05` classifier. List exclusions and false negatives; do not tune the threshold on RLT outcomes. F1/F8 is first-order weight fragility, not a measured Jacobian contraction constant.
- Keep the original depths-9-16 extrapolation label and the extended depths-17-32 label separate. A historical Q5 false negative at the first horizon may fail at the longer one; do not silently relabel or pool the horizons.
- RLT directly changes a weight perturbation's contribution to the carried state through eta. A lower F1/F8 can therefore reflect the schedule. Report raw fragilities and state perturbation gains alongside it; do not call a schedule-induced decrease evidence of contraction.
- A nominal val advantage at one seed is exploratory. Compare RLT minus its direct parent and minus matched damping, separately in each architecture. Report exact counts, paired item outcomes and inference budgets. Repeated validation selection cannot establish a test-set advantage.
- Smaller hidden norms alone show scale control, not better reasoning. A feedback improvement with stable norms, lower gain and higher Lean completion is evidence consistent with the proposed mechanism; a damping tie suggests a generic damping effect. Neither licenses a 400M launch.
- A tiny completion sample may saturate or have few certifiable goals. Report this as uninformative when it cannot distinguish arms. Keep [the existing 400M no-go](SILICO_400M_HANDOFF.md) until its stated gates are met.

## Run and resource envelope

Run from `research/lean_cgrt_hrm400`. CPU mode runs under a **6 GiB total process-tree commit cap** and two torch threads; use small batches. Priority can be idle, below-normal (default), or normal. The actual pilot resumed at normal priority after a fully loaded host starved its GPU-launch thread; the model/data/optimizer budget was unchanged. CUDA mode requires an existing vramorama lease, reserves 4 GiB VRAM, caps the allocator below that lease, and adds that allowance to the process-tree commit cap because Windows charges GPU allocations to commit. Training is sequential in one reused runtime, resumable and bounded by `--max-minutes`; Lean evaluation uses one worker on CPU. Completed-arm CPU diagnostics can overlap training under a separate 4 GiB cap, with measured RAM receipts; they do not select weights. Checkpoints and detailed logs go outside Git; copy only the compact experiment receipts back into `results/`. Per-arm final-segment wall times do not measure feedback overhead on this shared host; record restarts separately.

```powershell
# CPU: reduce batch/steps if measured throughput requires it.
python -u -m lcgrt.rlt_pilot --out <scratch-dir> --device cpu --batch 8 --max-minutes 180
# GPU: use the installed vramorama executable and a CUDA-capable Python.
vramorama run --vram 4G --owner rlt-pilot --guard-power 90 --pause-exit 75 --restart 2 -- python -u -m lcgrt.rlt_pilot --out <scratch-dir> --device cuda --batch 16 --max-minutes 180 --phase train --priority normal
# After the lease is released, use the same identity for CPU evaluation.
python -u -m lcgrt.rlt_pilot --out <scratch-dir> --device cuda --batch 16 --max-minutes 180 --phase evaluate --priority normal
```

The resource budget is not evidence that a run completed. The final report must identify completed arms and actual evaluation coverage. A four-hour user budget is an upper bound; stop after the first complete comparison if the result resolves this pilot.

Resource amendment during the pilot: after the 85C guard paused training, the runner gained configurable duty/backoff controls (default maximum duty 0.5, soft threshold 78C). The hard 85C pause remains. CPU diagnostics were serialized after a transient low-RAM pause. These changes affect scheduling, not the training data, optimizer updates or evaluation labels. Resource segments and checkpoint resumes must accompany the final receipt.

Evaluation resource amendment, before any diagnostic/completion outcome: because the CPU was saturated, full-precision CUDA diagnostics and proposals were enabled with `--eval-device cuda` under a separate 4 GiB lease. All arms use the same evaluation device, which is recorded. Lean checking remains on CPU with one worker. CPU/CUDA reference differences are retained; this resource change is not evidence about model quality.

Completion-depth amendment, before any completion outcome: score encoder K=8 and K=32 on the same 24 goals, merging both proposal sets into one Lean pass. K=8 is primary; K=32 is an exploratory mismatch check. Separate policy names and cache identities retain each budget. This measures compute-depth robustness, not task-length generalization.
