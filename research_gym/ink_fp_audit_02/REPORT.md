# INK-FP-AUDIT-02 report (2026-11-28)

Status: **local engineering complete; all remote stages blocked**. No Inkling sample was drawn,
no adapter was trained, and no native-model claim is made. Paid spend: 0. API requests: 0.

## Why the remote stages did not run

Campaign receipt: [`receipts/campaign_20261128/campaign_state.json`](receipts/campaign_20261128/campaign_state.json).

1. `TINKER_API_KEY` is not in the process environment (also absent from the user and machine
   environment scopes). The free-tier eligibility check cannot run without it, so the
   "stop if free-tier eligibility cannot be established" rule applies to Stages A-F.
2. Available DRAM was below the 6 GiB startup floor at the last three preflights (5.42, 4.48 and
   4.73 GiB). Other experiments on the host use that memory, and this campaign does not touch them.

Each remote stage is recorded as `blocked` with `resumable: true`. To resume, re-run
`python -m research_gym.ink_fp_audit_02.campaign --out research_gym/ink_fp_audit_02/receipts/campaign_20261128 --allow-remote`
with the key exported and at least 6 GiB available. Stage A is the audit-01 six-request smoke.
Stages B-F stop after Stage A until its receipt has been reviewed.

## Evidence classes in this PR

| class | what | files |
|---|---|---|
| published_record_replay | upstream RSI acceptance audit | `receipts/.../upstream_audit.json`, `fixtures/upstream_gspo_gate_765453d.json` |
| exact_finite_verification | 11 task families, reference answers verified, leakage audit | `curriculum_manifest.json`, `leakage_report.json`, `tasks_{train,dev,accept}.jsonl` |
| synthetic_cpu | BPD analytic-update proxy | `receipts/.../synthetic_cpu/bpd_proxy.json` |
| engineering_simulation | engine, acceptance, rollback, resume and exports on a toy policy | `receipts/.../engineering_simulation/` |
| native_remote_observation | none | - |
| prospective_held_out | none; sealed partition counted (8 roots per family), never materialised | `sealed_task_ids_sha256` in manifest |

## 1. Upstream acceptance discrepancies (regression fixture)

The upstream record at commit `765453d` is replayed along its reported predecessor chain, using the
published rule (at most 2 correct-answer losses per mode, repetition cap, total gain or completion
tie-break). All five generations are reported accepted. The rule rejects four of them:

| gen | predecessor | candidate | worst per-mode loss | reported | rule |
|---:|---:|---:|---:|---|---|
| 0 | 111 | 112 | 1 | accept | accept |
| 1 | 112 | 114 | 4 | accept | reject |
| 2 | 114 | 118 | 3 | accept | reject |
| 3 | 118 | 114 | 6 | accept | reject |
| 4 | 114 | 116 | 4 | accept | reject |

`tests/test_ink_fp_audit_02_acceptance.py` pins this result: it asserts the discrepant generations are
exactly [1, 2, 3, 4], with per-mode drops recomputed independently of `acceptance.py`.

## 2. Our acceptance is independently replayable

Every candidate record stores the frozen criteria, the predecessor checkpoint, both full paired
case sets, hyperparameters, scales examined, the decision and the next head. Records sit in a
hash-chained ledger. The tests re-implement the decision rule without importing it, replay
every ledger, and check that:

* the predecessor equals the accepted head at that point (a rejected candidate can never become a predecessor),
* the recorded verdict matches the recomputed one,
* rejected candidates are persisted along with their checkpoint identity,
* an edited verdict breaks the hash chain.

Frozen criteria (`ink-fp-audit-02.acceptance.v1`): at least +1 paired verified-correct case on the
generation-acceptance partition; per-family loss at most max(1, ceil(0.10 n_family)); repetition
rate up by at most 0.02; completion rate down by at most 0.02. Repeated acceptance evaluations are
development evidence only.

## 3. Curriculum

11 families, each with 8 accept, 8 dev and 16 train tasks. All 352 materialised tasks passed the
cross-partition leakage audit, which checks for no shared source world, semantic task id,
normalised prompt or proof certificate. Partitions are filled sealed-first, so a training
duplicate can never displace a held-out problem. Deduplicating by semantic id was necessary:
short Dyck strings and small modular problems recur across independent seeds, and the engine's
paired-acceptance check rejected the first build because of it. Lean tactic ranking is not
included, because DataGoblin's `rlt_lean_rank` needs torch and a Lean-checked pool. No
Compactification Theory show-your-work quiz module exists in the surveyed repos. `exports.py`
therefore defines the eight-field show-your-work schema itself and keeps those explanations
apart from verifier certificates.

## 4. Synthetic BPD proxy (synthetic_cpu; not Inkling evidence)

Linear layer, d_in 192, d_out 96, 96 protected input directions (null space dimension 96),
384 new-task samples, rank-8 target edit, 20 seeds. Table entries are relative errors.
"Conflict" is the fraction of new-task input energy inside the protected span.

| conflict | method | protected residual (max) | held-out protected (mean) | held-out new-task R2 (mean +- sd) |
|---|---|---:|---:|---:|
| 0.1 | ridge (soft, FP-Solve style) | 6.6e-04 | 3.5e-02 | 0.712 +- 0.014 |
| 0.1 | truncated SVD r8 | 2.6e-01 | 2.4e-01 | 0.868 +- 0.016 |
| 0.1 | projected (null basis) | 1.2e-16 | 4.8e-15 | 0.711 +- 0.014 |
| 0.1 | projected, low rank r8 | 1.1e-16 | 4.5e-15 | 0.841 +- 0.009 |
| 0.5 | ridge | 1.6e-03 | 3.7e-02 | 0.211 +- 0.029 |
| 0.5 | truncated SVD r8 | 2.3e-01 | 2.1e-01 | 0.937 +- 0.005 |
| 0.5 | projected | 1.4e-16 | 5.6e-15 | 0.179 +- 0.027 |
| 0.5 | projected, low rank r8 | 1.3e-16 | 5.1e-15 | 0.307 +- 0.026 |
| 0.5 | literal `pinv(X_new P)` projected | 2.6e-01 | 2.1e-01 | 0.281 +- 0.054 |
| 0.5 | matched-rank random | 1.8e-01 | 1.7e-01 | -0.686 +- 0.033 |
| 0.9 | ridge | 2.3e-03 | 4.1e-02 | -0.269 +- 0.034 |
| 0.9 | projected | 2.5e-16 | 9.9e-15 | -0.353 +- 0.026 |
| 0.5, float32 | projected | 4.4e-08 | 2.7e-07 | 0.179 +- 0.027 |
| 0.5, float32 | literal pinv projected | 8.9e-02 | 8.4e-02 | -1.597 +- 0.101 |
| 0.5, 20% off-span protected inputs | projected | 1.4e-16 | 1.2e-01 | 0.179 +- 0.027 |

Findings:

* The projected update holds X_keep dW^T at rounding level: about 1e-16 in float64 and 4e-8 in
  float32. Ridge, the soft-preservation method, leaves 6.6e-4 to 2.3e-3 on the fitted rows and
  about 0.04 on held-out protected inputs.
* Implementing P = I - X_keep^+ X_keep literally and solving U = pinv(X_new P) R is numerically
  unsafe. NumPy's default rcond keeps rounding-level singular values of X_new P, U grows to about
  4e14 along span(X_keep), and P U then leaks a relative residual of 0.26, worse than ridge.
  The stable form is D = N (X_new N)^+ R, with N an orthonormal null-space basis from the SVD.
  Both forms are kept in the code; the literal one is labelled diagnostic.
* Exactness covers span(X_keep) only. When 20% of the held-out protected energy lies off that
  span, the projected update's held-out error rises to 0.12, comparable to ridge's.
* Preservation costs plasticity as conflict grows. At conflict 0.9 every preserving method has
  negative held-out new-task R2. Truncated SVD, which preserves nothing, fits best everywhere,
  because the target edit is rank 8.
* Restricting the projected update to rank 8 raises held-out R2 at no cost to preservation (0.841
  vs 0.711 at conflict 0.1, 0.307 vs 0.179 at conflict 0.5). The rank restriction regularises.
* Adaptive rank allocation across 4 synthetic modules (budget 16) captured 0.9974 of spectral
  energy, vs 0.9912 for uniform allocation, and was at least as good on all 20 seeds.
* Strategy router: nearest-centroid on one measurable feature (the fraction of new-task input
  energy inside the protected span), choosing among ridge, projected and truncated SVD with
  utility = R2 - 2 x held-out protected error. Fitted on 30 dev scenarios, it matched the oracle
  on all 60 fresh scenarios (mean utility 0.568, vs 0.461 always-SVD, 0.182 always-projected,
  0.148 always-ridge, 0.264 random). The synthetic families separate perfectly on that feature,
  so this shows the selection mechanism runs. It is not evidence that the router transfers to
  real adapter updates.

Tinker exposes no activations, gradients or adapter tensors (see README), so none of these
methods can be applied to Inkling through the documented SDK. Arm H is recorded as
unsupported natively.

## 5. Engine (engineering_simulation; toy policy, no model evidence)

All arms ran 4 generations on the 176 train and 88 accept tasks, using a seeded toy policy (per-family
skill logits, an effort bonus, noisy updates). Results:

| arm | decisions (gen 0-3) | final head | replay mismatches |
|---|---|---|---:|
| A frozen | - | base | 0 |
| B SFT | accept, reject, accept, accept | B-g3 | 0 |
| C GRPO / importance_sampling | 4 accept | C-g3 | 0 |
| D effort-advantage GSPO | 4 accept | D-g3 | 0 |
| E shuffled rescue | 4 accept | E-g3 | 0 |
| F adaptive curriculum | 4 accept | F-g3 | 0 |
| G cost-sensitive | 4 accept | G-g3 | 0 |
| H analytic | unsupported natively | - | - |

The toy policy does not distinguish the arms, and was not designed to. What the run checks is
the plumbing: six samples per problem (2 low, 2 medium, 2 high), frontier selection, rewards and
advantages, paired predecessor/candidate evaluation, acceptance, rollback after rejection (arm B,
generation 1), resumption after a simulated API outage, and Aleph-format exports (arm B: 54 sft,
71 repairs, 162 outcomes, 54 routing rows). Separate tests force every candidate to degrade; the
head then stays at `base` and every rejected checkpoint is persisted.

## 6. Statistics

* Power preflight (one-sided exact McNemar on root-level paired outcomes; minimum effect 0.10;
  assumed discordance 0.30; alpha 0.05; power 0.80): **199 independent sealed roots** for the
  single primary contrast D vs A, and **336** with Bonferroni across 7 arm-vs-A comparisons. The
  sealed partition (8 roots per family) is therefore far too small to confirm a 0.10 effect, and
  must be expanded before Stage E. No mechanistic-power preflight script exists in the surveyed
  repos; `stats.power_preflight` is the replacement and is labelled as such.
* The FSI-01 bridge (`research_gym/scripts/evalmath_fsi_gate.py`, using EvalMath-mit's
  `spiral_ln.evaluation_finite_sample`) ran end to end on simulated dev outcomes, declared not
  preregistered and not a sealed readout. Verdict: `inconclusive` (88 roots). This only shows
  that the bridge runs.

## 7. Resources (this host, campaign run)

Figures are from the first `resources` entry in `campaign_ledger.jsonl` (the full run); `resources.json` holds the last, short resumed run that re-ran the stats stage.

The full local campaign ran inside a 4 GiB Job Object at below-normal priority, with
`CUDA_VISIBLE_DEVICES=-1`.

* Peak process-tree RSS 246 MB; Job Object peak committed memory 267 MB.
* Minimum available system DRAM 3.35 GiB, set by other workloads on the host. The governor's
  pause threshold (3 GiB) was not crossed.
* CPU 160 s, wall 260 s; local GPU allocation 0 bytes; 0 API requests; 0 remote training tokens.
* Job Object enforcement was tested: a child process in a 64 MiB job is refused a 256 MiB
  allocation and completes a 16 MiB one. On this host `PeakJobMemoryUsed` also counts the
  refused commit attempt, so it is not used as proof of the cap.

## 8. Not done, and why

* Stages A-F on Inkling: blocked on the credential and on DRAM (section 0). Free-tier sampling,
  effort controls, token accounting, LoRA training, checkpoint save/resume, sampling from trained
  adapters, rate limits and the OpenAI-route/adapter interaction are all unqualified.
  `TinkerBackend` and `NativeTinker` follow the installed SDK's signatures, but they have only
  run under fake transports and have never touched the live service.
* The datum layout for the `gspo` and `importance_sampling` losses (`target_tokens`, `logprobs`,
  `advantages`) follows the SDK's documented RL inputs but has not been checked by a live
  `forward_backward`.
* Lean tactic ranking and proof repair: not integrated (torch plus Lean pool).
* Claims about internal recurrence, Layer 17 mechanisms or contraction: none made. Reasoning effort
  is an observable inference control.
