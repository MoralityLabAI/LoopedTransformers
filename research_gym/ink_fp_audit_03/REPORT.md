# INK-FP-AUDIT-03: budget-aware research intelligence (2026-11-28)

Primary endpoint: verified research progress per unit of computation. Every Inkling request used
`thinkingmachines/Inkling-2-Mini:peft:65536:free`; paid spend 0. Tables between `reconcile:` markers are
regenerated from receipts by `reconcile.py`; `tests/test_ink_fp_audit_03_reconcile.py` fails on drift.

Evidence classes: **native_remote_observation** (Inkling via Tinker), **synthetic_cpu** (BPD proxy),
**cpu_numpy_training** (MeTTa students), **published/campaign records** (trajectories). Everything below is
development evidence unless a section says it was registered before sampling.

## 1. Audit-02 reconciliation

Replaying the 288 raw audit-02 records corrects two report numbers: **28 truncations** (not 27), all of them
failures and none correct-but-truncated; and **9 families** correct at all three efforts (not 10;
mcp_retrieval is 7/8 at high). The audit-02 report now carries generated blocks:

| Effort | Correct | Completion tokens | Rescues vs. low |
|---|---:|---:|---:|
| Low | 89/96 | 45,831 | — |
| Medium | 87/96 | 58,330 | 0 |
| High | 84/96 | 70,048 | 0 |

## 2. Native Tinker training qualified (`receipts/native_qual_20261128/`)

`tml-renderers` 0.5.0 is the real renderer package but ships Linux/macOS wheels only, so on Windows pip sees just
the empty 0.0.1. Qualification ran in a WSL venv (tinker 0.40.1, tinker-cookbook 0.7.1, tml-renderers 0.5.0, CPU
torch) with a 4 GiB cgroup cap; peak RSS 419 MB.

| Check | Result |
|---|---|
| Renderer | `tml_v0` (cookbook `get_recommended_renderer_name`) |
| Prompt tokens vs OpenAI-compatible route | 98 = 98 |
| Effort control | render-time float (0.2 / 0.7 / 0.9); not a sampling parameter |
| Sampling with token ids and logprobs | logprobs aligned 1:1 with tokens at every effort |
| Native low effort, temperature 0, vs OpenAI route | same final answer |
| cross_entropy, importance_sampling, gspo steps + AdamW | all ran; GSPO sequence log-ratio −0.0086 |
| Checkpoint save, sampler weights, sample from adapter | correct answer on the free base |
| Restore with optimizer state, then train | ran |
| Checkpoint storage rate | 0.0 USD per GB-month |

Because effort is set when the prompt is rendered, native sampling yields **exact behaviour-policy logprobs for
an effort-conditioned policy**. On-policy GSPO therefore needs no re-tokenised text. The OpenAI-compatible route is
kept only for behavioural effort experiments.

## 3. INK-RESEARCH-FRONTIER-01 (`frontier.py`, `receipts/frontier01_20261128/`)

Eight verifier-backed families: storyworld_v2 (plans or exact single-variable unreachability certificates under
irreversible transitions), experiment_choice (exact expected information gain per cost, with confounded,
reused-validation and underpowered options), power_design (exact one-sided McNemar minimum roots),
graph_certificate (2-colouring or odd cycle), stability_certificate (spectral radius), sorting_network
(prefix-constrained, exhaustive 0/1 check), hillclimb_diagnosis (repeated proposals, local optima),
research_memory (the same decision rendered as full history or a structured summary). Partitions: **64 sealed
tasks reserved first, stored as an id digest only**, 48 accept, 48 dev, 96 train; the leakage audit is clean.

Not built: MathCan and Lean/MCP-backed exercises (no CPU-lean path); algorithm discovery is limited to
sorting networks, because arbitrary model code is not executed.

## 4. INK-BUDGET-01 behavioural pilot (`budget_eval.py`, `budget_policies.py`, `receipts/budget01_dev_20261128/`)

One 8,192-token call per (task, effort) at temperature 0. The outcome at a smaller cap c is implied:
success(c) iff the long run finished correctly within c tokens. Arm G (search with external memory) needs new
calls and was not run.

<!-- reconcile:a03_budget_effort:start -->
| Effort | Correct | Completion tokens | Truncated | Rescues vs. low | Losses vs. low |
|---|---:|---:|---:|---:|---:|
| Low | 38/48 | 126,035 | 9 | — | — |
| Medium | 33/48 | 166,481 | 15 | 0 | 5 |
| High | 30/48 | 191,420 | 18 | 0 | 8 |

Records: 144. Truncations (finish_reason=length): 42; failures: 43, of which truncated: 42; correct-but-truncated: 0. Families correct on every task at all three efforts (4): graph_certificate, hillclimb_diagnosis, research_memory, storyworld_v2.
<!-- reconcile:a03_budget_effort:end -->

<!-- reconcile:a03_budget_families:start -->
| Family | Low | Medium | High |
|---|---:|---:|---:|
| experiment_choice | 6/6 | 1/6 | 0/6 |
| graph_certificate | 6/6 | 6/6 | 6/6 |
| hillclimb_diagnosis | 6/6 | 6/6 | 6/6 |
| power_design | 0/6 | 0/6 | 0/6 |
| research_memory | 6/6 | 6/6 | 6/6 |
| sorting_network | 2/6 | 2/6 | 1/6 |
| stability_certificate | 6/6 | 6/6 | 5/6 |
| storyworld_v2 | 6/6 | 6/6 | 6/6 |
<!-- reconcile:a03_budget_families:end -->

<!-- reconcile:a03_budget_arms:start -->
Paired tasks with all three efforts: 48.

| Arm | Success | Tokens | Success / 1k tokens | Truncation rate | Correct given complete | Escalated | Rescued by escalation |
|---|---:|---:|---:|---:|---:|---:|---:|
| A low@1024 | 27/48 | 33,063 | 0.817 | 0.417 | 0.964 | 0 | 0 |
| A low@2048 | 28/48 | 53,142 | 0.527 | 0.396 | 0.966 | 0 | 0 |
| A low@4096 | 34/48 | 86,183 | 0.395 | 0.271 | 0.971 | 0 | 0 |
| A low@8192 | 38/48 | 126,035 | 0.302 | 0.188 | 0.974 | 0 | 0 |
| B medium@1024 | 24/48 | 38,025 | 0.631 | 0.500 | 1.000 | 0 | 0 |
| B medium@2048 | 27/48 | 60,000 | 0.450 | 0.438 | 1.000 | 0 | 0 |
| B medium@4096 | 30/48 | 101,031 | 0.297 | 0.375 | 1.000 | 0 | 0 |
| B medium@8192 | 33/48 | 166,481 | 0.198 | 0.312 | 1.000 | 0 | 0 |
| C high@1024 | 21/48 | 41,285 | 0.509 | 0.562 | 1.000 | 0 | 0 |
| C high@2048 | 27/48 | 64,652 | 0.418 | 0.438 | 1.000 | 0 | 0 |
| C high@4096 | 27/48 | 107,660 | 0.251 | 0.438 | 1.000 | 0 | 0 |
| C high@8192 | 30/48 | 191,420 | 0.157 | 0.375 | 1.000 | 0 | 0 |
| D1 low1024>med4096>high8192 | 31/48 | 263,923 | 0.117 | 0.354 | 1.000 | 21 | 4 |
| D2 low2048>low8192 | 38/48 | 165,888 | 0.229 | 0.188 | 0.974 | 20 | 10 |
| D3 low512>low2048>low8192 | 38/48 | 181,248 | 0.210 | 0.188 | 0.974 | 30 | 20 |
| D4 low1024>low4096>high8192 | 35/48 | 222,182 | 0.158 | 0.271 | 1.000 | 21 | 8 |
| E router lam=0.25 | 38/48 | 126,188 | 0.301 | 0.188 | 0.974 | 0 | 0 |
| E router lam=1.0 | 38/48 | 126,188 | 0.301 | 0.188 | 0.974 | 0 | 0 |
| F allocator | 38/48 | 170,925 | 0.222 | 0.188 | 0.974 | 18 | 8 |
| E abstain lofo tau=0.2 | 38/48 | 126,035 | 0.302 | 0.188 | 0.974 | 0 | 0 |
| E abstain lofo tau=0.4 | 34/48 | 123,093 | 0.276 | 0.188 | 0.872 | 0 | 0 |
| E abstain loto tau=0.2 | 38/48 | 76,883 | 0.494 | 0.062 | 0.844 | 0 | 0 |
| E abstain loto tau=0.4 | 38/48 | 76,883 | 0.494 | 0.062 | 0.844 | 0 | 0 |
| H oracle | 38/48 | 52,023 | 0.730 | 0.208 | 1.000 | 0 | 0 |

| Adaptive arm | Success | Tokens | Best static arm within that token budget | Static success | Static tokens |
|---|---:|---:|---|---:|---:|
| D1 low1024>med4096>high8192 | 31 | 263,923 | A low@8192 | 38 | 126,035 |
| D2 low2048>low8192 | 38 | 165,888 | A low@8192 | 38 | 126,035 |
| D3 low512>low2048>low8192 | 38 | 181,248 | A low@8192 | 38 | 126,035 |
| D4 low1024>low4096>high8192 | 35 | 222,182 | A low@8192 | 38 | 126,035 |
| E router lam=0.25 | 38 | 126,188 | A low@8192 | 38 | 126,035 |
| E router lam=1.0 | 38 | 126,188 | A low@8192 | 38 | 126,035 |
| F allocator | 38 | 170,925 | A low@8192 | 38 | 126,035 |
| E abstain lofo tau=0.2 | 38 | 126,035 | A low@8192 | 38 | 126,035 |
| E abstain lofo tau=0.4 | 34 | 123,093 | A low@4096 | 34 | 86,183 |
| E abstain loto tau=0.2 | 38 | 76,883 | A low@2048 | 28 | 53,142 |
| E abstain loto tau=0.4 | 38 | 76,883 | A low@2048 | 28 | 53,142 |
<!-- reconcile:a03_budget_arms:end -->

Findings (development, 48 roots):
* **Higher effort never rescued a low-effort failure** (0 of 48 at medium or high, as in audit-02). It lost 5
  (medium) and 8 (high) tasks, almost all by truncation. On experiment_choice it fell from 6/6 to 1/6 and 0/6.
* **Static low effort with the full budget is the strongest fixed policy, and no escalation ladder, effort router
  or length allocator beats it at matched tokens.** Effort-escalating ladders lose successes and spend more
  (D1 31/48 at 2.09x its tokens, D4 35/48 at 1.76x). Cap-only ladders and the length allocator match its 38/48
  but spend 1.32-1.44x. The leave-one-family-out effort router chose low on 47 of 48 tasks.
* The oracle needs 52,023 tokens for the same 38 successes (2.4x fewer). The waste comes from full-budget
  attempts on hopeless tasks (power_design 0/18 at every effort, sorting_network 5/18) and over-long budgets on
  easy ones.
* A family-aware abstainer (leave-one-task-out) keeps 38/48 at 76,883 tokens. Leave-one-family-out, it saves
  nothing: what it learns is family-specific, not a transferable difficulty signal. Four abstainer variants were
  examined, so this was registered and confirmed on fresh tasks (section 5).

## 5. Registered abstainer confirmation (`receipts/budget01_confirm_20261128/`)

The family-aware abstainer was trained on all 144 dev records, frozen, and its predictions for the 48 untouched
acceptance tasks were committed and pushed (db50c689) before any of those tasks was sampled. Every task was then
sampled, including the abstained ones, so the successes lost by abstaining are measured, not assumed.

<!-- reconcile:a03_budget_confirm:start -->
Registration sha256 `0aede87242469778e62caa9e92cb3fb41084e6e1e0d21405f28495aaf4b046eb`; 48 fresh acceptance tasks at low effort, 8,192-token cap.

| Policy | Success | Completion tokens | Success / 1k tokens | Abstained | Successes lost by abstaining |
|---|---:|---:|---:|---:|---:|
| Static low@8192 | 41/48 | 121,001 | 0.339 | 0 | 0 |
| Registered abstainer | 40/48 | 77,997 | 0.513 | 6 | 1 |

Token saving: 35.5%.
<!-- reconcile:a03_budget_confirm:end -->

The abstainer declined the 6 power_design tasks. That is one family it had seen fail every time in development,
so this is family-specific budget allocation, not novel-family transfer. The 35% token saving is deterministic
given the receipts. The one-task success difference (40 vs 41) is far below what 48 roots can resolve (section 6).
Measured as verified successes per 1,000 completion tokens, the gain is 0.513 vs 0.339 (+51%).

## 6. Statistics (`receipts/power_recompute_20261128.json`)

Exact one-sided McNemar, power 0.8, alpha 0.05 (Bonferroni for 7 comparisons), using discordance measured in
these campaigns:

| Measured contrast | Discordance | 5 pp, 1 comparison | 5 pp, 7 comparisons | 10 pp, 1 comparison | 10 pp, 7 comparisons |
|---|---:|---:|---:|---:|---:|
| audit-02 low vs high (dev) | 0.052 | 139 | 211 | impossible (effect > discordance) | impossible |
| budget-01 low vs high (dev) | 0.167 | 443 | 744 | 111 | 184 |
| budget-01 low vs medium (dev) | 0.104 | 278 | 464 | 69 | 105 |
| audit-02 assumption | 0.300 | 776 | 1,331 | 199 | 336 |

The 64 sealed frontier roots cannot confirm a 10-point paired effect at the measured discordance. The sealed
partition stays closed. A confirmatory study needs at least 111 fresh roots for a single registered contrast.

## 7. BPD protected low-rank updates (BPD `research/bpd-nullspace-01`, head 88dff81)

Module `mechinterp/bpd/experiments/nullspace_update/` (NumPy). Tests: 13/13, re-run by the campaign lead.
* The stable solve D = N (X_new N)^+ R keeps the protected error at ~1e-16 in float64 and ~8e-9 median in
  float32. The literal pinv form inflates U by about 1e9 and leaks a median 0.088.
* On 800 jointly random scenarios (protected rank, conflict, sample size, noise, condition up to 1e11, target
  rank, off-span energy, precision), reduced-rank null-space (rank 2) has the best mean utility (−0.566).
  Unprotected methods fit the new task far better (R² 0.85 vs 0.14-0.26) but break preservation. A plain
  null-space solve needs Tikhonov damping when X_new lies nearly inside span(X_keep).
* Learned method selector: a depth-3 tree beats every fixed method on held-out seeds (−0.549 vs −0.613), but not
  on a held-out property combination (train condition < 1e3 and conflict < 0.6; test the complement): −0.688
  vs −0.681, within noise. Adaptive rank allocation gave no gain (pooled R² 0.186 vs 0.190).
* No small activation archive exists in BPD (the only file under 50 MB holds 512/1536-wide weight factors), so
  the solver was not applied to real activations. No claim is made about Inkling.

## 8. MeTTa student failure basin (DataGoblin `research/metta-stop-01`, head ef3d0ef)

`MeTTa/stop_diagnostics.py`, `student_variants.py`, `stop_study.py`, `STOP_DIAGNOSIS.md`. Tests: 39 passed
(27 existing + 12 new), re-run by the campaign lead.
* **The failure is solution generation, not stopping.** With oracle stopping the default students reach the
  optimum in 2/180 (TRM) and 0/180 (LDT proxy) fresh cases. Teacher-forced canonical-edit accuracy is 0.285,
  0% from the sixth edit on. Train-set edit accuracy is 0.358 (16 steps) and 0.373 (200 steps): the student
  underfits its own 459 training rows. Separately, the action-class STOP fires at 681/681 unsolved states.
* Nine variants on 60 fresh roots × 3 seeds at matched compute: every variant is 0/180 solved_and_stopped except
  the LDT-TRM hybrid at 16 steps (1/180).
* The first nonzero basin needs extra compute. A climb from the hybrid (hidden 48, lr 0.3, 200 steps) gives
  6/180 solved_and_stopped on seeds that never entered selection (17/180 reach the optimum, 103 premature
  stops). Architecture hill climbing resumes from there; the verifier-conditioned halt on the hybrid was not run.

## 9. Research trajectories (`trajectories.py`, `receipts/trajectories_20261128/`)

`ink03.research_trajectory.v1` covers question, hypothesis, evidence, proposed experiment, estimated cost,
expected informative outcome, actual result, interpretation, updated hypothesis, decision, alternatives and
certificate. Every field is tagged mechanical, rule, transcribed or curated. **1,502 trajectories:**
* 1,497 hill-climb steps;
* the audit-02 effort decision (numbers from reconciled receipts);
* the BPD campaign (numbers read from campaign.json);
* three MeTTa diagnosis steps (numbers checked against STOP_DIAGNOSIS.md).

1,116 are negative outcomes. The next-action training rows give the trajectory prefix (question, hypothesis,
evidence) and ask for the next experiment. Results and decisions never appear in the input. The Compactification
Theory search covered 300 branches and found DISTILL-11 quotient-campaign rows (`Galacticity/distill_profile.py`)
and the QO-14 certificate-distillation protocol, but no show-your-work quiz module.
