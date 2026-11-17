# LS-SAE12 results: mid-reasoning interventions scored by verified future deductions

**Registered outcome: FAIL.** In both prospective families, 8 of 12 states have a positive primary contrast D_SI = V_structural − V_isotropic at dose 0.02. The registered gate requires at least 9 of 12, and both families must pass. The bootstrap criterion holds in both families: lower bounds are 10.39 (A) and 4.73 (B). The event criterion fails in both, by one state each. Pool, site rules, seeds, doses, outcome and gate are those registered in `f73d40b383f981bdbbbe96e5b3236a520a597414` ([protocol](../active_reasoning_v1/PROTOCOL.md)); none was changed after any data was seen. The LS-SAE08 negative, LS-SAE09 incomplete and LS-SAE11 failed results are untouched.

A local CUDA re-execution of the same stage, declared before any outcome, also fails overall, but there family B passes (9/12). Across the two platforms D_SI keeps its sign on only 12 of the 21 states they share. The contrast this test measures is not stable to float-level platform differences on unsolved boards (see [Platform reproductions](#platform-reproductions)).

## Provenance and audit

| Item | Value |
|---|---|
| Registration | `f73d40b`; source receipt `4e580db` (48/48 admitted under Algorithm X and an independent bitmask DFS) |
| Registered run | GitHub Actions run 44084948890, attempt 1, workflow commit `74b8882d727ac8da1fae8f735c800656afda4aab`, push event |
| Jobs | unit tests + source admission: success (12:37:56Z–12:39:28Z); engineering pilot: success; registered screening + `active_a` + `active_b`: success (12:53:32Z–14:22:12Z, 89 min, one process); publish evidence: success |
| Runner | Linux 7.0.0-1019-azure, AMD EPYC 9V45, Python 3.11.18, PyTorch 2.7.1+cpu, NumPy 2.1.3, 4 threads |
| Evidence | branch `evidence/ls-sae12-run-44084948890` at `70c765c`; all 8 gzipped receipts match its `SHA256SUMS` locally ([copy](ls_sae12/receipts/ci_44084948890/)) |
| Source receipt | the CI `source_48.json` is byte-identical to the committed [`source_48.json`](ls_sae12/source_48.json) (also enforced by `cmp` in the workflow) |
| CI audit | `audit.py` passes source, screening and both families ([active_a](ls_sae12/receipts/ci_44084948890/active_a-audit.json), [active_b](ls_sae12/receipts/ci_44084948890/active_b-audit.json)) |
| Windows replay | the registered auditor run unchanged on Windows fails with `isotropic vectors do not regenerate from their seeds` ([log](ls_sae12/receipts/windows_replay/active_a-audit.registered-exact.FAILED.log)); cause and repair below; the repaired replay is byte-identical to the CI audit for both families |
| Tables | `ls_sae12_analysis/analyze.py` re-derives both gates from the per-state contrasts with the registered bootstrap seeds and checks them against the saved statistics |

The auditor recomputes from saved digits, logits, seeds and vectors: source admission, screening schedules, eligibility, both site rules, family assignment, every per-row outcome, every contrast and both gates. It does not re-execute hidden states or Jacobian power iterations.

### Reproducibility failure: the auditor's isotropic check is platform-specific

`audit.py` regenerates each isotropic direction with `random.Random(seed).gauss` and requires exact float64 equality with the saved vector. Python's `gauss` calls the platform math library. Regenerated on Windows (Python 3.12.10), 148,157 of 196,608 components differ from the Linux runner's (Python 3.11.18) by at most 9 ulp (max |difference| 1.4e-16). The model never sees these differences: the edit is `dose × sign × vector` cast to float32, and that cast is bit-identical for all 1,179,648 component × dose × sign combinations. [`ls_sae12_analysis/replay_audit.py`](../ls_sae12_analysis/replay_audit.py) checks that every saved vector is within 16 ulp of the local regeneration and gives a bit-identical float32 edit, then runs the registered auditor unchanged on the verified saved vectors. The registered `audit.py` is not modified, because a change under `active_reasoning_v1/` would start another registered run. Its exact check passes only on a platform whose math library matches the one that wrote the receipt.

## Screening

- 48 puzzles screened from clean ACT1–ACT16 trajectories; 36 eligible (a cell wrong after ACT4 whose verified deduction step T is 6–16), 12 ineligible.
- All 12 ineligible puzzles are solved exactly by ACT16; 8 of the 36 eligible are ([table](ls_sae12/ls_sae12_screening.csv)).
- Families by registered pool order: A = positions 20–30 and 33; B = 35, 36, 37, 40–44, 46, 47, 49, 51. The remaining 12 eligible puzzles are unused.
- In every one of the 48 registered sites (24 states × 2 site rules), the batch-61 clean row reproduces the batch-1 screening digits at all 12 continuation steps, and the site's clean T is the same in both.

## Primary gate (observable-error site, dose 0.02)

| Family | States with D_SI > 0 | Mean D_SI | Bootstrap 95% (2000 draws) | ≥ 9/12 | Lower bound > 0 | Gate |
|---|---|---|---|---|---|---|
| A (seed 2026111813) | 8/12 | 46.83 | [10.39, 88.38] | no | yes | **fail** |
| B (seed 2026111814) | 8/12 | 36.39 | [4.73, 76.42] | no | yes | **fail** |

V is the mean verified deduction displacement Σ|T′ − T| over all blank cells (in ACT steps) across an arm's edits: 2 signs for STRUCTURAL and LOGIT, 8 draws × 2 signs for ISOTROPIC. T is the selected site's clean verified deduction step.

| Family | Pos | File | Site | T | V_struct | V_logit | V_iso | D_SI | cos(s,l) | Clean ACT16 correct |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 20 | 104087.txt | 19 | 12 | 96 | 56 | 59.56 | +36.44 | 0.9975 | 25/57 |
| A | 21 | 104303.txt | 63 | 8 | 30.5 | 44 | 43.88 | −13.38 | 0.9985 | 33/57 |
| A | 22 | 104520.txt | 19 | 10 | 99.5 | 36 | 74.25 | +25.25 | 0.9976 | 34/60 |
| A | 23 | 104738.txt | 33 | 12 | 93.5 | 82.5 | 131.44 | −37.94 | 0.9954 | 29/59 |
| A | 24 | 104951.txt | 59 | 6 | 0 | 0 | 0.00 | 0.00 | 0.9984 | 59/59 |
| A | 25 | 1050.txt | 61 | 15 | 51 | 50 | 42.25 | +8.75 | 0.9974 | 22/60 |
| A | 26 | 10500.txt | 4 | 9 | 176 | 166.5 | 93.19 | +82.81 | 0.9968 | 28/59 |
| A | 27 | 105168.txt | 29 | 10 | 315.5 | 323.5 | 116.94 | +198.56 | −0.9747 | 27/58 |
| A | 28 | 105386.txt | 37 | 8 | 38.5 | 38.5 | 0.31 | +38.19 | 0.9998 | 57/57 |
| A | 29 | 105645.txt | 11 | 15 | 119.5 | 114.5 | 85.94 | +33.56 | 0.9951 | 26/58 |
| A | 30 | 105890.txt | 68 | 15 | 252 | 261.5 | 62.25 | +189.75 | 0.9971 | 23/58 |
| A | 33 | 106590.txt | 2 | 6 | 0 | 0 | 0.00 | 0.00 | 0.9762 | 58/58 |
| B | 35 | 107037.txt | 18 | 14 | 99.5 | 87 | 44.19 | +55.31 | 0.9993 | 34/58 |
| B | 36 | 10710.txt | 31 | 14 | 57 | 63 | 51.75 | +5.25 | 1.0000 | 21/59 |
| B | 37 | 107263.txt | 74 | 13 | 65.5 | 75 | 74.50 | −9.00 | −0.9978 | 34/58 |
| B | 40 | 107943.txt | 19 | 14 | 248.5 | 257.5 | 126.38 | +122.12 | 0.9998 | 35/58 |
| B | 41 | 108168.txt | 56 | 6 | 0 | 0 | 0.00 | 0.00 | 0.9921 | 57/57 |
| B | 42 | 108388.txt | 62 | 12 | 304 | 242.5 | 89.31 | +214.69 | 0.9938 | 26/57 |
| B | 43 | 108613.txt | 53 | 15 | 81 | 100 | 93.81 | −12.81 | 0.9986 | 55/57 |
| B | 44 | 108847.txt | 4 | 16 | 47.5 | 35.5 | 39.00 | +8.50 | 0.9989 | 28/57 |
| B | 46 | 10920.txt | 17 | 15 | 79.5 | 68 | 43.31 | +36.19 | 0.9994 | 31/59 |
| B | 47 | 109293.txt | 24 | 11 | 39 | 26 | 28.00 | +11.00 | 0.9996 | 18/58 |
| B | 49 | 109726.txt | 4 | 11 | 66.5 | 88 | 69.44 | −2.94 | 0.9647 | 57/57 |
| B | 51 | 110174.txt | 11 | 16 | 59 | 64.5 | 50.69 | +8.31 | 0.9985 | 24/57 |

### Failure cases

- **Ties at zero (A 24, A 33, B 41).** The clean model solves these boards exactly by ACT16, and no dose-0.02 edit in any arm moves any verified deduction. D_SI = 0 does not count as positive. Family A has two such ties and two negative states; family B has one tie and three negative states.
- **Negative states (A 21, A 23; B 37, B 43, B 49).** Isotropic edits move more deductions than the structural direction. The largest gap is A 23: isotropic 131.4 vs structural 93.5.
- **Solved board that does move (A 28).** The clean model solves it exactly, yet the structural and logit directions move 38.5 deductions while isotropic edits move 0.31.

## Secondary contrasts (registered as descriptive, not gates)

The same rule (≥ 9/12 positive and a positive bootstrap lower bound) applied to every other site rule × dose × contrast cell ([table](ls_sae12/ls_sae12_family_gates.csv)):

| Family | Site rule | Dose | Contrast | States > 0 | Mean | Bootstrap 95% | Meets the rule |
|---|---|---|---|---|---|---|---|
| A | model-blind median | 0.02 | L − I | 5/12 | −8.72 | [−28.96, 9.14] | no |
| A | model-blind median | 0.02 | S − I | 6/12 | 2.73 | [−21.30, 27.00] | no |
| A | model-blind median | 0.02 | S − L | 3/12 | 11.46 | [−4.92, 33.33] | no |
| A | observable error | 0.02 | L − I | 7/12 | 38.58 | [−2.12, 86.63] | no |
| A | observable error | 0.02 | S − I **(registered primary)** | 8/12 | 46.83 | [10.39, 88.38] | no |
| A | observable error | 0.02 | S − L | 6/12 | 8.25 | [−2.42, 21.42] | no |
| A | observable error | 0.25 | L − I | 8/12 | 48.86 | [−1.98, 98.42] | no |
| A | observable error | 0.25 | S − I | 9/12 | 48.86 | [−8.03, 106.40] | no |
| A | observable error | 0.25 | S − L | 8/12 | 0.00 | [−51.42, 46.62] | no |
| A | observable error | 2.00 | L − I | 10/12 | 37.93 | [7.58, 75.17] | yes |
| A | observable error | 2.00 | S − I | 8/12 | 33.89 | [−4.96, 88.92] | no |
| A | observable error | 2.00 | S − L | 2/12 | −4.04 | [−25.29, 22.42] | no |
| B | model-blind median | 0.02 | L − I | 9/12 | 45.65 | [10.92, 85.05] | yes |
| B | model-blind median | 0.02 | S − I | 7/12 | 34.06 | [3.03, 68.49] | no |
| B | model-blind median | 0.02 | S − L | 3/12 | −11.58 | [−30.79, 4.33] | no |
| B | observable error | 0.02 | L − I | 9/12 | 33.05 | [6.97, 64.77] | yes |
| B | observable error | 0.02 | S − I **(registered primary)** | 8/12 | 36.39 | [4.73, 76.42] | no |
| B | observable error | 0.02 | S − L | 5/12 | 3.33 | [−6.96, 15.83] | no |
| B | observable error | 0.25 | L − I | 8/12 | 33.80 | [4.22, 68.43] | no |
| B | observable error | 0.25 | S − I | 7/12 | 21.13 | [−7.16, 54.29] | no |
| B | observable error | 0.25 | S − L | 4/12 | −12.67 | [−30.04, 4.38] | no |
| B | observable error | 2.00 | L − I | 5/12 | 27.05 | [−25.67, 95.71] | no |
| B | observable error | 2.00 | S − I | 6/12 | 18.30 | [−4.71, 44.23] | no |
| B | observable error | 2.00 | S − L | 6/12 | −8.75 | [−55.67, 24.79] | no |

- 3 of the 22 non-primary cells meet the rule: A observable-error L − I at dose 2.00, and B L − I at dose 0.02 at both sites. These are uncorrected secondary cells.
- **The observable-error site is not special in family B.** At the model-blind median site, S − I in B has mean 34.06 and bootstrap [3.03, 68.49] (7/12), close to the selected-site value of 36.39. In A the control site gives 2.73 [−21.30, 27.00].
- **The STRUCTURAL and LOGIT directions nearly coincide.** |cos| ranges from 0.965 to 1.000 (median 0.998) over the 48 sites, so the S − L contrast is near-degenerate, as the engineering pilot predicted. The two edits differ by at most about 0.0053 in L2 at dose 0.02. Even so, their V values differ by up to 117 deductions (A 29, control site: 189 vs 72). Verified-deduction displacement is not a smooth function of the edit direction at this scale.

## Dose response and the isotropic floor

Mean V by arm, observable-error site:

| Family | Dose | STRUCTURAL | LOGIT | ISOTROPIC |
|---|---|---|---|---|
| A | 0.02 | 106.0 | 97.8 | 59.2 |
| A | 0.25 | 134.5 | 134.5 | 85.6 |
| A | 2.00 | 133.2 | 137.3 | 99.4 |
| B | 0.02 | 95.6 | 92.2 | 59.2 |
| B | 0.25 | 106.0 | 118.6 | 84.8 |
| B | 2.00 | 125.2 | 134.0 | 106.9 |

Dose 0.02 is 0.09% of the site token's norm (√512 ≈ 22.6). Even so, 146 of 192 (A) and 176 of 192 (B) isotropic edit rows move at least one verified deduction, with a median of 50.5 and 48.5 step-units per row. Raising the dose 100-fold raises the isotropic mean by less than a factor of 2.

## Transport

The Spearman correlation between the edit's hidden-state gain ‖Δz_H‖/dose and its VDD (selected site, dose 0.02, all edit rows) is low after one step and rises as the trajectories diverge:

| Family | ACT5 | ACT8 | ACT16 |
|---|---|---|---|
| A | 0.21 | 0.51 | 0.78 |
| B | 0.10 | 0.60 | 0.62 |

The immediate effect of an edit barely predicts how many deductions it moves. How far the trajectory has diverged by the end predicts it well.

## Platform reproductions

Two local runs of the same `--stage full` were declared in [STATUS.md](ls_sae12/STATUS.md) before any family outcome on any platform was seen: Windows CPU, PyTorch 2.7.1+cpu, 4 threads (`2092288`); and the RTX 5080 Laptop GPU through [`run_native_cuda.py`](../ls_sae12_analysis/run_native_cuda.py), PyTorch 2.11.0+cu128 (`22966d8`). They do not replace the registered outcome. Each is compared with the Actions run by [`compare_runs.py`](../ls_sae12_analysis/compare_runs.py) per pool position and per state. Full receipts are on branch `evidence/ls-sae12-local-reexecutions`. In the engineering pilot, D_SI changed sign between CPU and CUDA on 2 of 3 states (`ls_sae12/engineering_pilot.cpu_vs_cuda.local.txt`).

### Local CUDA (RTX 5080 Laptop GPU, PyTorch 2.11.0+cu128): the family verdicts change

The run took 14:04:33Z–14:37:56Z. The registered `audit.py` passes both families on the CUDA receipts ([A](ls_sae12/platform/active_a-audit.cuda.json), [B](ls_sae12/platform/active_b-audit.cuda.json)). Receipts are in `cuda/` on the evidence branch at `062da2a`; the comparison is in [`compare_actions_vs_cuda.csv`](ls_sae12/platform/compare_actions_vs_cuda.csv) and [`.json`](ls_sae12/platform/compare_actions_vs_cuda.json).

| | Actions CPU (registered) | Local CUDA |
|---|---|---|
| Eligible | 36/48 | 33/48 (30, 42, 53 drop out) |
| Clean exact at ACT16 | 20/48 | 19/48 |
| Family A members | 20–30, 33 | 20–29, 33, 35 (11 shared) |
| Family A primary | 8/12, mean 46.83, [10.39, 88.38]: fail | 8/12, mean 21.32, [−3.35, 48.07]: fail |
| Family B members | 35–37, 40–44, 46, 47, 49, 51 | 36, 37, 40, 41, 43, 44, 46, 47, 49, 51, 54, 55 (10 shared) |
| Family B primary | 8/12, mean 36.39, [4.73, 76.42]: fail | 9/12, mean 31.75, [11.27, 56.27]: **pass** |
| Registered outcome | FAIL | FAIL (A fails) |

- **Clean screening diverges exactly on the unsolved boards.** Clean digits are identical over all 16 steps on 18 of 48 puzzles, and all 18 are boards that the Actions run solves exactly by ACT16. The other 30 first diverge between ACT4 and ACT11 (histogram in the JSON); only 2 of them are solved exactly on Actions.
- **Per-state signs are close to a coin flip.** Among the 21 states present in both runs, D_SI keeps its sign in 12. Excluding the four boards that both platforms solve identically (A 24, A 28, A 33, B 41: identical D_SI), it keeps its sign in 8 of 17. The selected site differs in 9 of the 21 shared states, even though both platforms apply the same rule.
- **On CUDA the batch changes the clean run.** The batch-61 clean row reproduces the batch-1 screening digits at all 12 continuation steps in only 9 of 48 sites (minimum 3 steps). The selected site's same-forward clean T differs from its screening T in 14 of 24 states. The outcome is still defined inside one batch, as registered, but on this device the clean trajectory that the edits are scored against is not the one that defined eligibility.
- **Confound:** PyTorch is 2.11.0 here and 2.7.1 on Actions, so device and library version are not separated.

| Family | Pos | Site (Actions / CUDA) | D_SI Actions | D_SI CUDA | Same sign | CUDA batch-61 clean = screening (of 12) |
|---|---|---|---|---|---|---|
| A | 20 | 19 / 59 | +36.44 | +10.19 | yes | 5 |
| A | 21 | 63 / 63 | −13.38 | +14.19 | **no** | 8 |
| A | 22 | 19 / 15 | +25.25 | −59.50 | **no** | 5 |
| A | 23 | 33 / 33 | −37.94 | +145.62 | **no** | 5 |
| A | 24 | 59 / 59 | 0.00 | 0.00 | yes | 12 |
| A | 25 | 61 / 54 | +8.75 | +18.75 | yes | 10 |
| A | 26 | 4 / 4 | +82.81 | +16.56 | yes | 8 |
| A | 27 | 29 / 6 | +198.56 | +57.69 | yes | 8 |
| A | 28 | 37 / 37 | +38.19 | +38.19 | yes | 12 |
| A | 29 | 11 / 57 | +33.56 | −5.19 | **no** | 6 |
| A | 33 | 2 / 2 | 0.00 | 0.00 | yes | 12 |
| B | 36 | 31 / 31 | +5.25 | −4.50 | **no** | 7 |
| B | 37 | 74 / 53 | −9.00 | +134.50 | **no** | 7 |
| B | 40 | 19 / 57 | +122.12 | +71.44 | yes | 3 |
| B | 41 | 56 / 56 | 0.00 | 0.00 | yes | 12 |
| B | 43 | 53 / 53 | −12.81 | +64.56 | **no** | 5 |
| B | 44 | 4 / 45 | +8.50 | −13.06 | **no** | 5 |
| B | 46 | 17 / 17 | +36.19 | +11.88 | yes | 9 |
| B | 47 | 24 / 24 | +11.00 | +10.06 | yes | 7 |
| B | 49 | 4 / 4 | −2.94 | +45.25 | **no** | 7 |
| B | 51 | 11 / 15 | +8.31 | +4.31 | yes | 7 |

### Local CPU (Windows, PyTorch 2.7.1+cpu, 4 threads)

This run is still going; its comparison is added here when it finishes.

### What the platform comparison means for the result

The registered FAIL stands as the outcome of the registered run. The CUDA re-execution shows that the per-state primary contrast does not survive a change of platform on unsolved boards, and that one family's gate verdict flips. The registered contrast is therefore not a stable property of the model on these boards. The isotropic floor, the transport correlations and the cross-platform divergence point the same way: at these doses VDD mostly measures that an edit moved the trajectory onto a different path, not which direction it was moved in.

## Files

- [`ls_sae12/ls_sae12_screening.csv`](ls_sae12/ls_sae12_screening.csv): 48 pool puzzles: eligibility, sites, clean blanks correct by step, solver nodes.
- [`ls_sae12/ls_sae12_per_state.csv`](ls_sae12/ls_sae12_per_state.csv): 96 rows (24 states × site rule × dose): V, D, recoveries, delays, site shift, hidden gain by step.
- [`ls_sae12/ls_sae12_per_variant.csv`](ls_sae12/ls_sae12_per_variant.csv): 1,920 edit rows: outcome, transport, margin certificates and blank flips at ACT5/8/16.
- [`ls_sae12/ls_sae12_family_gates.csv`](ls_sae12/ls_sae12_family_gates.csv), [`ls_sae12/ls_sae12_analysis_summary.json`](ls_sae12/ls_sae12_analysis_summary.json).
- [`ls_sae12/receipts/`](ls_sae12/receipts/): CI audits, `RUN.json`, `SHA256SUMS`, and the Windows replay (failed exact check, isotropic cross-platform check, repaired audits).
- [`ls_sae12/platform/`](ls_sae12/platform/): Actions vs local CUDA comparison (CSV, JSON), CUDA audits and log.
- Full receipts (`screen.json`, `active_a.json`, `active_b.json`, pilot) are on branch `evidence/ls-sae12-run-44084948890`; the local re-executions are on `evidence/ls-sae12-local-reexecutions`.
