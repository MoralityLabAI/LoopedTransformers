# LS-SAE11 results: within-forward transport and solver-hard Sudoku accuracy

**Registered scientific verdict: FAIL in both prospective hard families.** `test_champagne_a`: 3/8 puzzles with paired gain D ≥ 1 (6 needed), bootstrap lower bound −1.81. `test_champagne_b`: 2/8 (6 needed), lower bound −3.94. Each family met the clean-inexact requirement (6/8 and 5/8 inexact at ACT6, 4 needed); both failed the event count and the bootstrap bound. The verdict held in a local CPU re-execution, though per-puzzle values and even clean ACT12 exactness on unsolved boards changed with the platform (see below). Nothing here changes a threshold, puzzle, vector, dose, seed or sign.

LS-SAE08's negative result and LS-SAE09's incomplete result are unchanged.

## Provenance

| Item | Value |
|---|---|
| Registration (protocol only) | `e94a4439db78eacdb4988ecd167bcfb62a91230f` |
| Code that produced the data | `09402a610e3e41b848e1fa6b1da476fad673b908` (PR head; LS-SAE11 paths identical in merge `d545f9d315476282290f3410e67030d876ac4ce6`, PR #142) |
| Protocol text between registration and data | unchanged |
| Actions run | [44042014383](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44042014383), attempt 1, all four jobs success |
| Original LS-SAE08 evidence | run [43851165980](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43851165980), artifact 14220373987, `full.json` SHA256 `69c4cd1a…a7f9` (matches registration) |
| Checkpoint | `932b5e0d…8c38` |
| Evidence copy | branch `evidence/ls-sae11-actions-44042014383`: artifact zips at `77997308cbd80e74a81288864fc959753930dc71`, written by the workflow `mechinterp-ls-sae11-evidence-recovery.yml` (main `15a5ddb`); the local re-execution zip added at `4e7744a` |

The artifacts need GitHub authentication to download. The recovery workflow copied the five artifact zips (four from run 44042014383, the LS-SAE08 full evidence from 43851165980) into the evidence branch and stopped unless each zip's SHA256 equalled the digest GitHub recorded at upload. The SHA256 of every zip was checked again locally against the digests on the public run page:

| Artifact | GitHub digest = local SHA256 |
|---|---|
| ls-sae11-frozen-20-Champagne-source-only-exact-witnesses | `465cd322…085e` |
| ls-sae11-hard-four-engineering-within-forward-native-and-audit | `fe7d3ece…30b7` |
| ls-sae11-original-16-reference-native-transport-and-audit | `e334cc9f…c0` |
| ls-sae11-hard-full-16-native-state-transport-accuracy-and-audit | `2c5665b5…095c` |
| ls-sae08-hard-52-54-native-all81-causal-evidence-and-audit | `a41f623d…263d` |

Extracted JSON SHA256: `source_20.json c502414e…5b517`, `pilot.json 66e777d2…faa`, `reference.json 2759e08e…dacd`, `hard.json 30cf6375…3c40`, LS-SAE08 `full.json 69c4cd1a…a7f9`.

## Independent audit

The registered standard-library auditor ran three times on the same model outputs:

| Run | Platform | Receipts |
|---|---|---|
| Original CI (in each job) | GitHub ubuntu-latest, Python 3.11 | `receipts/ci_artifact_audits/` |
| Linux replay (recovery workflow) | GitHub ubuntu-latest, Python 3.11 | `receipts/linux_replay/` |
| Windows replay | Windows 11, Python 3.12.10 | `receipts/windows_replay/` |

All three passed every check for the LS-SAE08 evidence, the 20-puzzle source receipt (exact unique solutions, Git blobs, singles stall), the pilot, the reference cohort and the hard cohort. The Linux replay receipts are byte-identical to the CI receipts; the Windows receipts have identical JSON content. The auditor recomputes source identity, saved 81×9 logits, integer Sudoku correctness, every strict margin certificate, every paired score, the bootstrap and the Pythagorean norm and singular-value bound checks. **It cannot reconstruct the 97×512 hidden states or rerun the model**, so the hidden-state transport numbers rest on the model job alone.

`native_withinrun_v1/analyze.py` recomputes every reported Sudoku score from the saved predicted digits and exact targets, checks every saved certificate flip count against the digit score, and replays the registered gate with the registered bootstrap. Its gate statistics are identical to the saved and audited ones for both families.

## Failed runs (preserved)

Two earlier runs of the same workflow failed at the unit-test step, before any model or source stage, so no data came from them:

| Run | Head | Failed step | Cause (reproduced locally) |
|---|---|---|---|
| [44041816665](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44041816665) (#1) | `de62e25` | Unit regression (job 133897938775) | `IndexError: index 25 is out of bounds for axis 0 with size 25` in `run_case`: `ref_idx=idx+1` read the row after each signed variant |
| [44041953396](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44041953396) (#2) | `0993340` | Unit regression (job 133898386461) | same off-by-one; `0993340` only changed the toy readout tie |

`09402a6` set `ref_idx=idx` (clean is row 0, edits rows 1..24); its five unit tests pass locally under PyTorch 2.7.1 CPU. Had the off-by-one not raised, every edited row would have been scored as the next variant's output. The fix is already on main, so no repair branch was needed. The job logs need authentication; the public check-run annotations show only `Process completed with exit code 1`, and the cause above comes from rerunning the unit suite at each head.

## Engineering pilot (excluded from the gate)

Four Champagne puzzles, donors `test_52[0,1]` and `test_54[0,1]`.

| Seed | Holes | Clean blank-correct ACT3 / ACT6 / ACT12 | ACT12 exact |
|---|---:|---|---|
| 100153 | 58 | 20 / 21 / 22 | no |
| 100371 | 58 | 31 / 58 / 58 | yes |
| 100587 | 58 | 22 / 23 / 27 | no |
| 10080 | 59 | 33 / 31 / 30 | no |

No ±8.00 edit had a sufficient global margin certificate at ACT6.

## Retrospective original reference (16 LS-SAE08 boards, diagnostic only)

Every reference board is solved exactly at ACT6 and ACT12; at ACT3 one board (`test_54[3]`, seed 202628040300) has 30/54 correct, the other 15 are exact. This cohort tests transport on solved boards, not accuracy.

- **Contraction.** Median hidden-state gain `G_H` for the donor's own STATE_POWER and LOGIT_POWER edits is ≈ 0.5 at ACT3 and 0.07–0.14 at ACT6/ACT12; SAE and RANDOM edits contract to 0.003–0.034. Between 11 and 30 of 32 signed edits per arm and dose are classed contracted (`G_H < 0.1`) at ACT6.
- **Certificates.** At ACT6 and ACT12, 31–32 of 32 signed edits in every arm and dose carry a sufficient global margin certificate.
- **Flips.** Every flip in the reference cohort is on `test_54[3]` or `test_54[6]` at ACT3, except one: SAE −0.20 on `test_54[3]` at ACT6 corrupts 20 cells (`G_H` = 951). On `test_54[3]` at ACT3, edits only repair (SAE −8.00 repairs 24 cells); on `test_54[6]` at ACT3, STATE_POWER and LOGIT_POWER −8.00 corrupt 10 and 12 cells.
- **Cross-run drift against the LS-SAE08 archive** (diagnostic, no threshold): clean logits differ by at most 0.0043 / 0.0108 / 0.0008 at ACT3 / ACT6 / ACT12; edited logits by at most 0.0089 / 0.385 / 0.071. No argmax digit changed in any of 384 variants per horizon. The ACT6 drift is far above the 0.0002 and 0.001 cross-run tolerances that left LS-SAE09 incomplete, which confirms that exact cross-run logit replay is not a usable gate on this hardware.

## Prospective hard test (16 Champagne puzzles, registered gate)

Clean accuracy (blank cells correct / total):

| Family | ACT3 | ACT6 | ACT12 | Inexact at ACT6 | Exact at ACT12 |
|---|---|---|---|---:|---:|
| `test_champagne_a` | 232/460 | 277/460 | 306/460 | 6/8 | 3/8 |
| `test_champagne_b` | 298/460 | 321/460 | 373/460 | 5/8 | 5/8 |

Registered endpoint at ACT6, dose 8.00: `C_arm = ½[(correct(+) − correct(clean)) + (correct(−) − correct(clean))]`, `D = C_STATE_POWER − C_RANDOM`. All donor directions were eligible for all 16 puzzles, so no `C = 0` substitution occurred.

| Family | Seed | Clean ACT6 | C_state | C_random | **D** | Mean flips state / random | `G_H` state / random | Rowspace fraction state / random |
|---|---|---|---:|---:|---:|---|---|---|
| A | 100804 | 25/57 | −3.0 | 0.0 | **−3.0** | 9.0 / 8.0 | 5.3 / 4.7 | 0.56 / 0.58 |
| A | 101034 | 57/57 | 0.0 | −12.0 | **12.0** | 0.0 / 12.0 | 1.9 / 17.7 | 0.31 / 0.48 |
| A | 101250 | 28/58 | −3.0 | −7.5 | **4.5** | 9.0 / 11.5 | 6.3 / 7.5 | 0.61 / 0.52 |
| A | 101465 | 57/57 | 0.0 | 0.0 | **0.0** | 0.0 / 0.0 | 0.01 / 0.01 | 0.31 / 0.29 |
| A | 101676 | 29/58 | −3.0 | 3.0 | **−6.0** | 10.5 / 12.0 | 7.4 / 7.5 | 0.55 / 0.55 |
| A | 101887 | 23/58 | 23.0 | 23.5 | **−0.5** | 24.5 / 24.5 | 17.3 / 17.6 | 0.67 / 0.66 |
| A | 102107 | 31/58 | 12.5 | −1.0 | **13.5** | 18.0 / 8.0 | 15.0 / 3.4 | 0.72 / 0.64 |
| A | 102318 | 27/57 | −3.5 | −2.0 | **−1.5** | 11.0 / 8.0 | 5.4 / 3.3 | 0.66 / 0.64 |
| B | 102531 | 28/57 | −3.0 | −2.5 | **−0.5** | 4.5 / 6.0 | 7.2 / 5.7 | 0.41 / 0.56 |
| B | 102766 | 58/58 | 0.0 | 0.0 | **0.0** | 0.0 / 0.0 | 0.02 / 0.04 | 0.29 / 0.30 |
| B | 10290 | 59/59 | 0.0 | 0.0 | **0.0** | 0.0 / 0.0 | 3.0 / 1.6 | 0.29 / 0.28 |
| B | 103012 | 33/57 | 11.0 | 24.0 | **−13.0** | 18.0 / 24.0 | 16.0 / 25.6 | 0.67 / 0.79 |
| B | 103235 | 27/57 | 14.5 | 1.0 | **13.5** | 19.5 / 9.5 | 17.2 / 5.0 | 0.62 / 0.60 |
| B | 103448 | 25/57 | −3.0 | −0.5 | **−2.5** | 12.0 / 13.5 | 5.1 / 6.3 | 0.60 / 0.66 |
| B | 103661 | 33/57 | 9.5 | −3.0 | **12.5** | 18.5 / 9.0 | 17.0 / 6.5 | 0.63 / 0.59 |
| B | 103875 | 58/58 | 0.0 | 0.0 | **0.0** | 0.0 / 0.0 | 0.16 / 0.07 | 0.30 / 0.28 |

| Registered gate | `test_champagne_a` | `test_champagne_b` |
|---|---|---|
| Clean ACT6 inexact (≥ 4) | 6 ✓ | 5 ✓ |
| Puzzles with D ≥ 1 (≥ 6) | 3 ✗ | 2 ✗ |
| Mean D | 2.375 | 1.25 |
| Bootstrap 95 % (2000 draws), lower bound > 0 | [−1.8125, 7.125] ✗ | [−3.9375, 6.5] ✗ |
| **Family** | **FAIL** | **FAIL** |

### Failure cases

Eleven of 16 puzzles have D < 1. They fall into three groups:
1. **Solved at ACT6, D = 0 by construction** (A 101465; B 102766, 10290, 103875). Nothing can improve; the only way to score is for RANDOM to break the board.
2. **D ≥ 1 only because RANDOM corrupted a solved board** (A 101034: RANDOM −12, STATE 0). This counts as a success event under the registered rule but is not an accuracy gain.
3. **Stuck or recovering boards where STATE_POWER did no better than RANDOM** (A 100804, 101676, 101887, 102318; B 102531, 103012, 103448). On the two boards the clean model solves between ACT6 and ACT12 (A 101887, B 103012), large edits of every arm pull that recovery earlier: C ≈ +23 for both STATE and RANDOM on 101887, +11 STATE vs +24 RANDOM on 103012.

The three clean-exact boards in family B leave at most five boards on which STATE_POWER could truly improve, one short of the six required; family B could pass the event count only with RANDOM corruptions of solved boards. This is a property of the registered gate, recorded here without amending it.

### Secondary contrasts (no gate)

`ls_sae11_family_gates.csv` applies the registered rule to STATE, LOGIT and SAE minus RANDOM at every horizon and dose, 27 contrasts per family. **None passes.** The closest is LOGIT_POWER − RANDOM at ACT6, dose 8.00 in family A: mean 4.81, bootstrap [0.81, 9.31], but only 4/8 puzzles with D ≥ 1.

### Transport, certificates and delayed recovery

- **Unsolved boards amplify any edit.** At ACT6 the median `G_H` of a ±0.20 ACT1 edit is 56–85 depending on arm (maximum ≈ 1030), 17–21 at dose 2.00 and 4.8–5.3 at dose 8.00. The resulting displacement `G_H × dose` (≈ 16, 34, 42) depends weakly on dose and not on direction. Solved boards contract (`G_H` 0.01–0.16).
- **Rowspace.** About half of the board displacement lies in the centered 9-digit readout rowspace on the hard cohort (median 0.50–0.57 at ACT6), against ≈ 0.30 on the solved reference boards.
- **Certificates track solved boards after ACT3.** At ACT6, 9–11 of 32 signed edits per arm and dose carry a sufficient global certificate; at ACT12, 14–16. All but one are on clean-exact boards (5 at ACT6 and 8 at ACT12, × 2 signs); the exception is RANDOM −0.20 on 102318 at ACT6 (margin 0.82 > 2 × 0.41). At ACT3, 24 small edits (mostly dose 0.20) are certified on four unsolved boards (101887, 102318, 102531, 103235), each with zero flips.
- **Delayed recovery.** Between ACT6 and ACT12 the clean model recovers 35 cells on 101887, 24 on 103012, 24 on 103661 and 9 on 103235, while stuck boards churn, recovering 1–5 cells and losing 2–6. Edits of every arm make **44 signed edits newly exact at ACT12** on five stuck boards (A 100804, 101250, 101676, 102107; B 103235) — STATE 11, LOGIT 11, SAE 12, RANDOM 10 — and **break 11 signed edits** on three boards the clean model solves by ACT12 (A 101887; B 103012, 103661). At dose 8.00, STATE_POWER repairs 185 and corrupts 81 cells at ACT6 versus 147 and 101 for RANDOM; 169 of the STATE repairs persist to ACT12.

The pattern is that an ACT1 edit re-rolls the trajectory of an unsolved board. Whether the re-roll lands on a solution depends on the board, not on which direction started it.

## Local CPU re-execution

The registered hard stage was re-executed with the unchanged code on this laptop: Windows 11, AMD Ryzen AI 9 365, Python 3.12.10, PyTorch 2.7.1+cpu, NumPy 2.1.3, 4 threads, MATH attention (CI used Python 3.11 on a GitHub ubuntu-latest runner). It ran 2288 s and its output passed the same independent auditor. `native_withinrun_v1/compare_reexecution.py` compares it with the CI output variant by variant (`ls_sae11/local_reexecution/`).

**The registered verdict reproduces; the per-puzzle numbers do not.**

| | CI (run 44042014383) | Local re-execution |
|---|---|---|
| `test_champagne_a`: D ≥ 1 / bootstrap 95 % | 3/8, [−1.81, 7.13] | 3/8, [−1.56, 7.50] |
| `test_champagne_b`: D ≥ 1 / bootstrap 95 % | 2/8, [−3.94, 6.50] | 2/8, [−4.19, 6.25] |
| Registered gate | FAIL, FAIL | FAIL, FAIL |
| D that differ | — | A: 100804 (−3.0 → −1.5), 102107 (13.5 → 14.5); B: 102531 (−0.5 → −1.5), 103448 (−2.5 → −3.0), 103661 (12.5 → 12.0) |

| Horizon | Max \|Δlogit\| (all variants) | Variants with any argmax change (of 400) | Clean boards with argmax change (of 16) |
|---|---:|---:|---:|
| ACT3 | 0.17 | 1 | 0 |
| ACT6 | 35.8 | 62 | 4 |
| ACT12 | 88.1 | 172 | 7 |

The clean model itself behaves differently on unsolved boards. At ACT12 it solves **102107 exactly locally (58/58) but not in CI (29/58)**, and likewise **103235 (57/57 locally, 36/57 in CI)**; on 100804, 101250, 101676, 102531 and 103448 it disagrees with CI on 7–11 cells. On the five boards solved by ACT6 the clean logits agree to within 0.0018 at every horizon. The differences start below 0.015 in the clean ACT3 logits and the recurrence amplifies them on unsolved boards, the same amplification measured for the deliberate edits.

So the LS-SAE11 hard-cohort outcome, every clean ACT12 accuracy on an unsolved board, and every per-puzzle D are properties of the model *on a specific floating-point platform*. The family-level verdict is robust to the platform change tested here; per-puzzle effects and clean ACT12 exactness on unsolved boards are not. The same holds for LS-SAE09's failed cross-run replay: a 0.001 tolerance cannot hold on these boards.

## Files

- `results/ls_sae11/ls_sae11_per_puzzle_clean_accuracy.csv` — every puzzle in the three stages: clean ACT3/6/12 accuracy, exactness, validity, solver nodes, singles, clean ACT3→6 and ACT6→12 recoveries.
- `results/ls_sae11/ls_sae11_per_puzzle_signed_edits.csv` — 2592 rows, one per stage × puzzle × horizon × arm × dose × sign: correctness, flips, repairs, corruptions, every transport and rowspace number, both certificates, mechanism labels.
- `results/ls_sae11/ls_sae11_per_puzzle_structural_vs_random.csv` — sign-symmetrized `C` for all arms and the paired contrasts at every horizon and dose.
- `results/ls_sae11/ls_sae11_per_puzzle_delayed_recovery.csv` — errors at ACT3/6/12 per signed edit, edit-induced ACT6 repairs/corruptions and whether they persist or recover by ACT12.
- `results/ls_sae11/ls_sae11_family_gates.csv` — registered rule for every contrast × horizon × dose; the primary row is flagged.
- `results/ls_sae11/ls_sae11_reference_cross_run_drift.csv`, `ls_sae11_aggregate_transport_certificates.csv`, `ls_sae11_analysis_summary.json`.
- `results/ls_sae11/receipts/` — artifact manifest with digests, source receipt, the three sets of audit receipts.
- `results/ls_sae11/local_reexecution/` — per-variant local-vs-CI comparison (1200 rows) and summary.
- Code: `native_withinrun_v1/analyze.py` (tables), `native_withinrun_v1/compare_reexecution.py` (re-execution comparison), `.github/workflows/mechinterp-ls-sae11-evidence-recovery.yml`.

Reproduce the tables from the evidence branch:

```bash
git fetch origin evidence/ls-sae11-actions-44042014383
git archive FETCH_HEAD | tar -x -C /tmp/ev && cd /tmp/ev && for z in ls-sae1*.zip; do unzip -o -q "$z" -d x; done
python -m mechinterp.sae_causal_transfer.native_withinrun_v1.analyze --evidence /tmp/ev/x --original /tmp/ev/x/full.json --out mechinterp/sae_causal_transfer/results/ls_sae11
```
