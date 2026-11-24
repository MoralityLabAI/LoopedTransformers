# OLR-08A results: reissue custody, gate outcomes and spent cohorts

2026-11-24. Branch `research/olr-obligation-ledger-20261123`, PR #318, starting commit
ded064c8a993fea70c2d7033b064cd09ae788d19.

**The OLR-08 CPU campaign was not run.** OLR08_CPU_gym.zip (210361 bytes, SHA-256
e5e4b26af9837a828b743a9479e396621fbec2d6f3e846711a02c639d53750cb) was not attached to the
work order and is not on the run machine. Its package_manifest.json, RUN.md, hardened
Session/Trace/Main Lake project, roster and graph-reader code exist only inside it, so the
package import, Session build and smoke, roster audit, S1/S2/S3/S3c search, calibration
reference and fitting were not started. No new Lean outcome was collected and no model was
trained. The prerequisites that do not depend on the package are complete and recorded below.

## 1. 07A/07B exact reissue: run history

Workflow [olr07-scope-reissue.yml](../../.github/workflows/olr07-scope-reissue.yml).

| Run | Head | Outcome | Cause |
| --- | --- | --- | --- |
| [45383957646](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45383957646) | c1c787e9 | failed | Trace.lean used the reserved keyword `scoped`; fixed in afb4870b |
| [45387408849](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45387408849) | afb4870b | failed | sentinel demanded tactic-level rejection of local-recursion closure; refined in d5336d7e |
| [45388143553](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45388143553) | d5336d7e | cancelled at 2m50s | the push of ded064c8 started a newer run in the same `cancel-in-progress` group |
| [45388529946](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45388529946) | ded064c8 | success | both jobs: 07A 11:55:12 to 11:58:26 UTC, 07B 11:55:11 to 12:02:58 UTC |

Run 45388143553 collected no outcomes that were used. Run 45388529946 is the authoritative
reissue. No infrastructure failure remained to repair inside the reissue itself.

The cancellation came from the trigger: a `pull_request` path filter is evaluated against the
whole PR diff, so every push to PR #318 restarted the reissue and cancelled the one in flight.
The workflow now triggers on pushes to the research branch that change the tracer, runner or
workflow, and queues instead of cancelling. Budgets, source pin and runner code are unchanged.

## 2. Custody of the corrected manifests

Artifact downloads need a token, so the custody job
[olr07-reissue-custody.yml](../../.github/workflows/olr07-reissue-custody.yml) fetched the exact
bytes with the job token in run
[45393955817](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45393955817)
and committed them (commit fe15513d) under [results/olr07_reissue/](results/olr07_reissue/).
[audit/olr07_reissue_custody.py](audit/olr07_reissue_custody.py) checks, per kind:

- the artifact bytes equal the digest GitHub recorded for run 45388529946;
- every file listed in the reissue manifest is present with its SHA-256 and no unlisted file exists;
- the reissue read the authenticated original archive, and its frozen registry equals the original registry;
- contract source commit 6caaee84, candidate HB500, human HB500 (07A) and HB20000 (07B) are unchanged;
- the recorded tracer and runner hashes equal Trace.lean and olr07_reissue.py at ded064c8;
- both the standalone and the in-run scope sentinels passed;
- the manifest census recounts from receipts.json, and every scored row carries `pre_command_v1`, env `before` and `decl_in_env_before = false`.

The same script was rerun on the run machine against the committed bytes, with the local copy
of the original 07A archive. Every field agreed with the CI custody record.

| Kind | Artifact id | Bytes | Artifact SHA-256 | Manifest SHA-256 | Listed files verified |
| --- | --- | --- | --- | --- | --- |
| 07A | 14918345837 | 236264 | b3b2fcfb28ddadc9af262acf082194ad27d3c63a36a9e33df4511dcd1cd47720 | b614ffabb3b199ef0d774b8cff0616447ec6b5574d1a56bce6d188fa5259ce87 | 94 |
| 07B | 14918860201 | 1817110 | e1df19a94877646dc71d91cdafdfdb04dc81aea1cc5ce0243284a41d511b09e8 | fc7bb50e72378b5af1588c3aaddd925fcea57ae6f2091bca33551d030c1bcbe1 | 490 |

| Original | Archive SHA-256 | Manifest SHA-256 | Files | Roots |
| --- | --- | --- | --- | --- |
| 07A artifact 14890402416 | 1bd5225cfef801941bc29c5466e7c6223c69a68e53a46737d711534cfb532814 | 846958ed… | 89 | 207 |
| 07B committed AUDIT ZIP at 85d61aad | 64dedf29138a27d43130db135f711569f2738c5ee4505707eb8c144abbb88b1e | b388335c… | 664 | 1441 |

Binary hashes recorded by the reissue: Lean 92c3d35b…, REPL ae00588f…, tracer 5ec81720…,
runner 73a61a91…. The original archives are unchanged.

## 3. Reissue outcomes

### 07A: native validity fails

Evaluation human closure is 65/73 = 89.0%, below the registered 95%. Per the incident
registration only descriptive counts are reported and no conclusion is drawn from the gate.

| Role | Selected | Scored | Human closed | Exclusions | Any-menu closure, old | Any-menu closure, corrected |
| --- | --- | --- | --- | --- | --- | --- |
| TRAIN | 101 | 79 | 76 | 10 source-state mismatch, 9 unreached, 3 nonportable | 61/79 | 39/79 |
| CAL | 21 | 17 | 17 | 4 unreached | 14/17 | 9/17 |
| EVAL | 85 | 73 | 65 | 4 mismatch, 4 unreached, 2 nonportable, 2 uncertifiable | 62/73 | 44/73 |

### 07B: valid; both headroom gates fail

Human closure is at least 96.8% in every role, with 649 TRAIN, 155 CAL and 556 EVAL scored
roots, so the corrected receipts are interpretable.

| Role | Selected | Scored | Exclusions | Any-menu closure, old | Corrected | Old minus corrected, 95% file bootstrap |
| --- | --- | --- | --- | --- | --- | --- |
| TRAIN | 670 | 649 | 20 nonportable, 1 uncertifiable | 538 | 312 | 0.282 to 0.407 |
| CAL | 155 | 155 | none | 127 | 95 | 0.132 to 0.330 |
| EVAL | 616 | 556 | 53 nonportable, 7 uncertifiable | 400 | 270 | 0.144 to 0.335 |

Source-only menu policies fitted on corrected TRAIN receipts:

| Split | Ceiling | Fixed | Binary tree | Diagnostic tree | Menu gap | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| CAL | 95 | 95 | 95 | 94 | 0.0 pp | fail |
| EVAL | 270 | 265 | 265 | 268 | 0.36 pp | fail |

The frozen fixed order is simp_all, exact?, omega, simp [*]. It closes every calibration
root that any of the twelve menu actions closes, so the menu leaves no headroom for a learned
action policy under corrected pre-command receipts. The original 07B negative headroom verdict
stands. The exact original registry reproduces Fable's recreated TRAIN (538 to 312 of 649) and
CAL (127 to 95 of 155) counts. EVAL differs from Fable's 404 to 272 of 560 because 60
evaluation states are nonportable or uncertifiable under the fail-closed scope rule and are
excluded here. This result concerns the 12-action menu; it does not decide the OLR-08
longer-budget search headroom gate.

## 4. Spent cohorts and the reserved SInt family

[audit/olr08_spent_cohorts.py](audit/olr08_spent_cohorts.py) authenticated the original 07A and
07B archives and the referee archive (SHA-256 23225d1d…, matching the scope incident), read the
declared 07B roster from source, and listed `src/Init/Data` at the pinned Lean commit. Output:
[results/olr08_spent/spent_cohorts.json](results/olr08_spent/spent_cohorts.json), SHA-256
ae5df5638361b851fca9ddff536b6549a4453a65babe70944aed86973efc8cc1; regeneration is byte-identical.

| Quantity | Count |
| --- | --- |
| Spent source files (14 from 07A, 55 in the declared 07B roster) | 69 |
| Exact goal strings spent (07A, 07B, referee registry) | 1929 |
| Spent (file, declaration) keys | 1731 |
| Referee roots; all fall in 48 of the 07B files | 1728 |
| SInt files at the pin / touched by any cohort | 6 / 0 |

Families by the role in which they were spent: Nat, List, Option, Char, String as TRAIN; Int as
calibration; Array, Vector, Fin, UInt, Sum as evaluation. The referee searched calibration and
evaluation files with S1, S3 and S3c, so those files are spent for confirmation.

The roster audit for the package is one command. It reports spent-file overlap, SInt access,
exact-goal and (file, declaration) collisions, and whether each file is new within a seen
family or from a new family:

```
python audit/olr08_spent_cohorts.py audit --inventory results/olr08_spent/spent_cohorts.json --roster <roster.json> --out <run>/roster_audit.json
```

## 5. What resumes the campaign

1. Place OLR08_CPU_gym.zip on the run machine and confirm SHA-256 e5e4b26a…53750cb before
   unpacking it into `research/lean_cgrt_hrm400/olr08/`.
2. Verify package_manifest.json, then follow RUN.md: Session build and smoke on Lean 4.19.0, the
   native sentinel, and the roster audit above before any new outcome.
3. The run machine has Lean 4.19.0 under elan, the lean4 source at 6caaee84 with a clean tree,
   Python 3.12, 20 logical CPUs and 47 GiB RAM. It has no Torch install; the package states its
   protocol tests passed under Torch 2.10 CPU.

## Files added

| Path | SHA-256 |
| --- | --- |
| audit/olr07_reissue_custody.py | 22690d5a1bd481897453d1298ffead30dd0f880ccec8205b81124cac8367fd85 |
| audit/olr08_spent_cohorts.py | 9fca1cb52cb362e417071d3b9f00ca51609f2f4f1dc1ab471d22b2e1238a8791 |
| results/olr07_reissue/custody.json | 3471855795da8de746863c28a24963fdf5b9438f40418d38cc00594d1867f55f |
| results/olr08_spent/spent_cohorts.json | ae5df5638361b851fca9ddff536b6549a4453a65babe70944aed86973efc8cc1 |

Tests: tests/test_olr07_reissue_custody.py and tests/test_olr08_spent_cohorts.py, 5 passed.
