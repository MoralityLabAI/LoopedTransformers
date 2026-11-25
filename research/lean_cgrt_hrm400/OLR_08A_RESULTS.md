# OLR-08A results: native term/action search, headroom gate failed

2026-11-25. Branch `research/olr-obligation-ledger-20261123`, PR #318 (draft). Protocol:
[olr08/RUN.md](olr08/RUN.md) and [olr08/run08.py](olr08/run08.py) with the two infrastructure
repairs in [olr08/REPAIRS.md](olr08/REPAIRS.md). Pre-outcome registration:
[OLR_08A_PREOUTCOME.md](OLR_08A_PREOUTCOME.md).

**Verdict: Gate 1 (headroom) failed on two of its registered criteria.** Calibration has 18
menu-hard roots against the required 24. The 1200-step S3 reference closes 5 of 18 against 4 of 18
for the best 300-step control, which is 5.6 percentage points against the required 10. Gate 0
(source validity) passed. No reader was fitted, Gate 2 was not reached, and no fresh-evaluation
outcome was collected. The 275 fresh evaluation roots remain unspent. SInt was not opened.
Thresholds were not changed.

## Gates

| Gate | Criterion | Threshold | Measured | Result |
| --- | --- | --- | --- | --- |
| 0 | TRAIN T-human replay | ≥95%, ≥20 audits | 159/163 = 97.5% | pass |
| 0 | CAL T-human replay | ≥95%, ≥10 audits | 34/34 = 100% | pass |
| 1 | Menu-hard TRAIN roots | ≥40 | 106 | pass |
| 1 | Menu-hard CAL roots | ≥24 | 18 | **fail** |
| 1 | Strongest 300-step control on CAL | ≤80% | S2 4/18 = 22.2% | pass |
| 1 | S3 at 1200 steps minus best 300-step control | ≥10 pp | 5/18 − 4/18 = 5.6 pp | **fail** |
| 2 | Replay-certified source action frames | ≥64 | not reached | — |

Without the one calibration root the smoke run had observed (Int.cooper_resolution_dvd_right,
unsolved by every arm): 17 hard CAL roots, S2 4/17, S3-long 5/17, headroom 5.9 pp. Both failures
stand.

## Cohort

| Role | Selected | Scored | Exclusions | Menu closes | Menu-hard |
| --- | --- | --- | --- | --- | --- |
| TRAIN (reused referee source) | 192 | 187 | 4 nonportable, 1 uncertifiable | 81 | 106 |
| Calibration (fresh files) | 39 | 36 | 3 nonportable | 18 | 18 |
| Evaluation (fresh files) | 275 | not collected | | | |

Nonportable states were refused, never opened in the later environment. Calibration files:
Int/Cooper, Int/OfNat, Array/BinSearch, Array/FinRange, Fin/Iterate. Menu actions closing the most
calibration roots: exact? 15, simp [*] 10, simp_all 10, omega 9.

## Search controls

Exact best-first search over enumerated actions, depth 3, per root 300 attempted tactics,
1200 native RPCs and 512 million raw heartbeats (fourfold for the long reference). Every
unsolved root below hit the step cap. None reached the RPC or heartbeat cap, and none exhausted
its finite action space. These are censored observations, not proofs of absence.

| Arm | Roots | Solved | Censored at step cap | Steps to solve | Credited paths replayed and certified fresh |
| --- | --- | --- | --- | --- | --- |
| Source S2 (hand prior, TRAIN) | 106 | 33 | 73 | 16 to 289 | 33/33 |
| CAL S1 uniform | 18 | 3 | 15 | 17, 23, 85 | 3/3 |
| CAL S2 hand prior | 18 | 4 | 14 | 10, 18, 32, 83 | 4/4 |
| CAL S3 counted prior | 18 | 3 | 15 | 2, 2, 23 | 3/3 |
| CAL S3c cost-aware prior | 18 | 3 | 15 | 4, 4, 23 | 3/3 |
| CAL S3 long, 1200 steps | 18 | 5 | 13 | 2, 2, 23, 397, 1069 | 5/5 |

Solved calibration roots, with the steps each arm needed:

| Root | File | S1 | S2 | S3 | S3c | S3 long |
| --- | --- | --- | --- | --- | --- | --- |
| Array.size_finRange | Array/FinRange | 17 | 18 | 2 | 4 | 2 |
| Array.getElem_finRange | Array/FinRange | 23 | 32 | 2 | 4 | 2 |
| Array.finRange_reverse | Array/FinRange | 85 | 10 | 23 | 23 | 23 |
| Array.binSearchContains | Array/BinSearch | — | 83 | — | — | 1069 |
| Int.OfNat.of_dvd | Int/OfNat | — | — | — | — | 397 |

The S3 long reference solves every root S2 solves plus Int.OfNat.of_dvd. Paired on the 18
calibration roots over 5 file clusters, S2 minus S3 is +1 root (95% file bootstrap 0 to +33 pp),
and S3 minus S3c is 0. The counted prior reached its two FinRange closures in 2 steps where S2
needed 18 and 32, but lost Array.binSearchContains within 300 steps. Int/Cooper's 7 hard roots
were unsolved by every arm.

The S3 prior was fitted on the 106 TRAIN S2 searches from exact attempt-event IDs (48 template
keys; prior SHA-256 4fb599f5…). One calibration root, Array.finRange_succ, has a state whose
premise enumeration exceeds the server's cap. That timeout was charged in every arm (S1 3, S2 1,
S3 3, S3c 3, S3 long 20 calls) and the state kept its non-premise actions (repair 0002).

## Resources

All native work is charged: premise enumeration, tactic execution and certification. Graph and
lemma-graph construction run only in reader arms, which were not reached. Raw heartbeats are
Lean heartbeats as counted by the server, not thousands.

| Stage | Step RPCs | Premise RPCs | Cert RPCs | Raw heartbeats | Server ms | Native wall s | Root opening s | Fresh replay: opening s, native s, raw hb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Source S2 | 25,122 | 8,660 | 33 | 2,241,168,728 | 289,240 | 312.8 | 296.6 | 91.7, 1.13, 8,729,839 |
| CAL S1 | 4,625 | 2,238 | 3 | 445,558,669 | 60,709 | 65.4 | 14.3 | 1.9, 0.22, 2,324,645 |
| CAL S2 | 4,343 | 1,448 | 4 | 415,792,664 | 48,515 | 50.9 | 11.1 | 3.7, 0.25, 3,011,986 |
| CAL S3 | 4,527 | 2,076 | 3 | 541,384,580 | 68,643 | 72.5 | 12.2 | 2.5, 0.24, 1,713,739 |
| CAL S3c | 4,531 | 2,090 | 3 | 435,915,330 | 54,631 | 57.7 | 10.8 | 1.8, 0.15, 1,713,739 |
| CAL S3 long | 17,093 | 6,779 | 5 | 2,039,593,986 | 245,314 | 260.5 | 11.6 | 4.2, 0.22, 2,324,444 |

The menu stage (231 roots, two fresh passes each, plus T-human replay at the 20000 cap) used
251,684,346 raw heartbeats, with 556 s and 560 s of root opening in the two passes. Per-root,
per-RPC journals with heartbeats, server ms and wall seconds are in the audit archive.

Run 3 took 1,933 s for the campaign after a separate registry preflight. It used 8 threads,
2 native workers and a Windows Job Object hard commit limit of 12 GiB. Peak job commit was
2.83 GB, and peak sampled process-tree RSS 2.89 GB. Wall times share the machine with other
jobs and are not a speed comparison. No end-to-end speedup is claimed.

## Run history and reproducibility

| Run | Code | Outcome |
| --- | --- | --- |
| Smoke (C:\olr08\smoke) | package | completed in 1,051 s; ineligible for gates |
| 1 (C:\olr08\full) | package run08.py | stopped in source S2: credited exact? path failed fresh replay (repair 0001) |
| 2 (C:\olr08\full2) | + repair 0001 | killed twice by an external process kill; then stopped in CAL S1 on a premise timeout (repair 0002) |
| 3 (C:\olr08\full3) | + repairs 0001, 0002 | completed through the calibration stages in one attempt; stopped before fresh evaluation as registered |

Each rerun refroze the registry byte-identical to the audited one (SHA-256 d61bf979…) before any
outcome. Checkpoints that overlap between runs agree field by field:

| Comparison | Roots | Identical |
| --- | --- | --- |
| Menu receipts, runs 1 and 2 against run 3 | 231 each | 231 and 231 |
| Source S2, run 1 against run 3 | 73 | outcome, steps, path, RPC counts and full attempt sequence 73/73 |
| Source S2, run 2 against run 3 | 106 | same fields 106/106 |
| CAL S1, run 2 against run 3 | 16 | same fields 16/16 |

Total raw heartbeats differ by at most 142 per root, against about 21 million per root on average.

## Disclosures

- The smoke run, prescribed by RUN.md, observed outcomes on calibration root
  Int.cooper_resolution_dvd_right and evaluation root BitVec.shiftLeftZeroExtend, which are also
  in the full registry. No setting changed afterwards; the gate verdict holds without the
  calibration root. The evaluation root's smoke outcome is the only fresh-evaluation outcome
  observed.
- Both repairs change run08.py and therefore the contract. They restore behaviour the protocol
  already assumes, a reproducible fresh replay and Fable's handling of a failed premise call, and
  neither alters caps, rosters, priors or gates.
- A failed headroom gate stops the run before fresh evaluation, as pre-registered in
  OLR_08A_PREOUTCOME.md. Resuming launch.py on run 3's directory would collect fresh evaluation
  outcomes for the fixed controls under the same contract.
- Fresh calibration files are new files within the Int, Array and Fin families; this cohort has
  no new-family calibration evidence.

## Artifacts

- Audit archive: [results/olr08a/olr08-full_AUDIT.zip](results/olr08a/olr08-full_AUDIT.zip).
  It holds the registry, all native menu and replay receipts, per-root search journals with RPC
  journals, the prior, resource telemetry, code, drivers, analysis and file hashes.
- Referee archive: [results/olr08a/olr08-full_FABLE.zip](results/olr08a/olr08-full_FABLE.zip),
  with code, contract, summary, frozen policy, prior, resources and the analysis report.
- Unpacked run records: [results/olr08a/full/](results/olr08a/full/).
- Gym source: [olr08/](olr08/), with the authenticated ZIP, repairs and drivers.
- Archive hashes and commit-pinned URLs: [results/olr08a/PINNED.md](results/olr08a/PINNED.md).

No model checkpoint exists, because no reader was eligible for fitting.

# Appendix: prerequisites completed on 2026-11-24

The sections below were written before the package was supplied and are kept as the record of
the 07A/07B reissue custody and the spent-cohort inventory.

## A1. 07A/07B exact reissue: run history

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

## A2. Custody of the corrected manifests

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

## A3. Reissue outcomes

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
excluded here. This result concerns the 12-action menu. The OLR-08A search headroom gate is reported above.

## A4. Spent cohorts and the reserved SInt family

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

## A5. Files added on 2026-11-24

| Path | SHA-256 |
| --- | --- |
| audit/olr07_reissue_custody.py | 22690d5a1bd481897453d1298ffead30dd0f880ccec8205b81124cac8367fd85 |
| audit/olr08_spent_cohorts.py | 9fca1cb52cb362e417071d3b9f00ca51609f2f4f1dc1ab471d22b2e1238a8791 |
| results/olr07_reissue/custody.json | 3471855795da8de746863c28a24963fdf5b9438f40418d38cc00594d1867f55f |
| results/olr08_spent/spent_cohorts.json | ae5df5638361b851fca9ddff536b6549a4453a65babe70944aed86973efc8cc1 |

Tests: tests/test_olr07_reissue_custody.py and tests/test_olr08_spent_cohorts.py, 5 passed.
