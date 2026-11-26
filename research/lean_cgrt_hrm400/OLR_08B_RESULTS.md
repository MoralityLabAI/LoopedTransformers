# OLR-08B results: allocation diagnostic and prospective headroom gate

2026-11-27. Branch `research/olr08b-frontier-mms-20261126`, PR #399. Protocol:
[olr08b/RUN.md](olr08b/RUN.md). Amendment registered before freezing:
[OLR_08B_ROSTER_AMENDMENT.md](OLR_08B_ROSTER_AMENDMENT.md).

**Verdict: the prospective headroom gate failed.** Feasibility passed: 248 menu-hard roots in 35
file clusters, with 95.1% T-human closure. The strongest 300-step control, hand-prior S2, closes
89/248 (35.9%), so the ceiling check passes. The registered 1200-step S3 reference closes 98/248,
a margin of 3.6 percentage points against the required 10. No learning campaign is authorized,
so OLR-08C was not prepared. No evaluation root was opened, SInt was not opened, nothing was
trained, and OLR-08A's results and evaluation registry are unchanged.

## Execution

- **Code and environment.** Code at d63e242a and later. Isolated worktree `C:\Users\patri\olr08b-399-wt`, separate venv
  (Python 3.12.10, Torch 2.5.1 CPU), outputs under `C:\olr08b-399`. No other process was
  terminated.
- **Native binary.** The repaired OLR-08A Session server binary (SHA-256 61923c65…), built from Lean sources
  identical to this branch's olr08/lean, with the pinned Lean source at 6caaee84.
- **Checks before outcomes.** Regression tests passed 10/10, and 11/11 after the freeze fix; CI passed on every
  push. The 08A gym tests passed 4/4. The native scope sentinel passed: the earlier lemma
  certified, while the self-reference, `simp only` self and future-theorem proofs were refused.
- **Resources.** 8 threads, 2 workers, Windows Job Object hard commit limit 12 GiB. Each phase completed in
  one attempt. A resume wrapper was in place but was never needed.

| Phase | Wall seconds | Peak job commit | Peak sampled RSS |
| --- | --- | --- | --- |
| A, spent diagnostic, 40 specs | 4,355 | 2.44 GB | 1.67 GB |
| B, freeze | trace only | | |
| B, fresh calibration, menu and 15 specs | 23,337 | 2.44 GB | 2.40 GB |

Wall times shared the machine with other jobs. Raw heartbeats and RPC counts are the stable cost
measures.

## Infrastructure repair before freezing

The freeze reused the 08A selector, which names a file's trace directory family_stem. Five pairs of
nested roster files share that key, for example Int/Bitwise/Basic and Int/DivMod/Basic. With two
workers their traces would overwrite each other, and the registry would attribute one file's
declarations, positions and goals to the other's path. Commit d63e242a gives each file a
directory that keeps every subdirectory, refuses colliding rosters, and adds a regression test.
Selection rules are unchanged. Phase A does not use the freeze path and ran under its own
contract.

## Phase A: spent diagnostic (exploratory)

The run covered the 39 OLR-08A calibration roots, including the disclosed smoke overlap. Menu
validity was 34/34 T-human; 18 roots were menu-hard, in 5 files. The diagnostic used the immutable
08A TRAIN-fitted prior. It cannot establish efficacy.

Solved, of 18, by step budget:

| Policy | 75 | 150 | 300 | 600 | 1200 |
| --- | --- | --- | --- | --- | --- |
| S1 uniform | 2 | 3 | 3 | 3 | 3 |
| S2 hand prior | 3 | 4 | 4 | 4 | 5 |
| S3 counted prior | 3 | 3 | 3 | 4 | 5 |
| S3c cost-aware | 3 | 3 | 3 | 3 | 3 |
| RR four-queue | 3 | 3 | 3 | 4 | 4 |
| UCB heuristic | 3 | 3 | 3 | 3 | 4 |
| Offline union of S1–S3c | 3 | 4 | 4 | 5 | 6 |

- The four single-arm searches at 300 steps reproduce OLR-08A exactly: same outcome, steps and
  path on all 18 roots.
- RR and UCB never beat the best single arm at the same budget. They lose 1 root to S2 at most
  budgets.
- Depth 5 and the wider premise caps (64/16 requested, 40/10 kept) each solve 3 at 300 steps, the
  same roots as depth-3 S3.
- Menu-cache hits occur only at the root, where four queues share one snapshot: 54 hits. The
  payload peaks near 1 MB, so the 64 to 4,096 MiB tiers give identical outcomes.
- Premise retrieval is 23 to 33% of charged raw heartbeats across the 300-step specs.
- 130 credited closures; all were replayed and certified in a fresh pre-command process.

## Phase B: prospective calibration

**Registry.** The amended 57-file roster froze to 455 roots (369 T, 86 R) in 45 files. Registry
SHA-256 778b8d406cde07c9d43bee8a26620a718b040a9397f1f2443d8100dc8f2f75fe was committed in
6112c422 before any outcome. The audit found no file, exact goal, declaration name or root id in
common with:

- the 69-file spent inventory;
- the full original 08A registry (TRAIN, CAL and EVAL);
- the parallel 08B cohort;
- the 08A smoke registry.

It also found no SInt file and no duplicates.

**Feasibility.**

| Criterion | Threshold | Measured | Result |
| --- | --- | --- | --- |
| T-human audits | ≥10 | 350 | pass |
| Human closure | ≥95% | 333/350 = 95.1% | pass |
| Menu-hard roots | ≥120 | 248 (167 T, 81 R) | pass |
| File clusters with hard roots | ≥12 | 35 | pass |

Scored 434, excluded 21 (20 nonportable, 1 uncertifiable). Hard roots by family: List 150,
Nat 75, String 19, Array 2, Int 1, UInt 1.

**Headroom gate.**

| Check | Threshold | Measured | Result |
| --- | --- | --- | --- |
| Strongest 300-step control | ≤80% | S2 89/248 = 35.9% | pass |
| S3 at 1200 minus strongest control | ≥10 pp | 98 − 89 = 3.6 pp | **fail** |
| Offline union of single arms at 300 (never deployable) | reported | 96, +2.8 pp over S2 | — |

**Controls on the 248 hard roots.** Every unsolved root reached the step cap. None hit an RPC or
heartbeat cap, and none exhausted its action space. All 1,046 credited closures were replayed and
certified fresh.

| Spec | Solved | Steps | RPCs | Premise RPCs | Raw heartbeats | Premise share |
| --- | --- | --- | --- | --- | --- | --- |
| S1 300 | 56 | 61,103 | 89,321 | 28,162 | 5,762,882,661 | 31.1% |
| S2 300 | 89 | 53,473 | 71,018 | 17,456 | 4,968,376,901 | 26.9% |
| S3 300 | 72 | 57,665 | 83,589 | 25,852 | 6,430,032,984 | 24.7% |
| S3c 300 | 66 | 59,647 | 87,141 | 27,428 | 5,233,848,699 | 31.4% |
| RR 300, no cache | 71 | 59,877 | 85,813 | 25,865 | 5,489,492,200 | 31.0% |
| UCB 300, no cache | 62 | 59,601 | 86,397 | 26,734 | 5,567,966,868 | 31.4% |
| S3 1200 (reference) | 98 | 200,814 | 268,718 | 67,806 | 21,340,037,693 | 21.2% |

**Paired against S2.** Exploratory, because S2 was chosen as strongest on the same cohort. The
interval is a 95% bootstrap over 35 file clusters.

| Spec | Difference in roots | Only this spec | Only S2 | 95% interval |
| --- | --- | --- | --- | --- |
| S3 1200 | +9 | 19 | 10 | −0.8 to +8.7 pp |
| S3 300 | −17 | 6 | 23 | −11.5 to −2.3 pp |
| RR 300 | −18 | 0 | 18 | −10.6 to −4.3 pp |
| S3c 300 | −23 | 3 | 26 | −13.4 to −5.0 pp |
| UCB 300 | −27 | 0 | 27 | −15.4 to −6.9 pp |
| S1 300 | −33 | 2 | 35 | −18.5 to −8.3 pp |

The shared-budget portfolios solve no root that S2 misses. Splitting 300 attempts across four
priors costs closures here.

**Memory tiers.** Results are identical across the 64, 256, 1,024 and 4,096 MiB tiers.

- Cache hits total 744, which is 3 per root: the shared root menu.
- Premise RPCs fall by 744 and premise heartbeats by 2.6%.
- Payload peaks near 0.95 MB, with no evictions.
- RR outcomes and schedules equal the no-cache run.
- UCB solves the same 62 roots. On one root its schedule diverges at the fifth pull, because UCB's
  cost term counts charged premise heartbeats and a cache hit removes them.

The tiers are the transparent memory control, not a BPD feature map.

## Disclosures

- **Parallel OLR-08B run.** A separate implementation, at commit 9612b6fa on
  research/olr-obligation-ledger-20261123, runs on this machine in C:\olr08b. Its registry
  overlapped 7 roster files where it had already collected outcomes. Those files were excluded
  before freezing, as registered in the amendment. Its registry also lists all 275 original OLR-08A
  evaluation roots with identical IDs. If that run proceeds to evaluation, the 08A evaluation
  registry will be spent.
- **Narrow human-closure margin.** Human closure cleared 95% by 0.14 points (333/350).
- **Cohort composition.** The fresh cohort is dominated by List and Nat files. Family overlap with
  earlier TRAIN cohorts is expected, and files are clusters, not independent samples.
- **Same-cohort selection.** The strongest control was selected on the same cohort, so the paired
  intervals are exploratory.

## Artifacts

- **Phase A.** [results/olr08b_pr399/spent/](results/olr08b_pr399/spent/): summary, contract,
  resource, analysis, attempt ledger, and `olr08b-spent_AUDIT.zip`. The archive holds the full run
  directory with every per-root checkpoint, seal and replay record, plus a manifest of every file's
  SHA-256.
- **Phase B.** [results/olr08b_pr399/cal/](results/olr08b_pr399/cal/), with the same layout and
  `olr08b-cal_AUDIT.zip`.
- **Pre-outcome records.** [results/olr08b_pr399/preoutcome/](results/olr08b_pr399/preoutcome/):
  registry, seal, provenance, census, audit and parallel-cohort evidence.
- **Tools.** [olr08b/tools/](olr08b/tools/): analysis, registry audit, archive and the resume
  wrapper.
- **Pinned URLs and archive hashes.**
  [results/olr08b_pr399/PINNED.md](results/olr08b_pr399/PINNED.md).
