# Proposal: Revising Gate G0 (RMP v0)

Status: proposal, 2026-10-02. Nothing below runs until it is approved. The
program document's gate stays in force until then.

## The problem

G0 asks both arms, at three seeds each, to reach exact match of 0.98 or more
on depths 1-8 of four families with 8 block applications. The first gate cell
(`LOOP-T-core4-s1`, step 32,750) is at or near the gate on three families and
stuck on the fourth:

| Family | Interim accuracy | Weakest depths |
| --- | ---: | --- |
| `sudoku4` | 1.000 | none |
| `pointer_chase` | 0.986 | depth 8 at 0.89 |
| `gated_pointer_chase` | 0.982 | depth 8 at 0.88 |
| `reachability` | 0.836 | depths 6, 7, 8 at 0.75, 0.64, 0.48 (chance is 0.5) |

Reachability has been flat for about 13,000 steps.

Reading the answer after each visit (`diagnostics/visit_cost.py`) shows that a
depth-`d` answer first becomes correct at visit `d + 2` for the pointer
families and `d + 3` for reachability. At those rates 8 visits cover pointer
chasing to depth 6 and reachability to depth 5. The pointer families reach
depths 7 and 8 only through a computation that is right at exactly the 8th
visit. If that rate holds, G0 as written cannot pass on reachability, however
long training runs.

The rate is measured, not proven: a model could learn a faster scheme. The
warm-start pair now queued (`LOOP-T-warm-reach-v16-s1` against
`LOOP-T-warm-reach-v8-s1`, both from the step-32,840 snapshot) tests directly
whether more visits let reachability depths 6-8 be learned.

## Options

| | A. More visits (recommended) | B. Fewer gated depths | C. Change the encoding |
| --- | --- | --- | --- |
| Change | 12 block applications for both arms (`LOOP-T` = `[0]*12`, `FF-U` = `[0..11]`) | Keep 8 visits; gate depths 1-5 for every family | Put start, goal and horizon before the edge list in `reachability` |
| Depths gated | 1-8, unchanged | 1-5 (`sudoku4` 1-3) | 1-8, unchanged |
| Unique parameters | `LOOP-T` 459,776 (unchanged); `FF-U` 2,628,096 (was 1,839,616) | unchanged | unchanged |
| Compute per step | 1.5x | unchanged | unchanged |
| Retraining | all six cells from scratch | the current cell continues; at its present state it would still miss on reachability depths 2-5 (0.92 to 0.97) | `reachability` regenerated; bundle hash changes; all six cells from scratch |
| Spare visits at depth 8 | 2 for pointer families, 1 for reachability | 1 for pointer families at depth 5, none for reachability | unknown |
| Main risk | Wasted if visits are not what binds | Reachability depth 5 needs exactly 8 visits, so 0.98 may still be out of reach | Changes the task, not the budget; the pointer families' depth-8 problem remains |

A also suits the program's later phases. The convergence-signal and
loop-overshoot analyses (program document, concept hunts) need visits left over
after the answer is formed. A model that uses every one of its visits on the
deepest questions has no settled state to study.

B fits the gate to the results already seen, which makes it post hoc. If B is
chosen, the depth range should come from a stated rule (depths whose step
count plus the measured start-up fits the budget) rather than from which
depths currently pass.

## Recommendation and decision rule

Adopt A, conditional on the warm-start test:

- If the 16-visit arm clears reachability depths 6-8 and the 8-visit control
  does not, the budget is what binds: adopt A.
- If both arms clear them, the budget does not bind and the plateau is a
  training problem: keep G0 as written and continue down the fallback ladder
  (deep supervision, input re-injection).
- If neither clears them, visits are not enough on their own: consider C, and
  report G0 as failed for reachability at 8 visits in the meantime.

## What changes if A is adopted

- Program document: the arm table, rung `M0` (visits 8 to 12; `FF-U` parameter
  count), and the Decisions section, with this file as the record.
- Gate cells: the six `core4` cells rerun with `--visits 12`; `variant`
  becomes `core4-v12` so they do not mix with the 8-visit cells.
- The current 8-visit queue stops after its running cell. `LOOP-T-core4-s1`
  is kept as the 8-visit evidence for the report.
- The task bundle, the validation split, the held-out depths 9-12, and the 0.98
  threshold are unchanged.

## Cost

On an uncontended GPU the first 8-visit cell ran about 7 optimizer steps per
second. At 12 visits that is roughly 4.5. If a cell needs 40,000 to 60,000
steps to reach its gate checkpoint, six cells take about 15 to 22 GPU-hours.
Today's contention and power limits made runs 2 to 20 times slower than that.
