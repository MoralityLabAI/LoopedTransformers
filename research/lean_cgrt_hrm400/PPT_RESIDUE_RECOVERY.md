# Proof feedback across different residue partitions

Earlier tests changed the wording of equivalent guards, and a lexical matcher still recovered their identity. This pilot changes the predicates themselves. The first Lean theorem reveals whether a checked residue belongs to one group. The second theorem uses a different partition of the residues, so its recovery decision must use the remaining possibilities.

## Registered pilot

Generate 48 source and 40 disjoint development pairs from the BPD recurrence and MathCan mask definitions. In every pair, both theorems use a common arithmetic expression modulo nine. Each proof has three disjoint exhaustive guard options. The first groups are size three. The source second groups are size three, and each development second menu has sizes two, three and four. Each first group intersects at least two second groups. Lean certifies all route outcomes and, independently, the actual common residue. Both proofs charge the native failed-prefix setup and their observed tactic attempts.

All policies share one fixed first attempt under a five-call recovery budget. It picks the route of highest immediate expected value under uniform residue mass. Only its native pass or failure becomes evidence. Score the second proof at three, five and seven recovery calls. Each residue gets one input slot describing its first route and a second slot describing its second route, with the four-source warm features from MathCan, EvalMath, LoopedTransformers and BPD. Mask residues inconsistent with the checked bit and project the remaining probability into the second menu's three disjoint guard classes. The exact saved-prefix planner uses those probabilities to compare continuing after a failed route and stopping.

Fit the two-slot model through exact expected native second-proof return. Compare TRM and affine models trained on the same reward, and matched two-slot and affine models trained on the certified source residue labels. All arms receive 400 batches of 16 source pairs, the same features and costs, and AdamW at .002. Greedy evaluation keeps the experienced first check fixed. Uniform reset and uniform conditioning measure the value of the observed bit with a fixed prior. Give the oracle the actual residue only as a structural reference. The primary endpoint is equal-root second-proof utility with a +.025 descriptive gain floor over the strongest non-oracle control.

For BPD, when the learned policy and uniform-conditioned planner disagree between two available routes, exchange the scores of the most favored compatible residue in each route. A matched orthogonal sham preserves their score gap at the same L2 dose. Apply both inside the same support mask after the same paid first proof. Independently compile completed scripts and replay selected paired traces. This is a development pilot under `FSI-01 REDESIGN_OR_PILOT_ONLY`.

## Development results

| Policy | Three target calls / 40 | Five / 40 | Seven / 40 | Second proofs / 120 | Root utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform reset | 16 | 20 | 32 | 68 | 0.328125 |
| Uniform conditioned on first check | 13 | 22 | 33 | 68 | 0.327917 |
| Two-slot return | 10 | 19 | 32 | 61 | 0.278125 |
| Two-slot supervised | 12 | 20 | 35 | 67 | 0.333750 |
| TRM return | 8 | 18 | 29 | 55 | 0.233542 |
| Affine return | 15 | 22 | 33 | 70 | 0.344375 |
| Affine supervised | 15 | 20 | 34 | 69 | 0.336042 |
| Two-slot return, chosen residue swap | 7 | 18 | 30 | 55 | 0.230000 |
| Two-slot return, orthogonal sham | 10 | 16 | 27 | 53 | 0.213958 |
| Actual-residue reference | 18 | 33 | 40 | 91 | 0.541250 |

The two-slot reward model gained six proofs and .044583 utility over matched TRM. Across paired roots, it won on 14, tied on 15 and lost on 11 relative to TRM. It trailed the two-slot supervised model by six proofs and the strongest affine policy by nine. Uniform conditioning exchanged nine previously failed proofs for nine previously successful ones at the same first route, so the paid evidence changed outcomes without a net count gain.

The residue interventions shared 38 eligible attempts and a mean L2 dose of 5.4684 logit units. The chosen swap lost 14 proofs and gained eight. The orthogonal sham lost ten and gained two. Both had larger downstream effects than the small gain relative to TRM. The sham was more damaging, so this intervention provides no evidence that the learned score placement itself was beneficial.

## Post-hoc acquisition diagnostic

A frozen uniform planner was allowed to choose the first route for expected utility across both proofs. It changed 38 first choices, increasing second proofs from 68 to 69 while decreasing first proofs from 36 to 34. Combined utility fell from .412292 to .401875. This shows a different allocation of effort without an observed joint-return gain on these roots. The diagnostic reproduced every registered output when it was added.

The fit SHA-256 is `3be887406a7efe6791f3369548b919b874f50e74e4c115ea871bfffd70a87b93`. Lean independently compiled 176 bank proofs and 59 distinct successful evaluation scripts. The final evaluation replayed 12 selected paired sequences and peaked at 0.49 GiB committed. Execution used logical CPU 2 within a 2 GiB process-tree cap. All 147 native-enabled gym tests passed. [Native bank](results/ppt_hybrid_rl/ppt-residue-recovery-bank-1133.json.gz) · [Numerical receipt](results/ppt_hybrid_rl/ppt-residue-recovery-audited-1133.json.gz)

```text
python -m configs.freeze_ppt_residue_recovery BPD MATHCAN BANK
python -m configs.run_ppt_residue_recovery fit BANK BPD MATHCAN WARM PRIOR FIT RESULTS
python -m configs.run_ppt_residue_recovery eval BANK BPD MATHCAN WARM PRIOR FIT RESULTS --acquisition-diagnostic
```
