# Paid feedback across rewritten Lean guards

In the paired-proof benchmark, carrying an exact guard identity across proofs improved the second proof by 22 successes with the first choice held fixed. The recurrent reward correction did not improve on a joint planner that already knew those identities. This pilot changes the second theorem's syntax and asks whether a learned binding can recover the value of a paid observation.

## Registration

Use the existing eligible failed-prefix pairs from the BPD and MathCan corpus, with the original four-source MathCan, EvalMath, LoopedTransformers and BPD warm features. Keep the first theorem's native statement. In the second theorem, replace each guard equality with an arithmetic equivalent. Source rewrites use translation, symmetry with translation, or positive scaling. Development rewrites compose translation with positive scaling, sometimes in reversed order. Lean checks every rewritten route and the independent certificate compiler validates each successful script. Each route must retain its actual truth and tactic cost, with a different surface expression. The held compositions and deeper guards both differ from the source distribution.

Fix the first policy to the calibrated myopic continuation at five recovery calls. All arms receive the same actual result of its single paid check, if it attempts a route. The second proof has three, five or seven recovery calls. The two-slot model independently reads each pair of first and second guard texts, conditions on which first guard was actually checked, and assigns compatibility scores from the public inputs. Normalize over the six bijections. Under each possible mapping, update the calibrated guard prior with the paid source bit and use the mixture for the second theorem's exact saved-prefix planner. Keep both original theorem worlds in the same fold when fitting source priors.

The primary uses exact expected native target return as the training signal for the binding. Compare identical initial weights with supervised correspondence labels, TRM return, affine return and affine supervised binding. Match 400 batches of 16 source pairs and AdamW at .002. The lexical assignment uses a fixed softmax scale of 20 on the mean of character similarity, token similarity and numeric-set overlap. Also include the reset planner with no paid evidence and a structural reference supplied with the certified mapping. Freeze weights before evaluating the exposed development pairs. The primary endpoint is root-macro second-proof utility and the descriptive floor is +.025 over the strongest comparator without certified identities. Report binding accuracy and proof success separately.

At a paid observation, swap the two highest scored target bindings to the checked source guard. The equal-dose orthogonal sham keeps the same gap between those bindings. Leave the paid observation, physical menus and exact native outcomes fixed. Compile distinct successful scripts and replay selected two-proof sequences. This reused-cohort result has scope `FSI-01 REDESIGN_OR_PILOT_ONLY`.

## Development results

| Policy | Correct full bindings / 40 | Correct paid-guard bindings / 40 | Second proofs / 120 | Root utility on second proof |
| --- | ---: | ---: | ---: | ---: |
| Reset | — | — | 76 | 0.403958 |
| Certified mapping | 40 | 40 | 80 | 0.437292 |
| Lexical mapping | 40 | 40 | 80 | 0.437292 |
| Two-slot return | 30 | 35 | 77 | 0.410000 |
| Two-slot supervised | 40 | 40 | 80 | 0.437500 |
| TRM return | 40 | 40 | 79 | 0.427500 |
| Affine return | 40 | 40 | 77 | 0.410417 |
| Affine supervised | 40 | 40 | 77 | 0.410417 |
| Two-slot return, chosen binding swap | 23 | 25 | 73 | 0.375417 |
| Two-slot return, orthogonal sham | 29 | 33 | 78 | 0.417292 |

Carrying verified feedback through the certified mapping recovered four second proofs over reset. Lexical binding matched that proof count and the certified utility, making this corpus a weak test of learned correspondence. The return-trained two-slot model lost two proofs against TRM and three against supervised training. Both affine models ranked the correct entire mapping first at every root and still missed three proofs relative to the certified control, where the planner uses a point mass on the mapping. The binding distribution matters to action values even when its most likely assignment is correct.

The chosen swap changed the predicted paid-guard binding in 30 of 120 target attempts, losing six proofs and gaining two. The matched sham changed 12 attempted bindings, losing one proof and gaining two. Both had the same per-attempt L2 dose, averaging 2.04534 in correspondence-score space. This is evidence that the chosen binding scores contributed to downstream decisions in the development pilot.

## Post-hoc recurrence-unfolding stress test

After measuring the lexical tie, generate new versions of the same 40 development targets by unfolding one to three recurrence steps before the arithmetic wrapper. This changes the visible recursive depth and introduces explicit `geom` recurrence arithmetic or `nextMask` calls. Lean verified identical route truth and check counts. Evaluate the same frozen checkpoint and scorers. The first stress command omitted the unfolding flag, and the bank identity check stopped its evaluation. The results below come from the corrected, native-verified unfolding bank.

| Frozen policy | Correct full bindings / 40 | Correct paid-guard bindings / 40 | Second proofs / 120 | Root utility |
| --- | ---: | ---: | ---: | ---: |
| Reset | — | — | 76 | 0.403958 |
| Certified mapping | 40 | 40 | 80 | 0.437292 |
| Lexical mapping | 40 | 40 | 79 | 0.427292 |
| Two-slot return | 26 | 28 | 75 | 0.395417 |
| Two-slot supervised | 38 | 38 | 76 | 0.402708 |
| TRM return | 36 | 37 | 76 | 0.402708 |
| Affine return | 40 | 40 | 78 | 0.417917 |
| Affine supervised | 40 | 40 | 78 | 0.417917 |
| Two-slot return, chosen binding swap | 21 | 22 | 70 | 0.350417 |
| Two-slot return, orthogonal sham | 28 | 29 | 75 | 0.393958 |

Lexical binding still found all 40 correspondences after recurrence unfolding. The two-slot reward model lost four correct full bindings relative to its arithmetic-rewrite score and closed one fewer target proof than either TRM or the reset control. The two affine models retained all 40 bindings and closed 78 proofs. The chosen-binding intervention changed 51 paid-guard assignments, gaining two proofs and losing seven. The orthogonal sham changed 18 assignments, gaining one and losing one. Both used a mean L2 dose of 1.61351 score units. The stress test shows a larger syntax shift for the recurrent model, with lexical correspondence still sufficient for the best uncertified score.

The frozen checkpoint SHA-256 is `f69a8f6947ce6bfbe8cd3f3a4d3f1c0f03421636d613a28cb8f57f49e3cd815b`. Lean independently compiled 140 arithmetic bank proofs and 63 unfolded bank proofs. Each scoring phase independently compiled 79 successful scripts and replayed ten two-proof sequences. Runs used logical CPU 2 with a 2 GiB process-tree cap. Evaluation peaked below 0.54 GiB committed. All 144 native-enabled gym tests passed.

[Arithmetic bank](results/ppt_hybrid_rl/ppt-rewritten-recovery-bank-1132.json.gz) · [Arithmetic results](results/ppt_hybrid_rl/ppt-rewritten-recovery-dev-1132.json.gz) · [Unfolded bank](results/ppt_hybrid_rl/ppt-unfolded-stress-bank-1132.json.gz) · [Unfolded results](results/ppt_hybrid_rl/ppt-unfolded-stress-dev-1132.json.gz)

```text
python -m configs.freeze_ppt_rewritten_recovery BANK BPD MATHCAN REWRITTEN
python -m configs.freeze_ppt_rewritten_recovery BANK BPD MATHCAN UNFOLDED --unfold
python -m configs.run_ppt_rewritten_recovery fit BANK BPD MATHCAN WARM PRIOR ANCHOR REWRITTEN FIT RESULTS
python -m configs.run_ppt_rewritten_recovery eval BANK BPD MATHCAN WARM PRIOR ANCHOR UNFOLDED FIT RESULTS
```
