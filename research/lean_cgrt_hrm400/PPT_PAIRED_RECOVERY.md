# Credit across two Lean recovery proofs

Previous failed-prefix work kept proof attempts independent. Calibrated public planning has been the strongest recurrent comparator. This pilot tests whether a first attempt earns value through a paid guard observation that affects a second theorem.

## Registered pilot

Use the existing BPD and MathCan failed-prefix bank, with the original four-source MathCan, EvalMath, LoopedTransformers and BPD feature prior. Use only theorem pairs with three distinct public guard expressions. The two members rotate those exact expressions across different native witnesses. Check that Lean's observed truth agrees after matching by expression. Give both orientations of every eligible pair the fixed first/second recovery budgets (3, 5), (5, 3), (5, 5) and (7, 5). Charge the genuine six-call failed prefix separately on both proofs.

Both public theorem menus and budgets are available before choosing the first action. The first proof can stop or attempt one affordable recovery fragment. It supplies exactly that checked guard outcome to the second proof, regardless of success. The second proof can use its full recovery menu and feedback, with the same calibrated planner in all learned arms. Combine the two criterion priors by averaging the frozen supervised joint probabilities after alignment by public expression. For source training, get both forecasts from the same out-of-fold affine guard classifier, leaving the whole pair out. The second policy conditions on the paid bit only. It never reads the outcome of an untried first-proof fragment.

The reset control scores each theorem separately. The myopic memory control uses the same paid evidence on the second proof but chooses the first action for immediate reward only. The joint planner values that observation across the two proof budgets and serves as the strongest structural comparator. A two-slot actor relates the same guard's continuation in the first and second proof, adding a bounded learned correction to the four candidate first choices. Compare matched TRM and affine reward policies, and the same two-slot architecture trained by Q regression on observed two-proof returns. Use 480 shared batches of 32 source sequences, AdamW at .002 and stochastic temperature .12. Use greedy evaluation. The primary endpoint is equal-root utility summed over both proofs, at one unit per proof less .025 per charged tactic. The descriptive floor is +.025 over the strongest non-oracle policy. A separate actual-return oracle selects within the score envelope without changing how the second proof uses observations.

Where the learned first choice differs from the joint planner and at least three options are legal, swap the two chosen score additions. The orthogonal control uses the same L2 dose and preserves their score difference. Both act after the identical forced native rejection. Independently compile distinct successful scripts on both proofs and replay selected complete two-proof traces. The constituent theorems were exposed in earlier development, and the final scope remains `FSI-01 REDESIGN_OR_PILOT_ONLY`.

## Development results

The identity filter kept 46 source pairs and all 40 development pairs. The fit used 368 source sequences. Development used 320 two-proof sequences.

| Policy | First proofs / 320 | Second proofs / 320 | Total proofs / 640 | Charged checks | Root utility per sequence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reset between proofs | 175 | 193 | 368 | 5,891 | 0.689766 |
| Myopic first choice with memory | 175 | 215 | 390 | 5,888 | 0.758750 |
| Calibrated joint planner | 181 | 227 | 408 | 5,907 | 0.813516 |
| Two-slot return | 167 | 225 | 392 | 5,878 | 0.765781 |
| Two-slot Q regression | 169 | 222 | 391 | 5,870 | 0.763281 |
| TRM return | 181 | 227 | 408 | 5,907 | 0.813516 |
| Affine return | 175 | 224 | 399 | 5,889 | 0.786797 |
| Two-slot return, chosen-score swap | 181 | 227 | 408 | 5,907 | 0.813516 |
| Two-slot return, orthogonal sham | 166 | 221 | 387 | 5,855 | 0.751953 |
| Actual-return oracle within score envelope | 242 | 228 | 470 | 5,929 | 1.005547 |

Holding the first route fixed, transferring its actual checked bit added 22 second proofs. The joint planner changed 35 first choices relative to the myopic planner and added another six first and 12 second proofs. Its gain over the reset planner was .123750 utility per sequence. TRM reproduced all joint-planner choices. The two-slot return arm improved on myopic memory by .007031 utility and trailed the joint planner by .047734. Relative to the joint planner, two roots improved, 33 tied and five declined. Most of the proof loss was on the first theorem. The bounded oracle leaves room for 62 additional proofs using the same target planner.

The chosen-score intervention changed all 24 eligible first actions and restored the joint-planner choice on every sequence. It gained 17 first proofs and lost three, and gained two second proofs. The sham changed nine choices and removed five proofs in aggregate. The doses matched at each eligible decision, with mean L2 dose .42346 utility in score space. This is evidence of harmful first-choice score placement on the observed development sequences. The improvement from transferring verified feedback belongs to the calibrated planner in this pilot.

The fitted checkpoint SHA-256 is `9355e76ce35a6aba701674df5c524d8b6eb2d2752a9cba76e1c21e22446966dd`. Lean independently compiled 108 successful scripts and replayed ten selected two-proof sequences. The terminal campaign ran on logical CPU 2 under the 2 GiB process-tree cap. Final evaluation peaked at 0.54 GiB committed. All 140 native-enabled gym tests passed. [Numerical receipt](results/ppt_hybrid_rl/ppt-paired-recovery-dev-1131.json.gz)

```text
python -m configs.run_ppt_paired_recovery fit BANK BPD MATHCAN WARM PRIOR ANCHOR FIT RESULTS
python -m configs.run_ppt_paired_recovery eval BANK BPD MATHCAN WARM PRIOR ANCHOR FIT RESULTS
```
