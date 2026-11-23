# Adaptive Lean queries and proof attempts

Extend the native residue task so a policy can choose proof routes directly, stop, or pay for a query. It can buy two target queries, including after a failed route. Spending always comes from the same recovery allowance. A failed route removes its residue group. A query intersects the current support with the witnessed outcome and restores the saved Lean state. Every subsequent choice may use that updated support and the remaining budget.

## Preflight

Compare exact finite dynamic programming under the frozen affine supervised, affine reward and uniform residue priors, each with zero, one and two queries. Use the standard three, five and seven target-call budgets. Include a true-residue route planner as a ceiling for this restricted action grammar. Compare proof utility (closure minus .025 per charged tactic) and successful proofs. Use the previous 48 source and 40 development pairs for this pilot and report how often the exact controls actually buy twice or interleave a proof attempt and a query. A second-queries fit on this task requires at least .075 ceiling headroom above the strongest exact planner as a source pilot allocation rule. The subsequent candidate target remains +.025 utility over TRM and the strongest non-oracle arm. Any results here are reused development evidence under `FSI-01 REDESIGN_OR_PILOT_ONLY`.

The first source run used the original calibrators on their own fitted expressions and produced .04340 headroom. For the compute decision, refit both calibrators in four folds grouped by arithmetic expression, with the original fit settings. The held-expression estimate was .09705, and the reused development estimate was .10958. The out-of-fold source estimate clears the .075 allocation threshold. The original in-sample result remains in the receipt.

| Development prior | No queries | Up to one | Up to two | Utility with two |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 68 / 120 | 78 / 120 | 79 / 120 | .417708 |
| Frozen affine return | 70 / 120 | 79 / 120 | 80 / 120 | .422708 |
| Frozen affine supervised | 69 / 120 | 80 / 120 | 81 / 120 | .431667 |
| Actual residue | 91 / 120 | | | .541250 at zero |

The strongest development planner used two queries in 13 scenarios and queried after a rejected route in three. Its second-query option added one proof and .00833 utility. The actual-residue headroom includes avoiding wrong route attempts and excess purchases, so it does not measure the incremental value of a second query alone.

## Sampled reward protocol

Fit a recurrent actor whose two slots persist between selected actions. One shared transformer block updates the retained slots on each result and reads them with a candidate action token. Train a persistent answer-and-latent TRM on exactly the same visible history. Compare a two-slot value regressor with matched weights at initialization. It regresses the realised perfect-information values of legal actions at its visited states, while the reward arms receive returns only for their sampled trajectories.

Use 240 batches of 16, AdamW at .002, seed 1163 and temperature .16 during training. Use the same episode indices and uniform sampling draws across arms. The reward loss is REINFORCE with a frozen exact-uniform action-independent baseline and return-to-go under the sampled native-witnessed trajectory. The public mask and exact-uniform scores are shared inputs. This is on-policy sampling against a finite replay of verified Lean transitions. It makes no live Lean calls inside the optimizer.

Freeze the fit before greedy development scoring. Compare against both frozen affine planners and the uniform planner at depth two. Reset learned slots to their initial state after each received event for an additional control. For BPD, after the first eligible paid query, replace the feedback-slot displacement with the one induced by the opposite query bit, while keeping the actual public support, other slot and native continuation. Apply an orthogonal matched-dose sham at the same decision boundary. Log dose, action and actual outcomes. Independently compile every distinct successful development script and replay sampled native sequences.

## Sampled reward results

| Policy | Proofs / 120 | Tactic calls | Queries | Two-query cases | Utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen affine supervised planner | 81 | 1,168 | 80 | 13 | .431667 |
| Persistent two-slot reward | 81 | 1,178 | 90 | 22 | .429583 |
| Persistent two-slot value | 81 | 1,181 | 95 | 27 | .428958 |
| Frozen affine return planner | 80 | 1,171 | 77 | 12 | .422708 |
| Persistent TRM reward | 80 | 1,178 | 60 | 12 | .421250 |
| Uniform planner | 79 | 1,155 | 79 | 12 | .417708 |
| Two-slot reward, slot reset | 81 | 1,179 | 89 | 22 | .429375 |
| Two-slot reward, opposite query bit | 81 | 1,179 | 91 | 23 | .429375 |
| Two-slot reward, orthogonal sham | 81 | 1,178 | 90 | 22 | .429583 |

The reward-trained slots gained one proof over TRM at the same number of calls. They traded three wins for three losses against the strongest frozen calibration and used ten more calls. The value regressor and reward learner exchanged five proof wins each. The +.025 utility floor was missed against all three controls.

The 120 scenarios share 40 proof pairs and 28 arithmetic expressions. The equal-expression reward-policy difference was +.01295 relative to TRM, with descriptive bootstrap interval [−.02862, .05451], and −.00422 relative to frozen supervised calibration, with interval [−.04187, .03383]. These reused roots and the single trained seed give pilot evidence.

The BPD arms edited 64 scenarios across 40 pairs at the same mean L2 dose of .49700 in the feedback slot. The opposite bit changed one trajectory and added one charged call. The sham preserved every trajectory. Neither changed a proof outcome. Slot reset also preserved every proof outcome. The exact public support is available at each decision and already summarizes the observed evidence for the frozen planner. This experiment does not establish that carrying the evidence in a learned slot improves proofs. The small reused pilot cannot calibrate a prospective sham floor.

Lean independently compiled all 40 distinct successful development scripts and replayed 35 selected sequences. The two-query structural preflight also verified 23 sampled sequences per receipt, including check-after-rejection paths. All 153 native-enabled gym tests passed. Execution was pinned to logical CPU 2 under a 2 GiB process-tree cap.

| Artifact | SHA-256 |
| --- | --- |
| [Source planning receipt](results/ppt_hybrid_rl/ppt-residue-rollout-source-1157.json.gz) | `1a3d4d958467fa49179152127c69fd360f46942ebcbfe4527b45503dd92d7cff` |
| [Expression-fold planning receipt](results/ppt_hybrid_rl/ppt-residue-rollout-folded-1157.json.gz) | `9b752f31c627461fb94707a50bcfa4f1095d13886df84c9e2ebd272e4101d2d2` |
| [Development planning receipt](results/ppt_hybrid_rl/ppt-residue-rollout-development-1157.json.gz) | `ed88433655b4c487626daa507fd6d23415e7c825f3d1796298c52fdeb8adf8fb` |
| Source sampled-policy weights | `1b7d4698d2421d54950715e2a3f72e6971807d77245ac59884e88d3e8ca02964` |
| [Sampled-policy development receipt](results/ppt_hybrid_rl/ppt-residue-online-audited-1163.json.gz) | `6a8947e1b7dccd1394c2d4e5eafa3d496b294746b35812b7d0e49f5a407b7b93` |

```text
python -m configs.run_ppt_residue_rollout source BANK BPD MATHCAN WARM PRIOR NATIVE CALIBRATION RESULTS
python -m configs.run_ppt_residue_rollout source BANK BPD MATHCAN WARM PRIOR NATIVE CALIBRATION RESULTS --folded
python -m configs.run_ppt_residue_rollout development BANK BPD MATHCAN WARM PRIOR NATIVE CALIBRATION RESULTS
python -m configs.run_ppt_residue_online fit BANK BPD MATHCAN WARM PRIOR NATIVE CALIBRATION FIT RESULTS
python -m configs.run_ppt_residue_online eval BANK BPD MATHCAN WARM PRIOR NATIVE CALIBRATION FIT RESULTS
```

The input artifacts come from the residue and paid-query experiments. Use a separate output path for each receipt and separate processes for fitting and evaluation.
