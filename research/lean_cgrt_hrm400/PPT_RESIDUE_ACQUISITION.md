# Buying a second Lean observation

The first theorem supplies one paid pass or rejection, as in the residue recovery pilot. Before recovering the second theorem, a policy can keep its entire remaining budget or spend one call on a native membership tactic. The five choices to check are the three target guard sets and two cross-guard sets derived from the public partitions. The check supplies one bit. Restore the saved proof state regardless of the result, so the check grants no hypothesis to the submitted proof. A query must split the support left by the first observation and leave at least one affordable route.

Freeze the query menus, actual native results and source fit separately from development scoring. Use the existing 48 source and 40 reused development pairs. Three budgets per pair share one source check; cases with a common arithmetic expression also share ground truth. The primary endpoint is target proof utility, success minus .025 per native check including setup and the extra query, with a +.025 descriptive gain floor over both TRM and the strongest frozen or trained control. Measure proofs, charges and purchases alongside utility. Report pair averages and arithmetic-expression sensitivity. This reused cohort is development evidence under `FSI-01 REDESIGN_OR_PILOT_ONLY`.

Every acquisition model starts with the exact expected utility of each action under the uniform conditional residue prior. A policy corrects those forecasts and takes one action. A fixed planner completes the target using the resulting posterior. Train matched two-slot reward and full-information value regression policies, a TRM reward policy and an affine reward policy on the same rows and minibatches. Compare with uniform commit and exact acquisition, plus the already frozen affine supervised and reward residue calibrators with and without exact acquisition. The two-slot model first encodes the paid source feedback in a recurrent slot and then updates it jointly with a target action slot. A reset at the boundary tests reliance on this learned state while keeping the public support and exact planner intact. An oracle choosing among realised continuations measures headroom for acquisition choice with the same uniform target planner.

Use 400 minibatches of 16 examples, AdamW at .002, seed 1141 and softmax temperature .20 for the reward objective. Train from the exact expected reward of the finite action policy under native source returns. This isolates one reward-trained decision with a fixed downstream planner. The compared value model observes the same source returns and regresses their magnitudes. All readouts use greedy legal selection in development.

For BPD, intervene when the candidate and uniform acquisition planner favor different legal actions. Exchange the candidate's two action scores and compare with an equal-L2 orthogonal edit that preserves their gap. Require at least three legal actions and log score changes, choices and dose. Group the comparison by pair. These output interventions can test score placement; a causal claim about the internal slots would require prospectively sampled expressions and a directional intervention above the matched sham envelope.

## Development results

| Policy | Second proofs / 120 | Queries bought / 120 | Target utility |
| --- | ---: | ---: | ---: |
| Frozen affine supervised, exact acquisition | 80 | 64 | .423958 |
| Frozen affine reward, exact acquisition | 79 | 63 | .415625 |
| Uniform exact acquisition | 78 | 65 | .408125 |
| Learned affine reward | 78 | 47 | .402083 |
| Two-slot value regression | 77 | 74 | .398125 |
| Two-slot reward, source slot reset | 76 | 70 | .390833 |
| TRM reward | 76 | 61 | .390625 |
| Two-slot reward | 74 | 68 | .374375 |
| Frozen affine reward, commit | 70 | 0 | .344375 |
| Frozen affine supervised, commit | 69 | 0 | .336042 |
| Uniform commit | 68 | 0 | .328542 |
| Two-slot reward, action score swap | 78 | 65 | .408125 |
| Two-slot reward, orthogonal sham | 73 | 64 | .363333 |
| Realised acquisition reference | 82 | 58 | .456250 |

Exact uniform acquisition gained 10 proofs over uniform commit. Frozen supervised calibration gained 11 from the same acquisition option and finished with the strongest measured policy. The two-slot reward policy trailed it by six proofs and .04958 utility. It also trailed TRM by two proofs and .01625 utility, and regressing the returns improved the matched two-slot policy by three proofs. Discarding the learned source slot gained two proofs while retaining the exact public posterior.

The realised reference sees every action's return under the uniform continuation and takes the best action per scenario. It gained two proofs and .03229 utility over frozen supervised acquisition. Its scope excludes other continuations and further purchases. The small gap gives this menu limited resolution for an architecture advantage above the registered .025 floor.

The 120 scenarios comprise 40 pairs and 28 distinct arithmetic expressions. Grouping by expression changes the reward-model contrast to −.02242 against TRM and −.04702 against the strongest frozen calibrator. Seeded descriptive expression bootstrap intervals are [−.06235, .01111] and [−.10055, .00461], respectively. These intervals describe reuse within the pilot and carry no confirmatory guarantee.

The BPD score swap and sham applied to 62 scenarios across 36 pairs at the same mean L2 dose of 2.04423 logit units. The swap gained six proofs and lost two, reproducing the exact uniform acquisition policy across the cohort. The sham gained one proof and lost two. The unedited-minus-sham utility change averaged .01104 across pairs and .00610 across expressions, with descriptive expression interval [−.01081, .02386]. The reused expressions do not support a calibrated upper sham floor. Together with the state-reset result, this gives no evidence that the learned slot improved the acquisition decisions.

At seven remaining calls, the reward model bought 39 queries and closed 37 proofs. Frozen calibration bought 35 and closed all 40. The largest observed opportunity for better acquisition is at five calls, where the realised reference closed 29 proofs and the strongest frozen calibrator closed 25. A follow-up needs more headroom under matched calibration before a larger fit, and an additional decision after an observed query to test whether repeated updates to the slots help.

Lean witnessed 240 source and 200 development query outcomes. It replayed 113 distinct purchased checks and 20 sampled paired proof traces, and independently compiled 52 distinct successful scripts. Execution used logical CPU 2 within a 2 GiB process-tree cap. The evaluation peaked at 0.58 GiB committed. All 150 native-enabled gym tests passed.

| Artifact | SHA-256 |
| --- | --- |
| [Native query bank](results/ppt_hybrid_rl/ppt-residue-acquisition-native-1141.json.gz) | `387f78eb41c4068058d51f21902d87053e55037d67d19eb7f741332634d80637` |
| Source acquisition weights | `191d8638eff991b06f92c216508636ae3fe85817bc73208b8c7c2b705fb2d7e5` |
| [Development receipt](results/ppt_hybrid_rl/ppt-residue-acquisition-audited-1141.json.gz) | `eba71c2fea1ce9aeddf0a9faa9ac8fb307435f774078c72396dfb65dd8c8be41` |

```text
python -m configs.freeze_ppt_residue_acquisition BANK BPD MATHCAN WARM PRIOR NATIVE
python -m configs.run_ppt_residue_acquisition fit BANK BPD MATHCAN WARM PRIOR NATIVE CALIBRATION FIT RESULTS
python -m configs.run_ppt_residue_acquisition eval BANK BPD MATHCAN WARM PRIOR NATIVE CALIBRATION FIT RESULTS
```

`BANK`, `WARM` and `PRIOR` are the existing residue experiment inputs. `CALIBRATION` is its frozen residue fit. Run fit and evaluation in separate processes.
