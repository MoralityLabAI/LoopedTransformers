# COG-09: frozen-router matched carry calibration

Executed at registered commit ecb77ba97af0acf40c3915520413d04d1073d2da. [Passing workflow](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45741176017). Four focused tests passed. The 64 new deeper development roots and two frozen checkpoint seeds yield 1,024 episodes across eight arms. Every online attempt was independently replayed using the pinned EvalMath finite verifier. The full artifact contains the registration, probe fit, all attempted steps and identical-state interventions, per-root outcomes and preflight. [Artifact](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45741176017/artifacts/15080699335); expires 2026-12-26T02:17:00Z. The JSON receipt beside this note preserves the summary printed by the run.

| Arm | Completed / 128 | Calls | Mean completion AUC | Rejected calls |
|---|---:|---:|---:|---:|
| baseline | 79 | 1708 | 0.204590 | 40 |
| erase | 52 | 1773 | 0.159668 | 169 |
| goal_ancestry | 66 | 1732 | 0.186523 | 0 |
| reset_history | 58 | 1763 | 0.167480 | 276 |
| sham0 | 75 | 1720 | 0.196777 | 119 |
| sham1 | 60 | 1757 | 0.171387 | 269 |
| sham2 | 76 | 1714 | 0.200195 | 67 |
| sham3 | 77 | 1708 | 0.203613 | 16 |

Identical-state diagnostic: 1,604 teacher-forced states with carried activations, baseline 1,599 useful top choices, erasure 1,464, and shams 1,589 / 1,550 / 1,599 / 1,600. Maximum measured same-state dose mismatch was 5.96e-7 L2. All neural proposals run four tied cells. Total cell calls vary because early completion ends a run. The primary root-average AUC loss under targeted erasure is 0.044921875. The 95/95 matched sham tolerance envelope is 0.3125, with the maximum over the 64 root-level worst-sham magnitudes used by the planner. The 0.10 scientific floor and 0.20 target are both below that envelope. The normal approximation estimates a resolvable effect of 0.3329447 with 128 roots; the target-minus-floor gap is negative, so the planner reports no finite 80%-power estimate at the requested target.

**REDESIGN_OR_PILOT_ONLY.** The probe direction appears behaviorally important on these public worlds, but the matched shams also alter performance. No causal specificity or sealed confirmation is established. No confirmation roots were generated or scored. The target direction uses a symbolic training label and one fixed probe recipe. The dose is matched at identical pre-action states; closed-loop trajectories diverge. The tighter budget and the structural policy's greedy tie order explain why it is not an optimal shortest-proof oracle. No native Lean proof replay was executed in COG-09. The run is conditional on the fixed COG-03 checkpoints, synthetic graph family and independent-root sampling assumption.

Next design: compare against a norm-preserving direction-specific intervention at a lower prospectively fixed dose; use independent development roots to estimate whether its primary effect clears matched-sham variation. Add a separately trained four-pass no-carry model before attributing advantages to training-time carry. Keep the sealed confirmation population untouched until a new preflight passes.
