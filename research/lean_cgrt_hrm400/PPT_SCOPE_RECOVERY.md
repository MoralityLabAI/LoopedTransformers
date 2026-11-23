# Keeping and restoring evidence in Lean proof search

The preceding adaptive-query task supplied an exact posterior at every decision. Here a tactic observation is available only while its scope is open. The target remains the original proof obligation from BPD, EvalMath, LoopedTransformers or MathCan, with an added witness chosen by the parity of two active alias values.

## Registered pilot

Start before a tentative update to one named address. The actor can buy either original alias value, once per alias. Opening the tentative scope expires those read actions. The actor can now buy the proposed new value and must pay for a native check that reveals whether the update survived. In the terminal scope, no further values may be bought. Keep the proposed value at the edited address after a successful scope and restore the checkpoint value there after rollback. A failed witness consumes too many calls for a second proposal. Each optional diagnostic and the mandatory scope check costs one native tactic and .05 utility. Utility also counts each closed proof as one. Test one, two and three optional diagnostic calls.

For each original payload, Lean verifies all 16 worlds of three independent alias bits and one scope bit. Verify that the structured proof observation is identical across valuations, that every paid value comes from an actual rejected tactic, and that the proof succeeds only with its correct witness. Independently compile the verified proofs with their original axiom inventories. Use the 15 original source and 18 original development payload roots from the archived rollback experiment. This creates new scoped proof statements on previously exposed payload roots. Use `FSI-01 REDESIGN_OR_PILOT_ONLY`.

The two-slot candidate writes addressed evidence with a tied block, keeps a checkpoint and reads the proposed write before scope resolution. It resolves the active slots with the publicly observed scope transition and iterates the same block across messages. The persistent TRM control receives the full four-event history and the same public state. All learned arms use the same native source observation graph. Train the candidate and TRM by expected return under the current continuation policy, then distill each improvement back into its policy. Compare exact-optimum imitation in the same architecture and frozen ridge calibration on raw history, explicitly scope-aligned history, and the frozen two-slot features. The analytic bridge from the queried bits to the native witness gives a structural ceiling.

Use 180 updates, AdamW at .002, seed 1171, and .10 improvement temperature for both the policy iteration and imitation targets. Greedy evaluation uses the frozen fit. The exploratory primary target is +.025 utility over both TRM and the strongest fitted control, averaging budgets and valuations within each original payload root. The checkpoint-reset control removes the two old observations after opening the scope. At fixed three-query histories, edit the resolved slot with the donor representation from the opposite scope when the active bit would differ. Compare an orthogonal edit at the same L2 norm. Both interventions preserve the real Lean state, actual scope, other slot, and queried observations.

## Development results

| Arm | Proofs / 864 | Paid diagnostics | Utility |
| --- | ---: | ---: | ---: |
| Scope-aware reward policy | 648 | 2,304 | .616667 |
| Scope-aware imitation | 648 | 2,304 | .616667 |
| Scope-aligned ridge | 648 | 2,304 | .616667 |
| Exact native observation tree | 648 | 2,304 | .616667 |
| Ridge on frozen scope slots | 648 | 2,464 | .607407 |
| Full-history TRM reward | 620 | 2,592 | .567593 |
| Raw-history ridge | 576 | 2,304 | .533333 |
| Scope-aware reward, checkpoint reset | 432 | 1,936 | .387963 |

The reward policy gained 28 proofs and .04907 utility over TRM. All extra proofs came at the three-query budget, where it closed 288 and TRM closed 260. At the one-query budget both closed 144, and TRM bought one extra observation in every world. The reward policy's mean utility gain over TRM was positive on all 18 payload roots. Scope-aligned ridge and imitation attained the same proof count, cost and utility as the reward policy and the exact tree. The .025 gate was met against TRM and missed against those stronger controls.

At a forced three-query terminal decision, grouping by payload and actual scope leaves 144 of 288 worlds solvable from the current view alone. The missing history matters experimentally. Resetting the checkpoint removed 216 proofs in greedy replay. The counterfactual slot patch targeted 144 fixed histories, 72 kept and 72 rolled back. Every unedited choice was certified, all 144 opposite-scope slot donors selected the alternate witness, and all 144 orthogonal shams preserved certification. Both edits used a mean L2 dose of 6.65952. The donor comparison measures decision sensitivity in this finite task. The sampled roots and sham do not calibrate a prospective mechanism threshold.

Lean independently compiled 528 scoped certificates and replayed 28 selected sequences, including fixed-history donor and sham choices for each edited address and scope direction. The model and controls saw only the projected address, phase, budget and observed bits. The original payload is proved by its audited continuation. The 18 development payloads therefore exercise two distinct projected decision graphs, and the score does not establish transfer to independent proof search policies. The checkpoint rule is explicit in the candidate and in the strongest calibration features. The exact tree ceiling leaves no utility headroom for another model on this grammar.

| Artifact | SHA-256 |
| --- | --- |
| [Native scoped bank](results/ppt_hybrid_rl/ppt-scope-recovery-native-1171.json.gz) | `648d6ae14c17bdada0f0266bcd933d1ecc43038d2b15752148933c55ebe51dc6` |
| Source policy fit | `d3fa7a8b3f9da093693909d875772a949e6811850109511b5d7c96a4ae4fef16` |
| [Development and BPD receipt](results/ppt_hybrid_rl/ppt-scope-recovery-audited-1171.json.gz) | `3bae3f9b2d6521b3991d497c8ce34990ecef4b3208e6a908a1717c6ad1f19b42` |

```text
python -m configs.freeze_ppt_scope_recovery ARCHIVED_ROLLBACK_BANK MATHCAN BANK
python -m configs.run_ppt_scope_recovery fit BANK MATHCAN FIT RESULTS
python -m configs.run_ppt_scope_recovery eval BANK MATHCAN FIT RESULTS
```

All 156 native-enabled gym tests passed. Collection, fitting and evaluation used logical CPU 2 with a 2 GiB process-tree cap.
