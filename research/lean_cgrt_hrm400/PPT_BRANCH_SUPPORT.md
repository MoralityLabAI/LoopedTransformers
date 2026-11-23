# Learning how to fund a Lean branch decision

The scoped rollback experiment had an explicit scope rule, and calibrated features reached its utility ceiling. This pilot adds a support period. A policy can pay Lean to reveal routing bits before it learns which of two query keys will be used. The policy then chooses between two scoped proof roots with identical model-visible goals and opposite scope outcomes. A common witness policy pays to read both original aliases and the actual scope. It uses the original values to select the witness. The audited scoped worlds give closure in eight of eight continuations after rollback and four of eight after committing the unseen replacement.

The support session has three private bits. Four paid actions query a single address or a two-address check. A query key of one, two or three addresses selects the native parity bit that determines which branch rolls back. The candidate receives an address and can write the observed bit into a fast evidence record. Direct evidence and messages composed from two recent reads have distinct slots in a tied transformer. The relation between addresses is public, so the full-history TRM and aligned calibration receive the same derived evidence. The scope routing bits themselves come only from witnessed native diagnostics.

Each source regime advertises two possible query keys. Freeze nine source decks with query keys one through six, and six development decks pairing a familiar key with the previously unseen three-address key. Score every possible native bit assignment, query draw and support allowance of one or two calls. Deduct .05 utility per purchased support check. All policies pay the same three query diagnostics; the utility omits this common charge. Retain total native calls separately. For each selected branch, evaluate all eight scoped worlds consistent with the observed scope on each original development proof payload.

Train the reward policy and TRM from conditional rewards backed up through the current continuation policy. Compare same-model optimum imitation, a frozen quadratic readout, a frozen slot readout, and a certainty-equivalent planner whose frozen decoder is fitted on source branch outcomes. The planner backs up possible future native observations using that fitted decoder. A separately labelled exact native observation tree provides the structural ceiling. The exploratory criterion is +.025 proof-minus-support-cost utility over both TRM and the strongest fitted or calibrated arm.

Use 144 optimizer steps, batches of 64 public states, AdamW at .002, seed 1187 and .10 for soft continuation targets. Refresh the reward backups every eight steps. Freeze the fit before development readout. For BPD, force the observed keys 3 and 4 on the novel query key 7. Replace the composed slot with its value under an opposite native source bit while keeping the actual proof world unchanged. Compare an orthogonal equal-L2 edit at the same decision. Keep the prior scoped roots in their archived source and development partitions. Their new decks are a transfer pilot under `FSI-01 REDESIGN_OR_PILOT_ONLY`.

## Development results

| Policy | Rollback choices / 192 sessions | Paid support checks | Proof utility |
| --- | ---: | ---: | ---: |
| Fitted decoder with native observation planner | 144 | 224 | .816667 |
| Exact native observation tree | 144 | 224 | .816667 |
| Ridge on frozen slots and history | 136 | 160 | .812500 |
| Quadratic history calibration | 140 | 208 | .810417 |
| Two-slot reward | 112 | 80 | .770833 |
| Two-slot optimum imitation | 122 | 256 | .751042 |
| Full-history TRM reward | 96 | 0 | .750000 |

The reward policy gained .02083 utility over TRM and .01979 over imitation. It gained 16 rollback selections over TRM and made ten fewer than imitation while saving 176 support checks. It missed the .025 floor and trailed the fitted decoder planner by .04583. For the unseen query key, the reward policy selected rollback in 48 of 96 sessions. The fitted decoder planner did so in 64 of 96.

The 192 sessions combine eight native wiring worlds, six decks, two budgets and two draws. Each decision is reused across the same 18 archived proof payloads and the eight compatible scoped valuations per proof. Thus 21,888 successful outcomes for the reward arm and 24,192 for the strongest calibration are aggregate counts over 27,648 continuations evaluated from the certified branch table. They represent 192 decisions over eight wiring worlds. The model sees the projected branch state, and the task isolates branch selection with an audited shared witness continuation.

In the eight forced-key BPD states, the natural composed slot selected rollback twice. The opposite-bit donor selected it six times, and the equal-dose orthogonal sham selected it four times. Both edited the composed cue at mean L2 dose 1.31471. This finite diagnostic shows a readout sensitive to the slot, with outcomes favoring the opposite-bit donor over the natural direction. Eight worlds cannot resolve a prospective sham threshold.

## Fixed-weight stage swaps

| Post-hoc intervention | Rollback choices / 192 | Paid support checks | Utility |
| --- | ---: | ---: | ---: |
| Learned support policy, calibrated branch readout | 112 | 80 | .770833 |
| Calibrated support planner, learned branch readout | 124 | 224 | .764583 |
| Plan support using the learned branch readout's predicted values | 128 | 136 | .797917 |

Replacing only the branch readout preserved the reward actor's aggregate score. Giving it the calibrated support schedule added rollback choices and cost enough to reduce utility. Backing up the learned branch readout's centered and clipped return predictions added .02708 utility over its joint policy. It still trailed the calibrated planner by .01875. Support has value only where the branch reader can use the resulting evidence, so the next learning comparison should verify the composed branch read separately before crediting purchases with its predictions.

The native bank witnessed all 56 route observations and checked their complements. Lean replayed 13 selected support and scoped proof traces in the registered evaluation and 16 in the stage-swap rerun. The latter reproduced every registered record and BPD result. All 158 native-enabled tests passed. Execution used logical CPU 2 within a 2 GiB process-tree cap.

| Artifact | SHA-256 |
| --- | --- |
| [Native support observations](results/ppt_hybrid_rl/ppt-branch-support-native-1187.json.gz) | `d506202d32ca0b08e3f2866691b276346f064fd85618a427922d538f914fc48a` |
| Source fit | `0c8e7d5eaeec6bab3f46f69c72e3cf5d13981cf1ac081761072890428f6d0ea3` |
| [Registered development and BPD receipt](results/ppt_hybrid_rl/ppt-branch-support-audited-1187.json.gz) | `21074b727040827191fc4e639b9978f9c690a2b4e1d962d84436fde498f8fcc3` |
| [Fixed-weight stage-swap receipt](results/ppt_hybrid_rl/ppt-branch-support-stage-swaps-1187.json.gz) | `4207390d06e1a62ca34bff9677cee19fd260fdb62c59ec8f73f119a17637721a` |

```text
python -m configs.freeze_ppt_branch_support MATHCAN SUPPORT
python -m configs.run_ppt_branch_support fit SCOPED_BANK SUPPORT MATHCAN FIT RESULTS
python -m configs.run_ppt_branch_support eval SCOPED_BANK SUPPORT MATHCAN FIT RESULTS
python -m configs.run_ppt_branch_support eval SCOPED_BANK SUPPORT MATHCAN FIT STAGE_RESULTS --stage-swaps
```
