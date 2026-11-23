# Recovering original Lean proofs from native residual states

This pilot searches continuations of the original archived BPD, EvalMath, LoopedTransformers and MathCan proofs. Each case starts at two distinct residual states of one checked proof. The target starts earlier in the proof and a one-tactic test at the later residual can supply verifier feedback. All target tactics execute on the target state. A closed sibling trial supplies evidence and never earns target proof credit.

## Registered design

Replay each archived proof to collect checkpoints. Select the initial state and the states two or three recorded calls before closure, deduplicating prefix lengths. Use the adjacent later checkpoint as the sibling when its public view differs. For one-tactic MathCan payloads, place the original goal in a conjunction with `True` to create separate residual states. Freeze the same prior archive's 15 training and 18 development payload lineages. These are new tactic searches on previously used proof payloads, so this is an `FSI-01 REDESIGN_OR_PILOT_ONLY` pilot.

Use public type-directed apply proposals and a fixed native tactic menu. Collect every registered action to depth two, continuing from progress states and retaining the original snapshot on a rejection or unchanged goal. Keep Lean resource-limit responses separate. The policy can stop, buy one sibling check before its first target action, or try a registered target tactic. Score budgets of one and two target calls. Charge .025 utility per target call and .05 for the sibling check. A target success earns one. Compile every distinct returned successful script with its original accepted prefix.

The four-source mixed prior supplies the same frozen action scores to all arms. The recurrent candidate passes messages between a slot for the target state and its last result and a slot for the sibling evidence. A paid native disposition changes the sibling slot at the next decision. Match the full-history TRM on all input cues. Train both by backing up the current continuation policy on the native source transition table. The same-architecture imitation arm fits exact optimistic source returns. Frozen ridge calibration fits those same source targets on raw history and on raw history plus the frozen recurrent representation. Include the original prior and the observed native tree's structural ceiling.

Use one seed (1201), 120 optimizer updates, batches of 48, AdamW at .002 and .12 soft policy targets. Refresh the current-policy backup every 12 updates. Freeze the fit before evaluation. Report root-macro utility, target proofs and queries purchased. The descriptive threshold is +.025 utility over TRM and the strongest trained or calibrated control.

For BPD, force one paid sibling `omega` trial and compare the true status embedding with an alternate status witnessed for the same tactic in the same source. Edit the sibling slot at the first target decision. Match the L2 norm with an orthogonal sham. Run all three traces with the actual native outcome and the same allowable target continuations. In the free-policy reset control, preserve the menu, sibling identity and actual verifier feedback while suppressing the sibling's outcome in the model input.

## Development results

The 33 original proof scripts compiled independently. The native bank has 43 source and 52 development residual cases. Nine development targets close in one registered call, and 14 more require two registered calls. The remaining targets supply search and stopping cases within this grammar.

| Policy | Target proofs / 104 | Native calls | Paid sibling checks | Root utility |
| --- | ---: | ---: | ---: | ---: |
| Two-slot reward | 32 | 133 | 8 | .267477 |
| Two-slot imitation | 32 | 172 | 46 | .249190 |
| Full-history TRM reward | 28 | 155 | 19 | .222222 |
| Frozen-slot ridge calibration | 19 | 218 | 76 | .105556 |
| Raw-history ridge calibration | 19 | 249 | 102 | .091435 |
| Four-source prior | 0 | 156 | 0 | −.037500 |
| Two-slot reward, probe feedback reset | 32 | 135 | 8 | .267014 |
| Observed native tree ceiling | 32 | 46 | 0 | .290046 |

The reward policy reached the observed tree's proof count. All its extra proofs over TRM were two-tactic BPD continuations. TRM found no extra proofs in the other direction. The utility gains were .04525 over TRM and .16192 over the best frozen calibration. Imitation solved the same targets, and the reward policy used 39 fewer calls for a .01829 gain. The .025 gate was met against TRM and frozen calibration and missed against imitation.

The reset preserved every free-policy proof. In the forced-probe BPD comparison, all 24 eligible cases came from BPD, the only source with both accepted and rejected sibling `omega` responses. Normal, donor and sham obtained 14, 14 and five proofs, respectively, at a mean matched L2 edit norm of 2.82761. The donor changed six trajectories and the sham changed nine. The sham's larger damage and the free-policy reset provide no evidence for a useful status-specific sibling slot on this cohort. The 104 scored scenarios reuse 18 original proof roots and two budgets. The comparisons are descriptive.

## Matched stopping calibration

After development scoring, select an independent stopping margin for each frozen readout using only its archived source cases. The thresholds range over 0, .025, .05, .10, .15 and .25 in its score units. The native choice rule stops when the leading action's score has less than that margin above stopping. Select by source root-macro utility, resolving ties toward the smaller margin. Keep the same action scores and developmental case set.

| Policy with source-selected stopping | Target proofs / 104 | Calls | Root utility |
| --- | ---: | ---: | ---: |
| Two-slot reward | 32 | 118 | .270949 |
| Two-slot imitation | 32 | 131 | .261690 |
| Full-history TRM reward | 28 | 106 | .234375 |
| Frozen-slot ridge calibration | 19 | 95 | .141551 |
| Raw-history ridge calibration | 19 | 192 | .105556 |

Stopping calibration narrowed the reward gain over imitation to .00926. The reward policy retained its 32 proofs and remained .03657 ahead of TRM and .12940 ahead of the strongest frozen readout in root utility. This is a post-hoc control fit on the archived source, with development exposure to the original comparisons.

## Binder presentation check

The learned policy's four extra proofs over TRM all use the locally named `h9999` at the second action. To test reliance on that name, remap every local identifier in the inputs to the frozen mixed prior to a case-specific opaque alias. Preserve the typed hypothesis, candidate association and executable native action. Freeze all policy and calibration weights. Use the original stopping rule for this post-hoc replay.

| Policy with opaque binders | Target proofs / 104 | Calls | Root utility |
| --- | ---: | ---: | ---: |
| Two-slot reward | 32 | 138 | .267361 |
| Two-slot imitation | 32 | 179 | .248495 |
| Full-history TRM reward | 27 | 161 | .208796 |
| Frozen-slot ridge calibration | 17 | 229 | .083102 |
| Raw-history ridge calibration | 18 | 249 | .081829 |

The reward fit preserved every reachable proof with opaque binders. The strongest calibration and TRM remained behind. The comparison leaves the measured gain localized to the reused native BPD residuals. A prospective proof cohort must vary the structural continuation as well as binder presentation.

## Provenance

Lean independently compiled 23 distinct successful continuations from the scored traces, including their original prefixes. The original evaluation replayed 35 selected traces. The matched stopping rerun replayed 63 traces and reproduced all original results and BPD traces. All 162 native-enabled gym tests passed. Execution used logical CPU 2 and a 2 GiB process-tree cap.

| Artifact | SHA-256 |
| --- | --- |
| [Native residual bank](results/ppt_hybrid_rl/ppt-residual-search-native-1201.json.gz) | `f3f8ccf702932a5834d9e7637ba8b3a3cccf4e67df1c2cd9e62c2ebe030130fb` |
| Source fit | `e91be9d89b31025f315fc45f261440ab806b17a0d1246efbb8dbd415ff09a478` |
| [Registered evaluation and BPD](results/ppt_hybrid_rl/ppt-residual-search-audited-1201.json.gz) | `1caeee69340d7b99d99f2831b74c0e396cd87e58e90a8b48da3f4647fe03af72` |
| [Matched stopping receipt](results/ppt_hybrid_rl/ppt-residual-search-source-stop-1201.json.gz) | `07a72e854cdba9d9a93783077daba12f736aa2c943f16bf2f9df0ec27022817a` |
| [Opaque binder receipt](results/ppt_hybrid_rl/ppt-residual-search-opaque-binders-1201.json.gz) | `1a31d32719398973427d5b6f53bf1418e4122f0a5f72a1e057e0458d476759f7` |

```text
python -m configs.freeze_ppt_residual_search ARCHIVED_ROLLBACK_BANK MATHCAN BANK
python -m configs.run_ppt_residual_search fit BANK MATHCAN WARM PRIOR FIT RESULTS
python -m configs.run_ppt_residual_search eval BANK MATHCAN WARM PRIOR FIT RESULTS
python -m configs.run_ppt_residual_search eval BANK MATHCAN WARM PRIOR FIT STOP_RESULTS --stop-match
python -m configs.run_ppt_residual_search eval BANK MATHCAN WARM PRIOR FIT OPAQUE_RESULTS --opaque-binders
```
