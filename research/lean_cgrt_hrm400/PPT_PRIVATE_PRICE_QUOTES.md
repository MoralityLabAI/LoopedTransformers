# Lean proof plans with paid price quotes

The public-price recurrence reached the proof ceiling before fitting. This pilot withholds actual downstream prices until the agent either buys a quote or commits to a bounded Lean plan. The claims are limited to choosing between two certified plans under effort constraints. Lean certifies the tactics. The benchmark meters the prices and quote observations.

## Cohort and environment

Freeze 48 source roots and 40 disjoint development roots from the LoopedTransformers Horn generator. Use a new seed namespace and exclude earlier premise-budget and public-price roots by canonical hash. Both plans have the same unit length and equal first-producer tariffs. Each plan has private downstream tactics. Replay both plans in Lean and compile the same scripts independently before collecting any fitted-policy scores.

Four equally weighted hidden worlds carry two binary tariff flags. Two quotes read actual downstream tariffs separately. A reconciliation quote reads a third downstream tariff. The partition of worlds induced by either flag alone leaves the cheaper plan ambiguous. Joint flags or the reconciliation quote distinguish the discounted plans when downstream prices are crossed. No public observation identifies the actual world before a quote. Quote charges, plan texts, the total allowance, and the bounds on proof prices are public. The bounds distinguish the zero-surcharge regime from crossed tariffs.

| Regime | Remaining allowance beyond cheapest proof | Quote charges | Decision tested |
| --- | ---: | --- | --- |
| Strict | 0 | 1, 1, 1 | Commit with uncertainty |
| Direct | 1 | 1, 1, 1 | Buy the resolving quote |
| Two quotes | 2 | 1, 1, 3 | Buy two individually inconclusive quotes |
| Too expensive | 1 | 1, 1, 3 | Stop when a second observation would be unaffordable |
| Either plan | 1 | 1, 1, 1 | Skip quotes when either plan can finish |

After commitment charge each paid native tactic in order. An unaffordable step stops the attempt and precludes retry. Report verified proofs, quote and proof effort, and proof success minus one sixteenth per quote effort unit. The primary descriptive endpoint for the learning comparison is mean utility by root, averaging all five regimes and four hidden worlds. The developmental floor is +0.025 over each fitted or frozen policy control. Keep family and regime results and all failed proofs. The analytical policy supplied with the generator is a finite ceiling. A one-quote lookahead using the same generator, forced acquisition, and a fixed first plan are feasibility diagnostics.

## Mechanism and fitting constraints

Keep concrete hidden prices, exact tariffs from the generator, and development outcomes out of the fit. Collect training feedback by logging actions and realized transitions on source roots. A linked-return acquisition policy may compose verified one-step outcomes across matching source states. Its empirical control has the same recurrence and sampler, with observed rollout suffix targets. Fit a full-history TRM and freeze a calibration from the same data and public inputs. The observation format carries the same typed-plan and quote information to each arm. Quote-swap interventions alter an observed bit while preserving native world and charges. Record both intended and actual intervention effects on action and proof. A structural performance gap alone cannot show a benefit from fitting.

The exact/one-step comparison tests bundle acquisition on a finite schedule design. The development cohorts and two generated families support `FSI-01 REDESIGN_OR_PILOT_ONLY`. They supply no prospective cohort, and the protected evaluations stay sealed.

## Feasibility results

The native bank contains 48 source roots and 40 separate development roots, balanced between both generator families within each split. Lean replayed and independently compiled all 176 scripts. The proof meter charges the recorded native tactic prefix step by step. An independent accounting pass recomputed the quotes, actual charges, paid prefixes, and terminal outcomes for all 4,800 control rollouts.

| Structural policy | Certified proof outcomes / 800 | Quote effort | Root-macro utility |
| --- | ---: | ---: | ---: |
| Exact selective acquisition | 640 | 480 | 0.7625 |
| Exact with one-quote lookahead | 560 | 160 | 0.6875 |
| Always buy while reserving minimum proof effort | 640 | 800 | 0.7375 |
| Always buy while any quote is affordable | 0 | 3,040 | −0.2375 |
| Commit to the first plan | 480 | 0 | 0.6000 |
| Exact with the received reconciliation bit flipped | 480 | 480 | 0.5625 |

The exact policy's 80-proof gain over one-quote lookahead came entirely from the two-quote regime. Always buying within the reserved allowance closed as many proofs as exact selection and spent 320 more units. Flipping the paid reconciliation bit removed all 160 direct-quote proofs at the same aggregate charge and preserved two-quote outcomes. That intervention checks the benchmark's observation channel before a learned-policy BPD analysis.

The source archive uses deterministic cuts chosen before observing feedback. Each selected action tape runs under all four worlds, and a restart is anchored to a previously observed paid prefix. The graph joins only observed transitions at the same root, regime, and evidence state. Each tape endpoint uses a terminal outcome metered against a certified proof script. Cut fragments contribute factual edges and supply an empirical suffix target only when they reach commitment. The archive contains 10,400 fragments, 11,904 observed events, and 5,376 paid feedback events. Across 672 distinct source quote decisions, backing up the checked graph produced 341 quote labels without observed terminal suffixes and 240 larger estimates where both labels existed. The graph's targets are values over its sampled continuations. These feasibility results precede the fitted comparison against TRM and frozen calibration.

The one-logical-CPU run peaked at 0.77 GiB committed under the 2 GiB process-tree cap. All 86 native-enabled gym tests passed. The receipts bind the [native proof bank](results/ppt_hybrid_rl/ppt-private-price-bank-1051.json.gz), [source fragments](results/ppt_hybrid_rl/ppt-private-price-fragments-1051.json.gz), and [structural policy replays](results/ppt_hybrid_rl/ppt-private-price-structure-budgeted-1051.json.gz).

## Frozen fitting and intervention design

The policy has a learned joint forecast across eight possible three-quote words and tied success readouts for the two plan slots. Paid feedback filters the forecast. Each predicted quote branches according to the filtered likelihood and subtracts the public fee before a shared budgeted backup. This expands the future observations the model could receive, with no tariff code or winning-plan function in its readout. A known minimum proof effort masks quotes that would make either commitment unaffordable. The full-history TRM sees the same quote identities, paid bits, quoted fees, effort limits, and typed-plan scores. Each policy also predicts the next quote bit from the observed transitions.

The typed-plan scores use the SHA-locked four-source prior trained with MathCan, EvalMath, LoopedTransformers, and BPD data. Its weights stay frozen across the comparison. Use identical initialization and 480 batches of 16 source decisions for the two belief policies, with learned quote forecasts supervised by the same source transitions. Fit one policy to stitched Bellman targets and the other to empirical suffix targets, keeping missing labels masked. Train the TRM on the stitched labels and observed quote forecasts. Both the frozen ridge calibration and a table indexed by the public fee regime and observed history receive stitched labels. The calibrated table deliberately tests whether a flexible lookup saturates this generator. Include the empirical suffix version of that table.

Freeze all weights and calibration tables before evaluating the 40 development roots. The intervention borrows a bit from another private world of the same root and fee regime after the policy buys a resolving observation. Use the reconciliation observation when it was bought, or the second flag when the policy bought both flags. Record changes to the committed plan and outcome, check the treatment and natural trajectories agree through the cost and identity of the purchased quote, and run an equal-observation sham. The sham checks identity and provides no independent perturbation distribution. Success and failure prefixes from the fitted comparisons are replayed in Lean. The benchmark remains developmental and uses structurally exposed evaluation roots.
