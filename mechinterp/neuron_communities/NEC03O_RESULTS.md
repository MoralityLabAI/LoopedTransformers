# NEC-03O A: reset1800-specific source SAE refit and directional-null audit

**Executed 21 November 2026, source-training only.** [Frozen implementation draft PR #261](https://github.com/MoralityLabAI/LoopedTransformers/pull/261), source code lock commit `db9d63f0560433ddbfe8eb831ea71ea2484ed77d`; [successful hosted CPU four-root run 44899750508](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44899750508). Four root-fit+independent-neural-audit jobs, source-lock preflight, and pure-stdlib four-root aggregation all **passed**. Source summary artifact `nec03o-four-root-reset1800-source-sae-audited`, artifact ID `14705856432`.

The eight native teachers were the **exact original NEC-03N-B reset-AdamW 900+900 source checkpoints**. This run trained **zero** further native updates. The 900-step continuous-optimizer ancestry experiment remains an original failed result, unaffected. No five-input development-evaluation or reserved contexts, embargoed five-input roots, or original four-input confirmation worlds were scored.

## Source-only ITT findings

| World | Arm | Native correct/2560 | Old900 SAE MSE | Fresh reset1800 SAE MSE | Directional coverage | Median retained physical dose | Source eligible |
|---|---|---:|---:|---:|---:|---:|---|
| dev5-000 | composed | 2357 | 0.008142 | 0.011966 | 96.71% | 59.16% | yes |
| dev5-000 | additive | 2557 | 0.007370 | 0.006451 | 99.87% | 61.00% | yes |
| dev5-001 | composed | 2560 | 0.008821 | 0.008495 | 98.97% | 67.87% | yes |
| dev5-001 | additive | 2560 | 0.005895 | 0.004783 | 100.00% | 36.57% | yes |
| dev5-002 | composed | 2547 | 0.010150 | 0.010886 | 97.43% | 52.18% | yes |
| dev5-002 | additive | 2560 | 0.007678 | 0.009334 | 97.77% | 64.62% | yes |
| dev5-003 | composed | 2387 | 0.017362 | 0.019824 | 99.80% | 60.98% | yes |
| dev5-003 | additive | 2560 | 0.010199 | no SAE: source candidate missing | unavailable | unavailable | **no** |

**Native source competence: 8/8 arms, 4/4 complete worlds.** Fresh new checkpoint-specific TopK16 SAE trained on **7/8** arms; all seven had 16 exactly active features on 100% of their source-fit activation samples and normalized MSE below 0.03. All seven re-ranked 12 recurrent visits on the disjoint 40-group source partition, and their NEC03K disjoint active-code nulls passed the ≥80% angular coverage, ≥80% positive dose, ≥25% median retained dose, and physically equalized dose gates. Independent auditing re-enumerated all available sham candidates from saved 32D contributions. Among the seven eligible arms, positive-dose coverage was 100% and minimum observed angular coverage 96.71%.

**Source-control feasibility: 7/8 arms and 3/4 paired worlds.** These proportions include the missing source candidate in the fixed intent-to-treat denominator. New versus old reconstruction MSE is descriptive; normalization and native weights belong to different frozen model conditions. Three of the seven fitted arms have lower new-model normalized MSE; four have higher new-model normalized MSE.

### The specific failed source candidate

`dev5-003` additive remains fully source-competent (**2560/2560**) but its 1800-step source graph has no qualifying sparse community. The frozen fallback mandates two nonhub neurons at each of its third and fourth chronological focal positions, visits **4 and 8**. At visit 8 the only two source-attributed neurons, `8:10` and `8:21`, occur in **40/40** source groups and are classified as graph hubs, leaving zero nonhub candidates at that required visit. At visit 4 there are eligible nonhubs, but the fixed two-visit source fallback requires both visits. The 900-step source model had selected an admissible fallback at visits 4 and 8, and its original source geometry passed; the recovered reset1800 state loses that specific source instrument.

This is a **source-site availability** failure under a frozen graph/fallback policy. It is not a failed native prediction checkpoint. Post-hoc alternate fallback visits would be a new intervention policy and require an independent future registration. This root remains in the original denominator.

## Scientific scope and next roots

All checks were **in-sample source-training feasibility**, with independently seeded Boolean worlds as root units and paired composed/additive arms nested inside each world. The original NEC03H/03K causal effects and **FSI-01 REDESIGN_OR_PILOT_ONLY** status remain unchanged.

The next preregistered task is **NEC-03O B: dev5-004..007 source-only native 900+900 reset-AdamW**, followed by fresh 40/40 source SAE fitting and source-null geometry, preserving the same graph/fallback and the failed-source admission rule. No new-world development-evaluation or reserved contexts may be accessed. A later signed causal test needs new-root effects and the separately required number of independently eligible sham roots.
