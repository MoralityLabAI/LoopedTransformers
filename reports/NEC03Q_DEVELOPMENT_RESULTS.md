# NEC-03Q: audited eight-world development causal result

**Completed 21 November 2026.** Development-only causal intervention with exactly one primary difference-in-differences contrast on the original eight five-input source-engineering worlds. Original executable and protocol were preregistered at [commit cca6c35](https://github.com/MoralityLabAI/LoopedTransformers/commit/cca6c3549f046487b6668a0ea0524dcc34ea1f92), before any five-input development-evaluation readout. The first [hosted original scorer run 44904576625](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44904576625) generated six completed root causal datasets. Five of the original independent audits passed and root7's initial logit reconstruction failed; the original failed audit is preserved.

The [separately registered read-only audit-v2 run 44905096670](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44905096670) used exactly the originally saved model/SAE checkpoints and raw causal records and passed **all six root independent audits plus the eight-root aggregate**, without new scoring-policy choices. The auditor's source/sham donor arithmetic was changed to use the same float32 decoder matrix-vector product as the original scorer, preserving the original logit tolerance. All original causal raw JSONL SHA256s and reports were verified byte-for-byte and unchanged. [Amendment method](NEC03Q_AUDIT_V2_METHOD.md). Final aggregate evidence artifact **14707197700**, named **nec03q-original-eight-root-causal-readonly-amended-audit**.

## Preregistered primary root-level result

The primary root outcome was

\[
D_r = \frac{(C^{\mathrm{composed}}_{\mathrm{SAE}}-C^{\mathrm{composed}}_{\mathrm{sham}})
-(C^{\mathrm{additive}}_{\mathrm{SAE}}-C^{\mathrm{additive}}_{\mathrm{sham}})}{768},
\qquad \bar D=\frac{1}{8}\sum_{r=0}^7 D_r.
\]

Six source-qualified roots (0, 1, 2, 4, 6, 7) were scored on all 24 complete five-input development contexts×32 focal masks in **each** paired model arm. Source-ineligible roots 3 and 5 were preregistered as **zero** in the 8-root ITT denominator and their development-evaluation groups were not scored.

| World root | Composed SAE correct | Composed sham correct | Additive SAE correct | Additive sham correct | Root DID percentage points | Audit |
|---|---:|---:|---:|---:|---:|---|
| dev5-000 | 616 | 612 | 744 | 744 | +0.521 | independent pass |
| dev5-001 | 715 | 715 | 768 | 768 | 0.000 | independent pass |
| dev5-002 | 635 | 636 | 683 | 682 | -0.260 | independent pass |
| dev5-003 | not scored | not scored | not scored | not scored | 0.000 | preregistered source ITT miss |
| dev5-004 | 548 | 545 | 768 | 768 | +0.391 | independent pass |
| dev5-005 | not scored | not scored | not scored | not scored | 0.000 | preregistered source ITT miss |
| dev5-006 | 561 | 567 | 762 | 762 | -0.781 | independent pass |
| dev5-007 | 644 | 640 | 753 | 751 | +0.260 | v2 amended independent pass |

Totals across the original six measured roots: composed SAE **3719/4608**, composed sham **3715/4608**, additive SAE **4478/4608**, additive sham **4475/4608**. Hence the composed arm gained **4** correctly scored cases versus its matched sham, and the additive arm gained **3**. The resulting original prespecified net difference-in-differences is **+1 correctly scored case** over 768 cases per model root. The eight-root ITT macro average is **+0.0001627604 correctness fraction**, or **+0.016276 percentage points**.

The two-sided exact exploratory root sign-flip p-value is **1.0**, with five nonzero root effects. The test's symmetry/exchangeability assumptions are scientific assumptions, not established by exact arithmetic. The result is far below the prespecified **+2.0 percentage-point** meaningful causal effect floor.

### Mechanistic interpretation of this development diagnostic

The matched active-code source/sham controls were drawn from the frozen 128-feature TopK16 dictionaries on original reset1800 native weights, with 40/40 disjoint source fitting/ranking and the original exhaustive disjoint control search. Every admitted input had its code directions calibrated to the hubs-only clean-chosen 32D post-MLP-down gradient and physically equalized L2 dose; any invalid active-code control at a selected visit gave the whole input two identical hubs-only trajectories. The independent audit reproduced every saved 32D sham selection, dose, true Boolean label and full 12-visit recurrent logit trajectory across all six admitted roots.

The small source-versus-sham correctness difference is consistent with the null of weak direction-specific mediation under this particular intervention. The composed models' clean development behavior was generally much stronger than either surgically modified trajectory. These results show a need for **instrument redesign** before scaling to an expensive independent-world causal campaign. They neither prove equivalence between source SAE and sham nor rule out a different recurrent mechanism. The direction-matched sham was deliberately strong, and the source donor's limited physical contribution cannot be interpreted as preserving the full clean network state.

### Statistical evidence class and protected cohorts

There are **eight registered source development roots** and **six measured**. Those roots were already selected and heavily inspected on source histories, so this is **development-only mechanism resolution**, not prospective confirmation. FSI-01's one-comparison conservative 95% Hoeffding margin at eight independent roots is approximately **86.54 percentage points**, substantially larger than the prespecified 2-point minimum. Only an independently preregistered causal contrast with a measured matched-sham floor and sufficient independent root power can support a causal mediation confirmation.

NEC-03P's prior source-only cohort successfully recruited **69/88 paired source-eligible independent worlds**, exceeding its preliminary 59-root availability gate. **Zero causal sham samples** from those 69 worlds have been scored. The nonparametric 95th-percentile/95%-confidence sham tolerance prerequisite still requires **at least 59 independently measured causal sham root outcomes** and a defensible conditional-root sampling assumption; it remains unfulfilled.

The **24 reserved context groups per original pilot world**, all development-evaluation and reserved groups on the 88 recruited source-only worlds, **all 32 five-input embargoed roots**, and **all 64 original four-input confirmation worlds** remain unopened. Original NEC03N strict continuous-AdamW replay failure remains recorded.

**FSI-01: REDESIGN_OR_PILOT_ONLY.** Evidence is a negative, power-limited developmental causal-resolution result for the registered SAE intervention/directional sham contrast.
