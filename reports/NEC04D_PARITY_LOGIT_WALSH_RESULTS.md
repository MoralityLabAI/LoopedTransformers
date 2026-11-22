# NEC-04D: native Looper parity signal is tiny in final-logit Walsh spectra

**Completed 22 November 2026.** Evidence class: read-only, exploratory posthoc output-signal analysis of independently replayed NEC-04C native Looper checkpoints on the **six original NEC04A calibration WORLD roots 6–11**. Research task: [issue #300](https://github.com/MoralityLabAI/LoopedTransformers/issues/300). Earlier native-model training and negative primary result: [NEC-04C report](https://github.com/MoralityLabAI/LoopedTransformers/blob/main/reports/NEC04C_PARITY_CURRICULUM_RESULTS.md), [merged PR #298](https://github.com/MoralityLabAI/LoopedTransformers/pull/298), successful fixed-code training [run 44993859839](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44993859839).

## Scope and verification

[**NEC-04D hosted run 44994677211**](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44994677211) passed **7/7 mathematical/corruption tests**, independently rederived the original complete 12-bit input histories and composed teacher labels for 2,560 source plus 768 owned validation cases in each of six calibration roots, checked six previous independent model-replay receipts, and verified original **24/24 native optimizer checkpoint-file SHA256 seals**. It re-evaluated **the already saved logits** from the two original model arms (`vanilla`, `parity_curriculum`) and fixed 240/900 optimizer checkpoints, with no new model inference or training. The source and model-code hashes were checked against the original NEC04C pretraining ledger from Git commit `b6e7a9a2e2b530e738967aa15d37c265643e98b4`.

**Native spectral artifact:** `nec04d-original-native-final-logit-parity-walsh-archive-audit`, artifact **14743512024**, ZIP SHA256 `27e97ad13bcb67279bc78d8676ba884a8c86ae5ce627733ff441b11079893929`. The archived JSON preserves **all individual original source and owned-validation parity-context measurements**, every root/checkpoint/arm, original checkpoint hashes and descriptive six-root macro summaries.

The analysis uses only MUX-selector=1 contexts. Every included context contains all original **32 focal masks**. The native two-class logit margin is m(x)=logit_1(x)−logit_0(x). The Walsh/Fourier coefficient of focal subset S is

```
Walsh(S) = (1/32) Σ_{x∈{0,1}⁵} m(x) · (−1)^{popcount(x & S)}.
```

For the five-bit parity branch S=31, the original teacher target sign `2*label−1` aligns the fifth-order coefficient with each context's fixed XOR offset. Independently recomputed label-aligned margin and direct full-five-focal Walsh amplitude agreed, and Parseval's identity reconstituted the original centered focal-mask margin variance. The **fifth-order power share** is `Walsh(31)^2 / sum_{S != 0} Walsh(S)^2`, a nonnegative fraction of **nonconstant final-logit focal variability within one context**. We average these context fractions within each original world root, then average the **six world roots equally**. This average of ratios is different from a ratio of pooled amplitudes.

## 900-update results on original owned calibration-world validation contexts

| Root | Vanilla parity accuracy | Curriculum parity accuracy | Vanilla fifth-order power share | Curriculum fifth-order power share |
|---|---:|---:|---:|---:|
| cal-006 | 50.00% | 49.55% | 0.00001% | 0.03648% |
| cal-007 | 50.00% | 49.72% | 0.00104% | 0.03220% |
| cal-008 | 50.57% | 51.70% | 0.04766% | 0.14316% |
| cal-009 | 49.48% | 50.00% | 0.00055% | 0.00000% |
| cal-010 | 50.00% | 48.61% | 0.00000% | 0.11881% |
| cal-011 | 50.00% | 50.00% | 0.00000% | 0.02972% |
| **Root-macro** | **50.0079%** | **49.9309%** | **0.00821%** | **0.06006%** |

The historical class-level parity correct counts remain **1,072/2,144 vanilla** and **1,071/2,144 curriculum** at 900 steps. These raw case totals are provided as descriptive provenance; the independent-world unit is a root, n=6.

The root-macro **oracle-aligned parity margin normalized to each original context's total logit margin RMS** was **−0.0000423** for vanilla and **+0.0028235** for curriculum, both near zero. Across complete parity contexts, the root-macro fraction of original focal masks assigned the locally dominant predicted class was **92.44% vanilla** and **82.88% curriculum**. This concentration means the model often settles on a locally preferred class across most focal masks, even while the exact parity target has precisely 50% positive/negative labels.

At the first, 240-update checkpoint, the vanilla and parity-curriculum root-macro fifth-order shares were **0.00552%** and **0.00051%**, respectively. The 240-step curriculum output was also nearly uniform in its predicted class over individual parity contexts (**97.21% majority-prediction fraction**), with 49.29% parity accuracy. At 900 steps, the small fifth-order fraction was observed on both **original source** and **owned validation** context panels: vanilla **0.00256% source** vs **0.00821% validation**; curriculum **0.06473% source** vs **0.06006% validation**.

The unit tests confirm that a perfect five-bit XOR logit teacher has exactly **100% of nonconstant Walsh energy at degree five**, while a strictly lower-degree signal has zero fifth-order energy, and a constant output has undefined centered spectral power (recorded as null/zero variance). No zero-variance contexts appeared in the original scored native snapshots.

## Interpretation and next research target

The NEC04C curriculum changed the magnitude and distribution of final-logit variation over focal masks, with a small increase in the fifth-order Fourier-power share. **Almost all measured nonconstant output variation remained outside the true five-bit parity component**, and the source/validation class accuracy stayed at chance. The displayed spectra describe the **native final classification head**; they cannot determine whether a recurrent hidden state internally represents XOR parity, nor whether a larger network, constructive supervision, alternate optimizer or more updates could learn it.

This supports a distinct **capacity-versus-optimization diagnostic** before attempting neuron-community necessity analysis: freeze a fresh *calibration-only* cohort of original roots **12–17**, test explicit two-/three-/four-/five-bit compositional XOR targets and recurrent auxiliary-state supervision with matched initial weights and compute, and report both output Fourier signal and whole-task correctness. Any such protocol must be prospectively locked before training and must preserve development/confirmation WORLD roots. The current work contains no new native-model training, source SAE, causal sham or mechanistic necessity scores.

The native NEC04B headroom-band gate remains failed and NEC04C's registered parity-gain stop decision remains `STOP_PARITY_CURRICULUM_CALIBRATION_PILOT`. Every original calibration-root reserved context, distinct 24 development WORLD roots, 16 confirmation WORLD roots, old NEC03P recruited-world causal outputs and the previous embargo/confirmation cohorts remain untouched. **FSI-01: `REDESIGN_OR_PILOT_ONLY`.**
