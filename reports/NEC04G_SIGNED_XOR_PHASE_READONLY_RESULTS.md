# NEC-04G: signed fifth-order XOR phase in archived native Looper logits

**Date:** November 22, 2026. **Evidence:** original six-world calibration-only, retrospective diagnostic.
**Status:** REDESIGN_OR_PILOT_ONLY. The original NEC-04F training result remains negative.

**Archived evidence:** [independent signed-phase audit, run 45036524952](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45036524952), [complete per-context report and independent FWHT artifact 14759911829](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45036524952/artifacts/14759911829). Original trained model and validation data are from the accepted [NEC-04F experiment](https://github.com/MoralityLabAI/LoopedTransformers/pull/312), [run 45015714628](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45015714628).

## Scientific question

NEC-04F's exact five-focal-XOR auxiliary learned to decode intermediate XOR with 90–97% owned validation accuracy in roots12,16,17, yet its original final parity task was below chance. Their native final logits had high unsigned fifth-order Walsh power (54.34%, 48.30%, 30.63%). This audit tests whether that high-order coefficient has the *correct teacher phase* in each original parity-selected context.

For each original complete 32-mask source-owned context, the true parity branch satisfies teacher(mask) = (popcount(mask) mod 2) XOR y0, with y0 the original independently checked class label at mask0. Define chi5(mask)=(-1)^popcount(mask) and m(mask)=native logit1 minus native logit0. The fifth Walsh coefficient is a5=mean[m(mask)*chi5(mask)], and its teacher-aligned sign is a5*(2*y0-1). Positive means the 5th-order parity component supports the correct teacher phase; negative means inversion. The signed normalized coefficient divides by sqrt of the sum of all squared nonconstant Walsh coefficients.

## Independently audited six-root 900-step results

| Historical calibration root | Auxiliary 5-bit XOR accuracy | Native final parity accuracy | Unsigned 5th Walsh power | Teacher-signed normalized 5th coefficient | Phase-inverted contexts | Posthoc truth-phase-oracle accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cal-012 | 95.05% | 42.92% | 54.34% | -0.14909 | 10/15 | 73.54% |
| cal-013 | 50.91% | 50.78% | 0.0534% | +0.00056 | 7/12 | 50.78% |
| cal-014 | 50.91% | 50.22% | 0.0077% | +0.00157 | 7/14 | 50.22% |
| cal-015 | 50.52% | 50.00% | 0.0275% | -0.00059 | 3/6 | 50.00% |
| cal-016 | 90.23% | 39.32% | 48.30% | -0.19687 | 8/12 | 78.91% |
| cal-017 | 97.01% | 44.38% | 30.63% | -0.17412 | 6/10 | 67.81% |
| **Six-root macro** | **72.44%** | **46.27%** | **22.226%** | **-0.08642** | **41/69 contexts** | **61.88%** |

Accuracy, unsigned fifth power and teacher-signed normalized amplitude are equal-root means. The 41/69 inverted contexts are descriptive counts of correlated contexts within these six worlds. The three XOR-competent models have substantially *negative* teacher-aligned fifth-order coefficients. At least 23 original parity contexts have both high fifth-order power (>=10% of nonconstant Walsh power) and an inverted teacher phase: 9 in root12, 8 in root16, 6 in root17.

## Teacher-informed phase oracle: retrospective only

The phase oracle takes the true y0 label from each original owned-validation context and changes only the final-logit fifth-order Walsh coefficient's sign to the correct teacher sign at fixed original magnitude. All other coefficients remain unchanged. It then reclassifies the same 32 masks per context. This is an analyst's **validation-label-informed counterfactual**, not a learned correction, model intervention, deployable gate or evidence of causal information recovery.

Its diagnostic accuracies were root12 73.54% (+30.63pp), root16 78.91% (+39.58pp), root17 67.81% (+23.44pp). All six roots combined, the equal-root macro rises from native **46.27%** to **61.88%** (+15.61pp) under this *posthoc oracle*. The three other roots have negligible oracle gains because they carry almost no fifth-order power. Matched shuffled-auxiliary native validation remains 49.98% after the same oracle. Shuffled mean signed normalized signal is -0.00128 with 0.0085% unsigned fifth-order power. Vanilla is also near chance.

## Independent provenance and code checks

All six original NEC04A CALIBRATION roots12–17, three model arms (XOR-prefix auxiliary, balanced shuffled-label auxiliary, vanilla), and the original 240/900 optimizer-step checkpoints remain in the dataset. Original complete 32-mask owned-validation parity contexts were independently SHA-checked against the archived source/oracle ledger; each fifth-order phase was derived from the original teacher, with an exact XOR-plus-context identity check. All native model and optimizer receipts, original independent checkpoint/logit replay flags and original native task correct counts were rechecked.

The initial analysis uses explicit 32-term Walsh character sums. A second source-independent numerical implementation uses a five-stage fast Walsh-Hadamard transform to check every phase, Fourier coefficient, squared power, original class prediction and teacher-informed oracle prediction. All **three GitHub CI jobs passed** in [run 45036524952](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45036524952), including Python 3.11/3.12 synthetic sign inversion tests. The historical independent audit verified **414 original context-arm-checkpoint records**, representing **13,248 previously saved native logit pairs**.

**New model forwards: 0. Optimizer updates: 0. GPU allocations: 0.** Original reserved contexts, 24 developmental WORLD roots, 16 confirmation WORLD roots, 88 NEC03P recruits and old embargoed records remain untouched. The original NEC-04F STOP_AUX_XOR_STATE_CALIBRATION_PILOT decision remains in force. This analysis carries no prospective uncertainty bound, population claim or successful model-training result.

## Scientific next step

There is a concrete prospective hypothesis: source-trained native 12-pass Looper models can learn focal XOR but may need explicit **context phase integration** to align it with the original MUX-selected teacher. Train a context-phase selector with an independently preregistered equal-budget matched control that shuffles the context phase labels, evaluate the actual final native parity task and alternate branch, and require the new model to infer phase from input context rather than true validation labels. Keep the previous six-root negative denominator and separate untouched root cohort for such a test.

This report diagnoses a signed readout problem in historical source-trained models. It does not establish that phase inversion is the sole failure cause or that a new gate will generalize.
