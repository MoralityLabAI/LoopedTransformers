# NEC-03N-B: Frozen 900-step checkpoint with reset AdamW source continuation

**Registered November 21, 2026 before any checkpoint-anchored continuation outcomes.** This is a new in-sample learning calibration on the four previously trained five-essential-input NEC-03M roots `dev5-000` through `dev5-003`. It is a distinct optimizer experiment from the [strict NEC-03N continuous-optimizer protocol](https://github.com/MoralityLabAI/LoopedTransformers/pull/257), whose full CPU run failed its preregistered exact step-900 provenance requirement on two roots.

## Why the original 1800-step experiment failed

The [original strict NEC-03N run 44895917012](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44895917012) passed its locked-source suite and exact 900-to-1800 state ancestry for `dev5-000` and `dev5-001`. The `dev5-002` and `dev5-003` jobs failed at step 900 on the preregistered minibatch-loss comparison, before recording their new 1800-step result.

A separate **source-only diagnostic** [run 44897647242](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44897647242) replayed only the initial 900 updates of those two failed roots, and compared every frozen old checkpoint tensor, original loss and in-sample source score:

- `dev5-002` matched **both** old native models bit-for-bit in this independent diagnostic, including their exact original step-900 loss and every original source prediction.
- `dev5-003` started from the original exact native initialization digests but diverged during optimization. Its composed step-900 source score was **1920/2560**, compared with the original **1567/2560**; maximum absolute parameter difference **0.572**. The additive model stayed at **2560/2560** but its final parameter values also differed, with maximum absolute difference **0.337**.

These outputs show that the registered exact-replay condition cannot be claimed satisfied across all four original hosted CPU root jobs. The discrepancy may arise from native CPU floating-point execution or another stochastic/numerical difference; its origin is not fully isolated. The negative strict protocol remains in history. Any 1800-step result that starts with a different intermediate checkpoint or reconstructed optimizer moments would represent a different scientific training condition.

## This newly frozen checkpoint-anchored recipe

Each root and each of its two model arms, composed and independent class-frequency-matched additive:

1. Restore the exact original NEC-03M **900-step** checkpoint from successful GitHub Actions run **44894980345**, checking the SHA256 file digest and model-weight digest against its source-trained receipt. Instantiate the same PEF-07 hidden32, twelve tied-visit native Looper, then load the frozen original weights with strict tensor key coverage. **There is no replay from initial weights.**
2. Score all 2,560 original source-training input histories with full 32-focal-state grouping and verify **the exact original source accuracy** before any new training.
3. Create a **new AdamW optimizer** with empty state, learning rate **0.003**, zero weight decay, batch **128**, native gradient clip norm **1.0**. This explicitly **resets optimizer moments at step 900**, so it does not represent the original uninterrupted optimizer trajectory.
4. Use a separate deterministic source-index generator initialized to `(original model seed)*7919+59`. Advance it by exactly **900 historical minibatch draws**, then use the next **900** fixed minibatches for warm-start updates. The first and last continuation minibatch index SHA256 fingerprints are saved for independent audit.
5. Freeze the new model after exactly **900 additional updates**, totaling 1,800 native parameter updates across two distinct optimizer stages. Save all new checkpoint bytes, weight SHA256, seed and original frozen-parent identity. Score **only** the same 2,560 old training-source inputs. Compute both class recalls, confusion matrix and full source predictions. All original failed and successful roots remain in the denominator.

All four original teacher/checkpoint worlds are included independent of outcome. The fixed **90% in-sample source accuracy floor** applies to both original and new models. Any model that worsens or remains incompetent is reported. The original NEC-03M source SAEs, their frozen source feature ranks and source-only directional nulls are **not** reused as a new causal instrument on altered weights; this condition trains **zero** new SAE models and scores no interventions.

## Audit contract and statistical boundary

The hosted public CPU campaign first verifies this exact registered protocol's Git source blobs, original five-input source registry SHA256 **6777d478f6e988e3f57f8882dfef2e84898f765b7adf7e142a914b4cbdcd1217**, and original four-input 64-confirmation spec commitment **670ea6e1bb4fe86ab9691d0af4800d5623fe64e489ca7555561a80c9739f1f39**.

An independent checkpoint/source auditor reloads each immutable source900 model and its new reset-AdamW 1800 checkpoint, recomputes every original Boolean source truth label and model prediction on all 2,560 source histories, verifies source checkpoint byte SHA256 and weight digests and the intended fixed source minibatch index generator offset. Its four-root aggregation includes every model, with explicit field `exact_continuous_1800_claim=false`. It also retains the original strict NEC-03N failed run as an immutable provenance dependency.

**All 24 five-input development-evaluation context groups per root, all 24 reserved groups, all 32 five-input embargoed worlds, and all original 64 four-input confirmation roots remain untrained or unscored.** The four original five-input worlds have already been extensively used for source-only pilot selection. This is **in-sample development engineering**; it supplies no independent-world significance or transfer claim. FSI-01 remains `REDESIGN_OR_PILOT_ONLY`.

A later prospective five-input model/SAE pilot must separately freeze the fixed optimizer schedule and dictionary training under a new root registration before using new heldout development contexts.
