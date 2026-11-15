# AF-08 numerical repair record

The frozen AF-08 protocol SHA-256 is 6aad992f2172ff6529b341ac071bb3308e3c4e40c5149e12d20c49dca7778c62. Qualified AF-06 source checkpoints (seeds 17 and 29), the Horn oracle, margins, coefficients, training budgets and scientific gates remain fixed.

## Invalid first training execution

[CI run 43680485938](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43680485938) used commit bbfd96b70208f1d2ab032882ee3e8a3664ee9338. The noncollapse training arm became nonfinite for both seeds, while the status and locality comparators stayed finite and development-qualified. Both independent evaluation jobs failed the named-NumPy-array versus Torch-checkpoint check: NaN values are not equal under numpy.array_equal. The original aggregate was skipped. These are engineering failures, and the affected neural results receive zero scientific credit.

The first version differentiated its Euclidean margin with sqrt(sum(square)). At a coincident code, the derivative is undefined and can produce NaN gradients.

## Repaired source

[Commit 44bc1dc](https://github.com/MoralityLabAI/LoopedTransformers/commit/44bc1dc76ba7bc087f5c3da8f8ce5b4e668f0593) retains the original Euclidean forward distance using torch.linalg.vector_norm, with defined zero subgradient at the singularity. Nonfinite losses, gradients or model weights now halt training immediately. A regression differentiates both noncollapse losses through exact zero-distance cases. Ten regular and optimized local tests pass. A separate 50-update training-only check of seed 17 preserved perfect development goal and exact joint-Q accuracy across the three continuations with finite losses.

[Repaired full CI run 43681114942](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43681114942) is independently source-pinned. The first run is retained as evidence of the engineering defect.

## Original qualified-source mechanism diagnostic

On the original AF-06 train/dev inputs, the already-qualified Q codes had class-centroid distances around 1.9 to 2.8, above the registered prototype margin 0.45. Sampled changed-status donor patches had carried-code distances above 1.88, exceeding the separate 0.35 margin. Thus both noncollapse terms initially had zero forward penalty and zero gradient on those source examples. The final scientific report must account for this when interpreting gains or null results. No held-out inputs were used to modify the original registered objectives.

AF-01 through AF-07 and the parent AF-06 source weights remain unchanged.
