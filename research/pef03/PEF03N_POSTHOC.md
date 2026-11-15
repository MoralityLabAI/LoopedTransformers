# PEF-03N | Posthoc frozen-head nullspace audit

Selected after original registered PEF-03 and paired PEF-03P outcomes.
This is exploratory. The original training, checkpoints, protocol
and scored results are unchanged.

## Immutable provenance

Frozen reference [PEF-03 hosted run 43680455308](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43680455308).
The independently successful [PEF-03N pinned replay 43681530296](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43681530296)
loaded the original model artifact without retraining.

- Seed 17 checkpoint SHA256:
  c174fbfd8bef4fa026806e62e66ad48a38827935c1121a487514de9948f61432
- Seed 29 checkpoint SHA256:
  76dfcf7c401a8d60ae4570cf281b0aa788362bb76bde7f0c040438333a5a6e01

Both checkpoints have 16D memory and a rank-four 4x16 affine task head.
FP64 SVD identifies a 12D head nullspace. The largest final task-logit
change after replacing that nullspace with its oldest-bit-flipped
counterfactual is 3.82e-6 in FP32. Algebraically W N = 0 and a final
nullspace translation leaves W h + bias unchanged.

## Replayed heldout results

The probes are fitted only on the original 56 probe-training histories,
then scored on the original 52 heldout histories and 52 counterfactuals
obtained by flipping only the oldest bit.

| Feature space | Seed17 original | Seed17 paired both | Seed29 original | Seed29 paired both |
| --- | ---: | ---: | ---: | ---: |
| Full 16D hidden state | 40/52 | 29/52 | 52/52 | 52/52 |
| Task-head rowspace (4D) | 29/52 | 15/52 | 38/52 | 24/52 |
| Task-head nullspace (12D) | 48/52 | 46/52 | 52/52 | 52/52 |

Paired-both-correct means the *same fitted probe* correctly recovers
the original oldest bit and its flipped value from the corresponding
frozen hidden states.

A descriptive posthoc 64-sample dimension-matched control fits the
same ridge procedure to 64 randomly oriented 12D subspaces. Their
median paired successes are 27/52 for seed17 and 52/52 for seed29.
Only 1/64 random subspaces matches the seed17 nullspace performance,
while 54/64 match seed29. These counts are not p-values and are not
independent trained-model replicates.

Different head-space dimensions receive feature-wise standardization
with the same ridge penalty. Conditioning and regularization can
affect the comparison. The full old-bit signal can live in several
subspaces. The precise conclusion is that a highly decodable history
signal survives in a subspace *invisible to the final task head*.

## Scientific boundary

The final linear-head nullspace can become visible after more recurrent
layers. PEF-03F measures this explicitly through internal nullspace
patches and intermediate-state Jacobians. Such hybrid-state patches
may lie off the model's natural state manifold.

The pure companion MathCan/OrderedTangent.lean proves generic additive
kernel invariance and finite shear reactivation witnesses.

The pinned workflow .github/workflows/pef03n-pinned-nullspace.yml
requires the original archived checkpoint SHA256s. Trained weights
are not rewritten and no new training is performed by this workflow.
