# PEF-05: Registered result and gradient-surgery audit

**Protocol first sealed:** [commit 754aa207](https://github.com/MoralityLabAI/LoopedTransformers/commit/754aa207554340cc8beecd1038c29f6663fdcdfb)
before intervention code, PEF-05 training or evaluation.

**Prospective registered trial:** [GitHub Actions run 43687506414](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43687506414),
artifact name pef05-four-seed-gradient-rescue-sealed-receipts,
artifact ID 14150782670.
Every model/checkpoint SHA256, optimizer state, source split,
gradient projection and source-only score is included in the
archived machine-readable receipts. The hosted run passed all
functional and integrity checks.

## Registered outcome

**Decision: criterion_not_met.** No vulnerable seed (101,127,157)
met the required eight extra heldout oldest-bit predictions plus
four extra jointly correct counterfactual pairs relative to sham.
The seed-109 retention control met its noninferiority bound.

The primary readout is h10 after natural (0,1) continuation,
from the new PEF-05 SHA256 pair-closed partition. Last-two-bit
task labels apply to the original length-eight history.

| Seed | Baseline oldbit /48 | Protect oldbit /48 | Sham oldbit /48 | Baseline both pairs /24 | Protect both pairs /24 | Sham both pairs /24 | Protect task /48 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 101 | 48 | 48 | 46 | 24 | 24 | 22 | 48 |
| 109 | 44 | 42 | 42 | 20 | 18 | 18 | 46 |
| 127 | 44 | 44 | 44 | 20 | 20 | 20 | 47 |
| 157 | 40 | 40 | 40 | 16 | 16 | 16 | 48 |

The unchanged baseline performed better on the newly salted
PEF-05 cohort than on the preceding PEF-04 cohort for several
seeds. Its full-horizon relative decay therefore failed to
reproduce the previous severe collapse in this new protocol.
The source split and h10 continuation differ from PEF-04, so
these results cannot be compared as a controlled seed-only
replication.

All seven-bit suffix-only predictors have exactly 24/48
oldest-bit accuracy and 0/24 jointly correct pairs on the
pair-closed cohort, by the generic MathCan theorem. There are
seven recent bits in the actual input; this source-only
control assumes any function of those seven bits and applies
unchanged to the last-two-bit task label.

## Training-only score and gradient intervention

The intervention uses a unit vector computed at step125 from
the *training-pair mean* old-bit difference at natural h10(0,1).
It removes task-gradient components that have a positive
inner product with the first16 training-pair score gradient
during exactly updates 126..150. Sham removes a random
component with matching gradient-norm reduction.

| Seed | Protect active updates /25 | Sham active updates /25 | Train-only signed score, protect at step150 | Sham step150 | Protect step250 | Sham step250 |
|---|---:|---:|---:|---:|---:|---:|
| 101 | 17 | 15 | 2.4045 | 0.5708 | 1.4259 | 0.7900 |
| 109 | 9 | 12 | 1.9680 | 1.4277 | 2.4650 | 1.5931 |
| 127 | 12 | 8 | 1.5383 | 0.7589 | 0.9900 | 0.9721 |
| 157 | 23 | 21 | 1.8406 | 0.5408 | 0.6401 | 0.4475 |

**Protected gradient integrity:** every active protected update
had measured float64 dot-product of its stored float32 gradient
with the training-only memory gradient at most 1e-5. The
Euclidean norm-matching error was below 1e-5 for every
active sham/protect update. Exact *gradient-space* no-decrease
does not imply nondecrease of the nonlinear memory score under
finite AdamW updates. The code computes the direction in float64
to satisfy the unchanged original sign guard at FP32 storage.

Seed 157 has the clearest mechanism contrast: protection nearly
triples the train-only signed score at step150 relative to sham,
while the primary heldout decoder remains 40/48 in every arm.
At step250 the difference in source margin persists in a weaker
form. This shows that the particular signed-margin objective
can be causally shifted without demonstrating the prespecified
gain in unseen counterfactual-history recovery.

Seed 101 shows a small 2/48 improvement over sham with no gain
over the unchanged baseline, well below the preregistered +8
target. Task-head performance remained high in all trained arms.

## Mechanistic interpretation and next tests

The outcome rules out **this intervention and success threshold**
as a reliable causal rescue in the specified four-seed finite
study. It is consistent with several different mechanisms:

- the chosen train-pair average signed margin may improve while
  some heldout pairs remain ambiguous or reverse orientation;
- task training outside the 25-update window may cause additional
  loss of decodability;
- the trained representation can rotate relative to the fixed
  step125 anchor, so a one-direction score misses other
  independently discriminative axes;
- AdamW's state-dependent preconditioning can change the actual
  parameter displacement relative to the Euclidean projected
  gradient, even when the stored task-gradient dot-product is zero.

The exact mathematical obstructions are documented independently
in [MathCan PEF-05 projection memo](https://github.com/patrickdugan/MathCan/pull/50).
The present data do not identify which mechanism dominates.

A future **separately preregistered** experiment could protect
a multi-direction old-history subspace, test optimizer-aware
preconditioning, and measure pairwise margins on an independent
source-family holdout. The original negative trial is preserved;
posthoc changes to endpoints, seeds or learning rates would not
make it positive.

No pretrained language-model or Lean reasoner performance gain
is claimed. These experiments concern one small tied recurrent
attention block and a fully enumerated finite synthetic input family.

## Replay

The GitHub Actions artifact contains:

- four common step125 SHA256-sealed checkpoints;
- twelve intermediate step150 and twelve final step250 checkpoints,
  each with the complete AdamW optimizer state;
- four per-seed sealed ledgers of training-only score and gradient
  projection receipts;
- independent evaluations on the fresh pair-closed probe and
  heldout cohorts; and
- the exact protocol decision and immutable source manifest.

Run tests with:
    python -m pytest research/pef05/test_pef05.py -q

Run registered trial with:
    python -m research.pef05.pef05 --out /tmp/pef05_campaign


## Independent frozen-checkpoint replay

A separate read-only verifier was committed after the registered trial
and applied to the original hosted artifact from run 43687506414.
[Independent replay run 43688245495](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43688245495)
passed. Its five-file artifact is named
pef05-independent-original-frozen-checkpoint-replay,
artifact ID 14151560936.

The verifier independently reconstructs the salted, pair-closed
train/probe/heldout source allocation. It checks all original
common, step150 and step250 checkpoint SHA256 digests and reloads
the final model tensors. It reconstructs the step125 anchor
from training-only histories, recomputes the signed margins from
saved weights at steps 125/150/250, refits ridge decoders on the
separate probe data, and scores 48 heldout histories again for
each of the twelve frozen final models. It recomputes task accuracy,
oldest-bit paired decode counts and the registered criterion.
Every recovered count agrees with the original sealed receipts.

The independent program also checks all 25 post-split projection
receipts for each modified arm, including the 1e-5 stored
first-order guard and matched Euclidean gradient norm accounting.

It independently reproduced the **0/3 vulnerable-seed rescue**
negative result and the successful seed109 noninferiority control.
It did not train new weights or select a new source split,
checkpoint, intervention window or outcome threshold.
