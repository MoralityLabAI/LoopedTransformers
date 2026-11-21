# RMP-INK-PC01: first cross-audited CPU development run

Date: 2026-11-21. Status: **synthetic trained-gate positive control, development-only**.

Evidence: [GitHub Actions #44900273337](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44900273337),
artifact [inkling-trained-gate-sourceonly-dev #14706051354](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44900273337/artifacts/14706051354).

Both Python 3.10 and 3.12 unittest suites passed. The Python 3.12 job
retrained the one-parameter gate on 48 independent TRAIN roots,
then scored **64 separately generated development source roots** and
all matched intervention arms. The public graph transition P and
requested hop budget h are given explicitly to the model. A single
scalar mixing gate alpha is learned by analytic cross-entropy gradients,
rather than learning graph structure from language tokens.

## Original experimental records and independent checks

| Measurement | Observed |
| --- | ---: |
| Learned alpha = sigmoid(theta) | 0.9924544921475802 |
| TRAIN roots / lengths / epochs | 48 / 1–4 hops / 70 |
| DEV roots / tasks | 64 / 512 (8 hop lengths per root) |
| Paired depth records | 3,072 |
| Matched intervention records | 1,600 |
| Targeted state-patch visit and L2 dose | 2 / 0.85 |
| Checkpoint digest | 75935f9314f4ca1a49da121a353c5f87b5497ae4697de2c9787aefaf14feb802 |
| BPD independent full recurrence/raw-state replay | PASS |
| EvalMath root-paired development audit | PASS |

### Inference visits versus exact answer

| Maximum permitted passes | Correct / 512 | Verified accuracy |
| ---: | ---: | ---: |
| 1 | 64 | 12.5% |
| 2 | 128 | 25.0% |
| 3 | 192 | 37.5% |
| 5 | 320 | 62.5% |
| 7 | 448 | 87.5% |
| 10 | 512 | 100% |

Given the correct explicit graph operator and hop budget, this
constructed recurrence advances along the permutation one step per
pass. The result validates the visit counter and exact task verifier.
It cannot establish that a transformer learned an algorithm, nor
that its latent state contains interpretable cross-loop routing tags.
Longer permitted passes after the requested hop count are explicitly
frozen by the model's input-specified stop rule.

### Matched loop-boundary interventions

Each of 320 tasks with hop length 4–8 was evaluated with 5 arms at
the same checkpoint. The target direction was deliberately chosen
from the current argmax state coordinate. Dose 0.85 was selected on
an earlier synthetic development pilot; the following comparison
is exposed to that development choice.

| Intervention | Correct / 320 | Notes |
| --- | ---: | --- |
| Clean | 320 | Reference |
| Zero perturbation | 320 | Negative control |
| Active-coordinate targeted direction | 0 | Mechanically shifts the known active state |
| Gaussian random direction, same L2 | 320 | Seeded within root and task |
| Random-coordinate shift, same L2 | 297 | Seeded within root and task |

The root-macro paired active-minus-random-direction effect is -1.0
on these DEV roots, and active-minus-random-coordinate is
-0.928125. The large selected effect is a **constructed active-state
causal positive control**. It depends on the prescribed active-index
selection rule and hand-supplied transition matrix. The random control
selection distribution is a different policy, so this result is not
evidence of a discovered pretrained mechanism.

## Independent audit

The BPD auditor source was fetched from pinned commit
5cbd5de63e95b6c70c3e351f68be6676cd830268. It reconstructed
the target graph walks, repeated the full learned recurrence from
the saved scalar theta, checked each pre/post patch raw state, verified
L2 dose, and replayed every patched final answer. Its result is stored
in the Actions artifact as bpd_independent_audit.json with
receipt_replay_passed=true, 64 roots and 320 intervention cases.

EvalMath's development-only depth auditor was fetched from pinned
commit ff32b93a311842967da1e4e33ac08e4ac576c46e. It checked
all 3,072 paired depth rows against root, model and cache mode
identities and exported fsi_dev_manifest.json. The 1-vs-3 paired
case/root macro effect is +0.25. Its FSI manifest retains
sealed_readout=false, hypothesis_preregistered=false and
clusters_independent=false, preventing mistaken confirmation claims.
No measured token generation, inference joules or device-memory use
exists for this synthetic one-parameter model.

## Why the next stage remains open

This study provides a reliable positive control for the collection
and BPD intervention machinery. Scientific follow-up requires a
trained neural model with input-to-transition interpretation that
achieves a preregistered independent task threshold, a named real
loop-boundary activation interface, and interventions against
representational hypotheses selected before fresh confirmation.
Huginn's archived experiments show modest multi-hop competence;
the Inkling variable-loop checkpoint remains a better large-model
target once available on a suitable host.

Nothing here changes the existing LSAD, RMP, LOOPSCOPE or sealed
BPD/EvalMath scientific verdicts.
