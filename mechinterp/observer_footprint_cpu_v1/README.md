# SAU-OBS01: observation footprint before mechanistic inference

**Status:** synthetic CPU witness run, three tests passed (2026-11-17). The GPU/real-model protocol below is prospective. The held-out ARG-03-N, LS-SAE and other sealed cohorts remain untouched.

Source: [SauersML/watching](https://github.com/SauersML/watching), especially [results and controls](https://github.com/SauersML/watching/blob/main/README.md#3-controls) and [real-input mechanism protocol](https://github.com/SauersML/watching/blob/main/docs/real-input-mechanism-protocol.md). This implementation is independent; no upstream source code is copied. The original shows that a *read-only* activation hook can alter compiled BF16 computations, providing a numerical input to a trained detector. Its detection varies across GPUs. This motivates a threat model for observer-dependent interpretation, not a consciousness or intentional-deception claim.

## Reproduce the independent CPU witness

From the repository root:

```sh
python mechinterp/observer_footprint_cpu_v1/observer_footprint.py --self-test
python mechinterp/observer_footprint_cpu_v1/observer_footprint.py
python -m unittest discover -s mechinterp/observer_footprint_cpu_v1 -p 'test_*.py' -v
```

A synthetic float32 calculation can either run without an extra materialization or store a BF16-rounded intermediate. There are five arithmetic cases. *Plain* has no materialization; *capture* rounds and retains the site; *barrier* rounds without retaining it; *matched_capture* rounds and retains, after the same barrier; *passive_copy* retains the existing value with no new round. The prefix is fixed across 64 synthetic contexts; later context varies.

Observed on local CPU (64/64 paired cases): capture changes the output vector; the unobserved barrier changes it identically; capture and barrier produce exactly equal outputs; an extra reference read after the common barrier leaves outputs exactly equal; passive_copy equals plain. The site change is -0.0016642212867736816. The six tests across this and the companion [curved-chart geometry witness](https://github.com/MoralityLabAI/Compactification-Theory/tree/research/sauersml-curved-chart-20261117/research/curved_sae_gauge_v1) are deterministic arithmetic checks, not independent model studies. The 64 contexts share one causal prefix and so are not 64 independent observations of the numerical mechanism.

### Interpretation

A downstream detector that receives the numeric site could distinguish *plain* from *capture* and would also flag *barrier without capture*. Under the matched-barrier comparison it receives identical inputs, so capture is information-theoretically undetectable from those inputs. This is a toy counterexample to treating a hook as computationally invisible solely because it returns the original tensor. The `passive_copy` result describes this toy graph; real Torch compilers may change arithmetic even for a copy operation.

## Prospective real-model observer-invariance gate

1. Freeze weights, tokenizer/chat template, adapter, random state, input tensor and masked positions, PyTorch/Inductor build, device, dtype, attention backend, compilation flags, batch shape and probe site. Registration and code hash must precede test observation.
2. Repeatedly evaluate exact same prompt/model in paired conditions: uninstrumented; ordinary activation capture; forced materialization without capture; forced materialization with capture; optional user-buffer capture. Record *all vocabulary logits*, source/boundary activations, attention inputs, later residuals and selected compiler kernels/configurations.
3. **Admission:** compare two ordinary repeats of each condition, require bitwise-equal logits, then prove the two shared-barrier arms have bitwise-identical full logits and trainable-tail inputs, including position IDs, masks and rotary tensors. Without full equality, classify the arm as a changed-runtime condition and record the first observed differing boundary. Do not label it a no-footprint negative control.
4. **Mechanism:** swap the earliest differing already-materialized activation between conditions in both directions. Same-condition copies must reconstruct all logits exactly; the next unmodified run must recover its baseline. Use physical dose/rank matched sham interventions.
5. **Readout:** report paired answer changes, logits in nats, fraction of vocabulary entries differing, state-vector/feature differences, key/value/attention readback and condition accuracy. Freeze detector thresholds on independent calibration only. Compare the original model and the trained detector; run at least two devices/precisions before any device-invariant claim.
6. **Statistics:** cluster by independently sampled prompt/graph family, not token, head, feature or repeated activation. Shared fixed-prefix signatures are one numerical source. Use exact sign/randomization inference or family-level bootstrap with fresh source seeds; separate measurement validity from task-causal claims.

For LoopedTransformers, integrate the gate ahead of LS-SAE native replay, ROC observers and BPD state reads. Compare recurrent steps 1, 3, 6, 12 with separate source-locked diagnostic and held-out cohorts. Continue native task scoring only once observational parity is established, or explicitly treat the observer as an intervention.

Cross-repo connections: [BPD nonlinear-basis boundary](https://github.com/MoralityLabAI/BPD/tree/research/sauersml-linear-basis-boundary-20261117), [Adict audit](https://github.com/MoralityLabAI/Adict/tree/research/sauersml-observer-audit-20261117), and [SlaveBench protocol](https://github.com/patrickdugan/SlaveBench/tree/research/sauersml-probe-invariance-20261117).

**License hygiene:** SauersML/watching has no license in the inspected repository root. This protocol and CPU witness were independently authored; do not vendor the upstream implementation without permission.
