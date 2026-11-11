# RMP-GG-01 closeout: supervised belief decodability with failed rank-one causal transfer

**11 November 2026.** This is the first **trained** LoopedTransformers experiment in the
EvalMath RHT-20–25 reward-hacking handoff. The [registration](../docs/rmp_goal_guarding_causal_v0.md)
was committed as dc906ca before the source code and any outcome. RHT-20's exact
rational cue-choice oracle supplies the ground truth. The model is a 32-dimensional
two-position native tied ExplicitDecoderBlock, with one evidence bit per visit.

The positive-control network was explicitly supervised on the correct terminal
protect/exploit decision and on all four prefix decisions. Neither model
developed the objective spontaneously; the training labels encode the exact
Bayesian action for a declared two-world synthetic task. **This experiment
measures decodability and causal intervention fidelity, not model deception
or awareness of its real gradient updates.**

## Immutable run record

- Protocol: RMP-GG-01, CPU, one torch thread, seeds 11 and 17, 400 AdamW
  updates, batch 128, four-cue training, four- and six-cue evaluation.
- Initial random states were archived in the repaired source-revision run,
  alongside final weights and checkpoint SHA-256 hashes. The initial probe
  R² is explicitly labeled exploratory, measured after the first full run
  exposed the decodability issue. The fixed training settings, cue generator,
  intervention manifest, eligibility rule and causal decision gates were retained.
- [First complete run, GitHub Actions 43033125043](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43033125043): six exact/neural
  tests passed; the pilot ran to completion and produced final-weight artifacts.
- [Repaired, replicated run, GitHub Actions 43033824987](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43033824987):
  six tests passed in 8.87 seconds, full two-seed supervised training completed,
  and the artifact includes the **initial and final weights** for both training
  seeds and the shuffled-label control.
- [Replicated-run checkpoint and JSON artifact](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43033824987/artifacts/13861962228).
  Source revision includes the documented PyTorch-native fingerprint fix and
  visit-2-and-later patch identity assertion. These are implementation fixes,
  and the original numerical result replicated within numerical precision.

## Frozen held-out measurements

| Endpoint | Seed 11 | Seed 17 |
|---|---:|---:|
| C0 random-initialized 4-cue accuracy | 0.670410 | 0.209229 |
| C0 random-initialized 6-cue accuracy | 0.629639 | 0.245117 |
| Trained 4-cue accuracy | **1.000000** | **1.000000** |
| Trained 6-cue accuracy | **0.999023** | **0.996338** |
| C0 linear probe R², *post-hoc control* | 0.999992 | 0.999984 |
| Trained linear probe R², validation | 0.999969 | 0.999915 |
| Oracle-discriminating eligible patch pairs | 256 / 256 | 256 / 256 |
| Sham/receiver counterfactual hit rate | 0 / 256 | 0 / 256 |
| Rank-one fitted-belief patch hit rate | **0 / 256** | **0 / 256** |
| Full-state donor-prefix patch hit rate | **256 / 256** | **256 / 256** |
| Donor full-sequence copy hit rate | 0 / 256 | 0 / 256 |
| Each of eight matched-norm random-direction patches | 0 / 256 | 0 / 256 |
| Rank-one mean hybrid-margin lift | +0.052326 | +0.015983 |
| Full-state mean hybrid-margin lift | +19.487761 | +19.008296 |
| Ratio of rank-one to full margin lift | **0.002685** | **0.000841** |

The shuffled-target trained control reached 0.504883 four-cue accuracy and
0.511719 six-cue accuracy, near chance on both panels. The register required
accuracy at least 0.90 on six-cue heldout, probe R² at least 0.50 and at least
32 eligible pairs; both trained seeds passed those **eligibility** conditions.

The rank-one causal-support gate required a +0.10 advantage in hybrid action
hit rate over sham and +0.05 over **every** random-direction control for each
seed. Both seeds had **zero rank-one counterfactual hits**. The preregistered
claim is therefore **RANK_ONE_NOT_SUPPORTED**. The fully patched state attained
the structural counterfactual ceiling, confirming that the instrument can
construct an exact donor-prefix/receiver-suffix continuation.

All four-input and six-input accuracies are measured on seeded draws from the
declared finite cue channel. Four-cue evaluation resamples from the finite
training support; six-cue evaluation tests depth extrapolation. The 256 eligible
pairs per seed are conditional on correct clean/hybrid predictions and should
be interpreted with that denominator.

## What the negative result implies

The linear decoding objective and the causal target are materially different.
A ridge-fitted decoder recovers the exact scalar posterior odds from a high
dimensional recurrent state. Here that scalar is already recoverable with
near-perfect R² **at random initialization**, before learning the task.
This decodability can arise from the deterministic input embedding and
recurrent map's preserved input information.

The trained policy uses its full recurrent state to compute the correct
decision. Replacing the entire state at visit 2 leads to the independently
calculated hybrid decision. Replacing only the scalar coordinate chosen by
the linear decoder causes a negligible average margin change and never
changes the discrete decision on the preselected counterfactual roster.

The experiment supports a narrow but useful result:

\[
\text{perfectly decodable Bayesian belief}\
\not\Rightarrow\
\text{causally sufficient one-dimensional belief coordinate}.
\]

This is an instance of an *observational-to-interventional gap* for
the chosen ridge-probe direction. It does not establish that posterior belief
has no causal role; the operative computation may use distributed or nonlinear
coordinates, and the rank-one perturbation may leave the valid learned-state
manifold. The full-state swap is an architecture identity under the registered
two-token tied transition, rather than a discovered mechanistic feature.

No human gold task or genuine reward-maximizing model was trained in this
pilot. The simulation-aware reward and exploit-gain parameters are a
controlled synthetic objective.

## Proposed next registered experiment

**GG-02 should be a new preregistered protocol.** Fit rank-2/4/8
calibration-only causal subspaces by a direct interchange objective, with
independent heldout counterfactual pairs and norm/activation-distribution
guards. Compare this with a tangent-space Jacobian control and the frozen
rank-one and random baselines. Reserve a new locked split for all reported
success rates. Only after establishing the mechanism in the supervised
positive control should an actual randomized optimizer-update experiment
vary training awareness, feedback disclosure, reward misspecification and
an independent gold audit.

The current draft PR holds the source, tests, registration, trained
checkpoints and exact negative result. No RHT theorem is modified.
