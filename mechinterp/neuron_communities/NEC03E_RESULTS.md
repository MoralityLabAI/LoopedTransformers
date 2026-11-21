# NEC-03E: attenuation-only native/SAE matched physical-dose results

**Date:** November 21, 2026. **Evidence class:** posthoc development on PEF-07's previously opened source family. This follows the negative-dose-feasibility [NEC-03D report](NEC03D_RESULTS.md) and source-locked [NEC-02B results](NEC02B_RESULTS.md).

## Hosted execution and chain of custody

- **[Successful public CPU run 44860683140](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44860683140)**. Six unit tests passed; both task arms completed the three pairwise interventions and the separate independent 16-state outcome/physical-dose audit.
- Input model and source corpus: the unmodified six-model PEF-07 checkpoint/causal archive from **NEC-02B run 44790504320** (artifact 14655419843).
- Input SAE: the two immutable source-trained 128-feature SAE checkpoints and source-only feature rosters from **NEC-03D run 44860214701** (artifact 14689443261). No SAE refitting or native model training was performed.
- Output GitHub artifact **`nec03e-attenuation-pairs-matched-physical-dose`**, ID **14689722881**, ZIP SHA256 **feddf4c1d0677c66be55e611f106a24e404c9d8c4c2f98f3e3e4a5e6f3a269bf**.
- The archive preserves both `pairwise_report.json` files, 1,296 candidate × pair × 48-context records containing clean/hubs/left/right 16-state logits, all per-site source target norms, achieved direct `mlp_down` output norms and attenuation coefficients, plus `nec03e_audit.json`.

## Numerical matching and retained signal

All **27 candidate-pair comparisons** (4 coalition + 5 threshold communities, with 3 donor pairs each) were within the declared **1% direct physical-dose matching tolerance on 100% of per-visit/example samples**. Exact per-source dose auditing succeeded.

Only **8/27** matched comparisons also met the source-available signal gate: at least 80% of site/examples must have nonzero common dose, and the median pair must retain >=25% of its original left-donor dose. All eight qualifying comparisons were between a native community and its source-frequency-matched, hub-disjoint native-neuron sham. The remaining four native/SAE and SAE/alternate SAE comparisons in the coalition task, and the ten corresponding threshold comparisons, fell below the signal-availability threshold.

| Task | Candidate-pair family | Numerically dose matched | Also retaining sufficient physical signal |
|---|---|---:|---:|
| Coalition | Native vs native sham | 4/4 | 3/4 |
| Threshold | Native vs native sham | 5/5 | 5/5 |
| Coalition | Native vs SAE | 4/4 | 0/4 |
| Threshold | Native vs SAE | 5/5 | 0/5 |
| Coalition | SAE vs alternate SAE | 4/4 | 0/4 |
| Threshold | SAE vs alternate SAE | 5/5 | 0/5 |
| **Total** | | **27/27** | **8/27** |

This result demonstrates that the local output-norm equality mechanism works under attenuation-only dosing. It also establishes that **fixed-cardinality SAE donors are too intermittently active to yield a sufficiently strong common physical intervention** in this particular small-model feature-selection setup.

## Selected conditional outcomes

The strongest coalition conditional-recovery candidate is `nec-68cd70ab69a8`, with hubs-only baseline **10/768 errors**:

| Pair | Left donor errors | Right donor errors | Left-minus-right correct predictions | Nonzero shared-dose fraction | Median left dose retained |
|---|---:|---:|---:|---:|---:|
| Native vs native sham | 1 | 2 | +1 | 100% | 95.9% |
| Native vs SAE | 4 | 5 | +1 | 60.0% | 27.6% |
| SAE vs SAE alternate | 10 | 10 | 0 | 23.6% | 0% |

The strongest threshold conditional-recovery candidate is `nec-b30c4edbbb42`, with hubs-only **59/768 errors**:

| Pair | Left donor errors | Right donor errors | Left-minus-right correct predictions | Nonzero shared-dose fraction | Median left dose retained |
|---|---:|---:|---:|---:|---:|
| Native vs native sham | 58 | 53 | -5 | 100% | 79.3% |
| Native vs SAE | 42 | 37 | -5 | 74.3% | 30.7% |
| SAE vs SAE alternate | 53 | 58 | +5 | 35.7% | 0% |

The threshold candidate `nec-0cf33bd0b573` is another useful negative specificity example: under matched native-versus-native-sham local dose, hubs-only caused **5 errors**, native restoration still **5**, and matched sham restoration **0**. All zero-effect candidate groups and their exact raw logits are in the artifact.

## Mechanistic implications and power

At adequate shared local signal, the native shams can perform as well as, or better than, the discovered native communities. This weakens a strong necessity or unique-native-reader interpretation. With native-vs-SAE donors, matching can make both interventions small or zero for many input/visit pairs. The resulting low behavioral dose prevents a test of which learned basis better rescues the circuit.

The SAE autoencoders fitted in NEC-03D have normalized reconstruction MSE around 0.007 yet activate ~63 of 128 features on average. Even an autoencoder with decent aggregate reconstruction can have a source-ranked individual feature that is absent on many contexts. A more useful next instrument would select **active** features through a source-frozen, label-blind per-input policy and evaluate true TopK SAE training with controlled L0, preserving matched direct physical dose and source-disjoint training. This new feature-selection policy must be frozen *before* its own new evaluation sample; using the already viewed PEF heldout still produces only developmental evidence.

**EvalMath FSI-01:** `REDESIGN_OR_PILOT_ONLY`. The model/task roots and 48 heldout source contexts were already exposed. Source-selected candidates were evaluated repeatedly; the highlighted stronger rescues were selected after seeing outcomes. No multiplicity-adjusted confirmation, causal mechanism uniqueness, language-model transfer or RSI/self-model inference is established by this study.
