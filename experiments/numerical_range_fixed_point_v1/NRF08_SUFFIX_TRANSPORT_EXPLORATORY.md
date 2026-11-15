# NRF-08: frozen-midpoint suffix transport, exploratory reconstruction

Date 2026-11-15. Protocol was frozen in `configs/nrf08_suffix_depth_intervention_v1.json` before the first suffix replay. The strict preregistered byte-identity gate **failed**: the six locally regenerated `.pt` SHA-256 values differ from the archived NRF-07 checkpoint hashes. The new `NRF-08X` observations are therefore exploratory.

The reconstructed original-suffix NRF-07 supervised MSE, supervised leakage, supervised J and random median J match the archived NRF-07 numeric metrics within 5×10⁻¹⁵. This is evidence of source-recipe agreement, with original binary identity still unresolved. A separate explicit 128×128 full-Jacobian calculation checks the existing reverse-JVP leakage measurement to 10⁻⁹ at every sampled shadow state.

## Frozen-state causal intervention

For each NRF-07 CPU model trained at T=16 or T=32 with seed 109/113/127 and final exposure 4096, compute the representation h_(T/2), train the rank-8 supervised decoder and eight paired random controls on fold 1, and retain each projection and ridge predictor. Use fold 5 for task MSE and fold 6 for 12 shadow Jacobian inputs. The **only** within-checkpoint intervention is whether the same fixed h_(T/2) receives an 8-visit or 16-visit Jacobian suffix. No additional model training occurs between those two measurements.

| Trained depth | Suffix | Supervised wins | Mean margin, J_sup − median J_rand | Mean supervised leakage |
|---|---:|---:|---:|---:|
| 16 | 8 | 3/3 | -0.164776 | 0.417087 |
| 16 | 16 | 3/3 | -0.119180 | 0.749538 |
| 32 | 8 | 3/3 | -0.035084 | 0.522412 |
| 32 | 16 | 0/3 | 0.236748 | 1.005469 |

The observed sign change for all three T32 model reconstructions occurs with unchanged task errors and unchanged projections. The finite λ crossover moves from 1.218–1.645 at suffix 8 to 0.294–0.348 at suffix 16. Lambda=1 was previously selected, so this is a mechanism diagnostic rather than a fresh confirmatory selection.

## Full-tangent decomposition

For each trained model, shadow midpoint and supervised rank-8 P, explicitly multiply the first eight tangent steps J and next eight steps S. With Q=I-P,

`P S J Q = (P S P)(P J Q) + (P S Q)(Q J Q).`

The squared Frobenius norm is the sum of the advected and fresh-injection energies plus their inner-product interference. The cross term is positive in **72/72** source-compatible reconstructed shadow states. Its contribution is approximately 45.5–46.2% of mean final squared leakage across the six checkpoints, so constructive transport, rather than a pure scalar power law, describes this local phenomenon. See MathCan PR #46 `SUFFIX_DARK_TRANSPORT.md` for the exact theorem, norm bounds and rotation counterexample.

## Reproduction and limitations

- Frozen protocol: `configs/nrf08_suffix_depth_intervention_v1.json`.
- Native decoder suffix runner and strict gating: `research_gym/analysis/nrf08_frozen_suffix_intervention.py`. Default stops before evaluation on SHA mismatch; `--exploratory` gives a separately labeled result.
- Independent full-Jacobian oracle: `research_gym/analysis/nrf08_suffix_transport.py`.
- Compact, exact-value rows with all six checkpoint hashes, λ thresholds and transport components: `nrf08_exploratory_summary.json`.
- Full eleven-arm per-cell JSON, all 72 per-state tangent decompositions, source-compatible standalone rerunner and regenerated checkpoints are packaged in the conversation's downloadable reproducibility ZIP.
- Local CPU PyTorch 2.10 tests passed; the private repository's Actions are known to fail before job runner setup.

The study does not establish byte-identical original checkpoints, full-depth semantic task utility, an architecture-invariant causal effect, or an increasing-leakage theorem for every recurrent map.
