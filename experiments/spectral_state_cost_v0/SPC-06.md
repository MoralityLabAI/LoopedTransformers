# SPC-06: Necessary phase-update subspaces and latent temporal observability

## Source

Continues the SPC-01 to SPC-05 sequence from OpenAI Math research-family
#129 (deferred-context state distinguishability) and #130 (exact Fourier
coordinate representation). The mathematical bridge is documented in
`patrickdugan/MathCan/research/spectral_aliasing/SPC-06.md`.

## Fixed sample and audit conventions

* Two frozen SPC-04 CPU checkpoints, `spc04_seed41522.pt` and
  `spc04_seed41523.pt`. Load through the existing weights-only hash
  verifier; checkpoint binaries are in the SPC-04 experiment archive.
* Train-only SVD fit uses prompt seed `904001`, 256 prompts and loop depths
  32, 48, 64. Held-out evaluation seeds are `905501`, `905502`, `905503`,
  64 prompts per seed per model, for 384 total examples.
* Phase labels come from actual graph-orbit periods and a full-state oracle
  reference; they are diagnostic labels, never used to fit the SVD.
* In all conditions, the prompt and weights are unchanged. This is a
  synthetic pointer-chase study with eight graph nodes and 48-channel
  answer-token states. No model-training claim applies beyond SPC-04.

## A: Repeated update suppression

From loop 4 onward, replace the answer-token increment each step with its
component orthogonal to a selected subspace. This is more stringent than
SPC-05's single-shot phase transplant: it asks whether continued cyclic
computation *requires* the selected directions.

| Condition | Loop 6 | Loop 12 | Loop 24 |
|---|---:|---:|---:|
| Native trained continuation | 382/384 | 362/384 | 358/384 |
| Freeze all 48 answer channels | 92/384 | 92/384 | 148/384 |
| Remove top 4 train-only SVD modes | 99/384 | 90/384 | 131/384 |
| Remove top 7 train-only SVD modes | 92/384 | 92/384 | 148/384 |
| Remove top 8 train-only SVD modes | 92/384 | 92/384 | 148/384 |
| Random raw eight-mode subspace (range of eight trials) | 276–332/384 | 198–258/384 | 192–239/384 |
| Random rescaled to equal removed update magnitude | 120–144/384 | 85–104/384 | 85–100/384 |

The seven-direction intervention has the same aggregate accuracy as the
full answer freeze for all registered horizons. A random eight-dimensional
orthogonal projection removes less energy and usually leaves computation
functional. Rescaling random directions to match the removed energy can
also damage computation severely. This prevents interpreting the result as
unique direction-specific necessity. The exact freeze test checks that the
model's readout remains unchanged, even while context continues updating.

## B: Instantaneous readout-null and exact-output-fiber interventions

`spc06_readout_observability.py` differentiates the actual native
LayerNorm + linear head. For seven independent logit margins, the analytic
48-channel Jacobian agrees with PyTorch forward-mode autodiff below `1e-9`
in CPU tests. A one-step phase displacement is split into immediate
margin-rowspace and margin-nullspace components. With finite changes, a
first-order margin-nullspace perturbation can still alter the current
readout due to nonlinear normalization.

The exact readout-fiber control addresses this by preserving **all eight
logits**, at the observed state, to float64 machine precision. It projects
into the linear head's centered nullspace and rotates the residual centered
state while keeping its norm and mean fixed. Random tangent controls on
the same fiber are constructed at exactly matched Euclidean chord lengths.

In 288 registered nonfixed cycles:

| Intervention | Subsequent next-phase selection | Current readout unchanged |
|---|---:|---:|
| Full answer transplant | 288 | 1 |
| Instantaneous margin-Jacobian null component | 129 | 176 |
| Exact all-logit fiber at scale 4 | 14 | 288 |
| Exact all-logit fiber at scale 8 | 16 | 288 |
| Eight random equal-chord exact-fiber controls, scale 8 | 8–14 | 288 |

At scale 8, the exact-fiber targeted intervention also recovers the
correct next-hop readout in 11/288 registered cycles while the immediate
logits remain invariant. Random equal-chord controls also sometimes
switch phase; the present evidence supports **future observability of
currently invisible state**, while directional superiority is weak and
should be tested on a larger independent sample. Scale 8 was included
in an exploratory fixed grid (0.5, 1, 2, 4, 8), rather than selected on
an external validation cohort. Logit equality is measured before the
subsequent 12 recurrent steps and does not hold them fixed.

## C: Future-output observability rank

`spc06_future_observability.py` perturbs all 48 answer-token coordinates
with paired positive/negative float64 differences at a fixed deep hidden
state, holding context and prompt unchanged. The resulting Jacobians of
readout *margin* vectors at future loops are stacked. The instantaneous
analytic Jacobian and central-difference estimate differ by roughly
`1e-10` relative norm. Two finite-difference step sizes, `2e-4` and
`1e-4`, produce identical relative-threshold rank classifications.

| Window of output margins | Trained rank (four model/prompt rows) | Untrained rank |
|---|---:|---:|
| Current only | 7 | 7 |
| Through next loop | 14 | 14 |
| Through two loops | 21 | 14 |
| Through four loops | 29–34 | 14–15 |
| Through ten loops | 32–40 | 14–15 |

Numerical rank uses `singular_value > 1e-3 * largest_singular_value`
of each stacked matrix. These are local derivatives, and the estimated
number of linearly distinguishable hidden directions depends on the
threshold, evaluation point, and horizon. The untrained models carry
fixed-point transients, which naturally attenuate many late directions.

## Execute

```bash
python -m unittest discover -s tests -p test_spc06_causal_observability.py -v
python -m experiments.spectral_state_cost_v0.spc06_phase_necessity \
    --checkpoint-dir /path/to/SPC-04/checkpoints --out /tmp/spc06_ablation.json
python -m experiments.spectral_state_cost_v0.spc06_readout_observability \
    --checkpoint-dir /path/to/SPC-04/checkpoints --out /tmp/spc06_fiber.json
python -m experiments.spectral_state_cost_v0.spc06_future_observability \
    --checkpoint-dir /path/to/SPC-04/checkpoints --out /tmp/spc06_observability.json
```

The lightweight CI tests can execute without the private checkpoint binary.
Frozen numeric summary and source JSON SHA-256 checksums are stored in
`receipts/spc06_summary.json`; the full raw JSON and checkpoint files are
in the SPC-06 standalone reproducibility archive. The workflow's runner
startup status is separate from CPU test verdicts.
