# SPC-08: disentangling phase-driving and deferred-output directions

Source pair: OpenAI mathematics families #129 (deferred-context state
complexity) and #130 (exact Fourier/tensor coordinate methods), originally
selected in SPC-01. Companion derivation and finite Lean certificates:
`patrickdugan/MathCan/research/spectral_aliasing/SPC-08.md`.

**Model:** two frozen SPC-04 `LoopedChase` checkpoints, trained with
three-hop supervision on eight-node pointer graphs, hidden answer width 48.
The exact original weight-hash verification remains enforced by
`spc05_phase_interventions.model_from_receipt`. All tests use CPU float64.

## Protocol and preregistered dimensions

Phase-state SVD uses 256 original training prompts, seed `904001`, at
loops 32, 48, and 64. The local temporal Jacobian training uses 16 prompts
from seed `920731`, depth 32, and 8 future output-margin steps. On each
training prompt, let `O` be all current+future seven-logit-margin
Jacobians and `J0=O[:7]`. Define the *prompt-local* current-readout
null projector `N0=I-J0†J0` using the exact analytic/fd rank check. The
three equally prompted, normalized Gramians are:

- `G0 = mean(J0ᵀJ0 / ||O||F²)` for the immediate output;
- `Gtotal = mean(OᵀO / ||O||F²)` for all output times;
- `Gdelayed = mean((O_future N0)ᵀ(O_future N0) / ||O||F²)` for future
  output sensitivity after projecting away immediate readout.

Each top-eight orthonormal basis is selected on this training cohort only.
The SPC-07 256-prompt orthogonal Procrustes fit supplies the source-target
alignment and its shuffled-input negative control.

Evaluation uses the eight untouched SPC-07 pointer seeds `910701..910708`,
64 inputs per seed per target model, depth 64, followed by 12 recurrent
steps. The 790 stable nonfixed cycles are registered using the exact
pointer period, numerical cycle closure, and a minimum phase gap. A
separate held-out derivative cohort uses seed `910709`, 12 prompts per
checkpoint, with zero training-prompt overlap. No checkpoint weights or
learned intervention directions are updated using held-out data.

## Results: eight-dimensional geometry

| Metric | Checkpoint 41522 | Checkpoint 41523 |
| --- | ---: | ---: |
| Phase basis captures immediate Gramian | 90.38% | 90.60% |
| Phase basis captures delayed-only Gramian | 5.06% | 5.05% |
| Deferred basis captures immediate Gramian | 0.46% | 0.58% |
| Deferred basis captures delayed-only Gramian | 75.30% | 79.63% |
| Phase/deferred projector overlap | 0.060 | 0.052 |

The held-out Jacobian captures on 12 fresh prompts per checkpoint were:

| Trained checkpoint | Phase: current | Phase: deferred | Deferred: current | Deferred: deferred |
| --- | ---: | ---: | ---: | ---: |
| 41522 | 88.23% | 9.25% | 1.09% | 58.21% |
| 41523 | 88.06% | 4.63% | 1.38% | 65.08% |

The trained local output Jacobian retains 28–42 numerically observable
state directions at 9 output times across the fresh prompts (relative
singular-value threshold `1e-3`). These numerical ranks are sensitivity
estimates, not kernel proofs of nonlinear state identifiability.

## Results: interventions and controls

| Condition | Correct registered phase | Full-batch phase switches |
| --- | ---: | ---: |
| Target phase-eight, natural projected magnitude | 788/790 | 788 |
| Target ordinary temporal-eight | 581/790 | 581 |
| Target deferred-eight | 2/790 | 2 |
| Deferred-eight rescaled to *full natural-step norm* | 149/790 | 149 |
| Paired-aligned source phase-eight | 786/790 | 786 |
| Paired-aligned source deferred-eight | 1/790 | 1 |
| Paired-aligned source deferred-eight, norm-matched | 96/790 | 96 |
| Immediate readout-Jacobian row component | 214/790 | 214 |
| Immediate readout-Jacobian null component | 368/790 | 368 |

Projected natural phase increments retain about 98% of total increment
energy in their phase-eight basis. Their deferred-eight projections
retain a median 2–3% of energy. The `full norm` matched deferred
interventions therefore multiply their projection norms roughly 6×,
which may create unnatural hidden-state inputs. The 149/790 result is
an adversarial sensitivity control, not a realistic on-manifold update.

The seven margin-Jacobian null components cause delayed phase changes
in 368/790 cycles, but immediate logit changes can be large due to the
nonlinearity of LayerNorm. A stricter exact-readout control uses the
existing SPC-06 method to rotate on a LayerNorm norm sphere, preserving
**every current logit** with numerical error below 1e-9. With the phase
basis, scale 8, this switches 41/790 registered cycles. Four
independent magnitude-matched random exact-fiber controls switch 12, 17,
12, and 15 respectively. Thus some current-logit-invisible dynamics can
influence phase, but the strong phase-control effect of natural-step
projection includes current-readout-changing components.

The 16-batch paired cluster bootstrap for target phase-eight versus
target deferred-eight reports a difference of 99.49 percentage points
(95% interval 99.01–99.87). This is a conditional finite-model rate,
not a general-population confidence guarantee.

## Reproduction

From the LoopedTransformers root with CPU NumPy, PyTorch and SciPy:

```bash
python -m unittest discover -s tests -p 'test_spc08_phase_readout.py' -v
python -m experiments.spectral_state_cost_v0.spc08_phase_readout \
  --checkpoint-dir /path/to/spc04/checkpoints \
  --out /tmp/spc08_full.json --summary /tmp/spc08_summary.json
python -m experiments.spectral_state_cost_v0.spc08_heldout_jacobians \
  --checkpoint-dir /path/to/spc04/checkpoints \
  --out /tmp/spc08_holdout_jacobians.json
```

The compact summary and held-out derivative receipt are committed. The
full per-prompt causal bits and the weight-verifiable checkpoint files
are retained separately in the SPC-08 reproducibility archive.

## Interpretation and limits

The top-eight *phase-driving* subspace is strongly represented in the
current readout-sensitive geometry. The top-eight *delayed-observable*
subspace carries future-output sensitivity that is largely absent from
the instantaneous readout. Their geometry and causal interventions
support a practical measurement distinction; parts of the full phase
transition also live in the immediate Jacobian nullspace. This is
consistent with distributed nonlinear computation and does not give a
perfect partition into independent neural modules.

Cross-checkpoint transfer is calibrated with paired unlabeled hidden
activations, and depends on the particular LoopedChase graph task. The
results say nothing by themselves about safety interventions or
language-model reasoning. Additional widths, task families, training
seeds, precision and test-time perturbation controls are needed before
those claims can be investigated.
