# SPC-05: phase-latch localization and low-rank causal controls

15 November 2026. New continuation of native LoopedChase SPC-04 on its two **unchanged**, verified trained checkpoints. All claims concern an 8-node pointer-chase toy model and 48-wide hidden state. Theory, exact-arithmetic oracles and the formal Lean specialization are in [`patrickdugan/MathCan` SPC-05](https://github.com/patrickdugan/MathCan/tree/feat/spc05-phase-latch-and-contrast-20261115/research/spectral_aliasing).

## Inputs and role split

- Load `spc04_seed41522.pt` and `spc04_seed41523.pt` from the SPC-04 research archive with `torch.load(weights_only=True)`; require matching state-dict digest and architecture. The binary checkpoints remain external to Git history.
- Input prompt is a fixed random 8-node permutation graph with one queried start vertex. The nine-token hidden state splits into eight context-token positions and one answer-token position. The input-injection tensor remains host-owned and unchanged under all interventions.
- New held-out prompt seeds: `905501,905502,905503` with `64` independent prompts each, per model (total 384). Train-only unsupervised SVD source: `904001` with 256 different prompts at fixed depths `32,48,64` (768 increment rows). Training itself only supervised hops 1..3 at SPC-04.
- Intervene at loop 64, continue for 12 loops. Evaluate the **current 76th hop** and separately the **shifted 77th hop**; this avoids the known third-hop-retention labeling bug.
- The independent full-state native reconstruction agrees exactly with the original `LoopedChase.forward` on tested early loops. A full-state future-phase transplant replays the shifted reference exactly by deterministic map composition.

## Registration and controls

Register a nonfixed computational cycle if its graph-derived integer pointer period exceeds one, the full-state relative period-return error at the evaluation depth is below `1e-4`, the relative one-step displacement is over `.05`, and phase references remain separated by more than `1e-8` in the Euclidean full state. Each intervention is assigned its nearest phase from **all p phase references**. Report excluded trajectories separately.

Of 384 prompt-model cases, 332 have graph period >1 and 288 meet the registration conditions (137 for seed 41522 and 151 for seed 41523). An answer-token transplant from the next step induces a phase +1 match in **288/288**, while a context-token transplant retains phase 0 in **288/288**. A full-state transplant induces phase +1 in 288/288 and serves solely as an exact semigroup control.

The answer state contributes around 96% and 79% of one-step phase-displacement energy for the two models at depth 64; these are prompt-dependent measured fractions. Under *equal full-state Euclidean norm* Gaussian noise of `.75 ×` each prompt's minimum full-state interphase distance, 244/288 retain phase zero when noise is confined to the answer token; the context-only matched-norm perturbation retains phase zero in 288/288. The noise directions are generated independently from a frozen seed for each condition and held-out item.

## Unsupervised contrast subspace

Compute the uncentered SVD of train-only answer-token phase increments; intervention rank `r` uses the projector onto the first `r` right-singular vectors. Compare to eight independently seeded random rank-`r` orthonormal projectors, each both in raw projected form and rescaled to the exact full-answer-transplant norm. The random controls use the same future state and prompts.

| Rank | Phase +1 among 288 registered, PCA raw | Random projector, equal-norm, range over eight trials |
|---:|---:|---:|
| 2 | 76 | 18–34 |
| 4 | 167 | 30–50 |
| 7 | **287** | 40–58 |
| 8 | **288** | 45–70 |
| 12 | 287 | 63–104 |
| 16 | 287 | 91–132 |

The eight-dimensional basis captures `0.97950461` and `0.98556797` of its respective **training** increment energy. Random-rank-eight controls switch only 45–70 registered states even after matching total intervention norm. The observed rank-seven near-perfect success is consistent with the elementary `q−1` upper bound on an ideal fixed `q=8` codebook's contrast span; a graph-conditioned representation can use additional dimensions.

## Reproduction

Run from the repository root with `torch==2.10.0`, `numpy==2.3.5`, Python ≥3.10, and the **archived SPC-04 checkpoints** supplied out of band:

```bash
python -m unittest discover -s tests -p 'test_spc05_phase_interventions.py' -v
python -m experiments.spectral_state_cost_v0.spc05_phase_interventions \
  --checkpoint-dir /path/to/SPC-04/checkpoints --batch 64 --out /tmp/spc05_interventions.json
python -m experiments.spectral_state_cost_v0.spc05_phase_subspace \
  --checkpoint-dir /path/to/SPC-04/checkpoints --batch 64 --random-repeats 8 \
  --out /tmp/spc05_subspace.json
```

The GitHub Actions workflow runs **independent CPU algorithm tests** that generate small initialized weights and verify all reference/transplant constraints, including checkpoint SHA guards. Full SPC-05 results were generated locally from archived external trained weights and are provided as compact frozen count receipts. Absence of the binary checkpoints on a GitHub runner must not be conflated with a failed experiment.

## Interpretive scope

The answer-token swap is an oracle intervention: it uses the same prompt's *future* hidden state to locate computational phase. The SVD basis is trained separately and then frozen, so its subspace generalizes to fresh toy prompts. It is not a causal decoder trained on correctness labels, and the results do not establish that the learned state is a pure eight-coordinate circuit or that the model is safe. The phase-contrast rank argument and the Lipschitz projection certificate concern idealized mathematical systems and require additional empirical regularity estimates before application as guarantees to a specific neural update.
