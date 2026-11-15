# SPC-04: trained LoopedChase preserves computational phase cycles

15 November 2026. CPU-only continuation of SPC-01/02/03/J. Work on an
isolated continuation branch derived from `feat/spc01-spectral-state-cost-20261115`.

**Most important correction:** A model supervised to compute hop `t` after
loop `t` must be evaluated on the current `t`-hop label. Comparing its
42-loop output to the 3-hop label measures a different property: frozen
answer retention. SPC-04 reports both metrics separately.

## Pinned protocol

- Native repository model: `research_gym/scripts/loopscope_toy.py`,
  `LoopedChase` with 8 nodes, width 48, one tied attention block, 4 heads.
- Two independent training seeds `41522,41523`. Train 1,200 AdamW updates,
  batch 64, lr `0.001`, weight decay `0.005`, gradient clipping 1.
- Train loop-1, loop-2, loop-3 readouts against their respective 1-, 2-,
  3-hop targets, with equal cross-entropy weights. Never include a future
  target in the tokens. Three fixed held-out seeds `881001,881002,881003`
  generate 512 inputs each (1,536 per model, 3,072 total).
- The untrained control is a deep copy of the *same seeded initial weights*
  before the optimizer's first update. Matched prompts and width are used.
- Evaluate readout on the current-hop label at loops
  `1,2,3,6,12,24,42`. Report old 3-hop answer retention at late loops
  as an explicitly separate metric.
- In float64, reconstruct the recurrence over the entire hidden state,
  including all nine token positions. Verify exact equality to the native
  model's answer token. Compute single-loop Jacobian estimates at 3/30
  and **ordered** cycle-Jacobian products at depth 64.
- Discover the pointer-cycle length from the tokenized permutation and
  start node, and measure return-to-period at depth 64 on 64 new prompts
  per seed. Classify a nontrivial matching cycle only if the true period
  has relative residual <1e-4, the lag-one residual is >0.05, and the
  smallest detected period equals the true period. Thus a fixed point
  receives zero credit even when F^p also returns it to itself.

## Local results (CPU, PyTorch 2.10.0, numpy 2.3.5, SciPy 1.17)

| seed | arm | hop 3 | current hop 6 | current hop 12 | current hop 24 | current hop 42 |
|---|---|---:|---:|---:|---:|---:|
| 41522 | trained | 1.00000 | .98047 | .92643 | .88021 | .88086 |
| 41522 | untrained | .14909 | .16471 | .19076 | .21094 | .19141 |
| 41523 | trained | .99935 | .99349 | .97852 | .96940 | .97201 |
| 41523 | untrained | .10742 | .08008 | .05599 | .04167 | .05729 |

The combined loop-42 trained accuracy is `.926432` over 3,072 held-out
examples, while the untrained control is `.124349`. Nominal chance is `.125`.
Both trained checkpoints are near perfect at the supervised depth. Unlike
answer retention, trained current-hop accuracy remains high at 42 loops,
14 times the largest supervised loop count.

The readout at loop 42 is usually **different** from the third-hop answer.
That is expected behavior for an unbounded pointer chaser. A model may be
successful at sequential computation while failing the frozen-label metric.

### Full-state and temporal evidence

| seed | arm | temporal DC energy | answer-state step norm at depth 41 | single-Jacobian rho at depth 30 |
|---|---|---:|---:|---:|
| 41522 | trained | .21339 | 16.2928 | .56331 |
| 41522 | untrained | .99621 | ~9e-16 | .44298 |
| 41523 | trained | .49574 | 30.6914 | .81201 |
| 41523 | untrained | .99657 | ~2e-15 | .42344 |

Despite trained one-step Jacobian spectral estimates below one, these
trained trajectories have large late *differences between successive
states*. This is consistent with a stable periodic orbit. Every
untrained control converges to a fixed point.

**Phase-alignment cohort:** Of 55 held-out prompts with a nontrivial
pointer cycle, 45/55 (seed 41522) and 50/55 (seed 41523) recover their
actual pointer period at depth 64 under the frozen thresholds. The
remaining failures are retained in the full receipt, including cases that
converge to fixed points and nonmatching/weakly closed cycles. The
matched untrained controls have 0/55 nontrivial aligned cases each.

### Ordered Floquet products at depth 64

| seed | graph cycle p | full-state ||F^p(h)-h||/||h|| | estimated rho(D F^p) | Arnoldi |
|---|---:|---:|---:|---|
| 41522 | 7 | 6.76e-12 | .05910 | converged |
| 41523 | 2 | 1.71e-8 | .62278 | converged |

These are approximate orbits and numerical Ritz eigenvalues. Their
combination is evidence for locally attracting **nonfixed** computations.
One-step Jacobian eigenvalues do not by themselves certify attraction
around cycles: the product's factor order matters. A separate exact
nilpotent counterexample is formalized in `MathCan/PhaseMemory.lean` and
proved for general parameters in the SPC-04 math note.

## Local reproduction

```bash
python -m pip install numpy==2.3.5 scipy==1.17.0 torch==2.10.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests -p 'test_spc04_trained_dynamics.py' -v
python -m experiments.spectral_state_cost_v0.spc04_trained_dynamics \
  --seeds 41522 41523 --steps 1200 --eval-items 512 \
  --checkpoint-dir /tmp/spc04/weights --out /tmp/spc04/trained_vs_init.json
```

The standalone `spc04_floquet_cycle.py` exposes
`phase_alignment_report` and `floquet_probe`, which consume exactly
native LoopedChase states. CI runs a small deterministic smoke and
uploadable JSON receipt; full training evidence and **binary checkpoint
files** accompany this research turn as a local artifact, since committing
trained weights to the repository would add binary baggage.

## Interpretation and limits

This is a two-seed demonstration of algorithmic phase preservation for a
small synthetic pointer task. The experiment does not test a large
language model, trained Huginn weights, neural belief states, post-training
safety effects, or generalizing a new theorem about transformers.

A separate mathematical construction shows that polynomial maps may
have exponentially many globally distinguishable superattracting phase
states despite zero Jacobians on their invariant set. This is a precise
reason to examine phase readouts and cyclic monodromy alongside local
linear contraction in mechanistic-interpretability work.
