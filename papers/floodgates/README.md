# Floodgates: Looped Models are Fast Multi-Token Predictors

Ayhan Suleymanzade, Sean McLeish, Sehoon Kim, Jinwoo Kim, Antoine Bosselut (2026).
Version 1, 25 November 2026.

- [Versioned paper](https://arxiv.org/abs/2611.33896v1)
- [Original PDF](https://arxiv.org/pdf/2611.33896v1)
- [HTML](https://arxiv.org/html/2611.33896v1)
- [Code (Apache-2.0)](https://github.com/MisakiTaro0414/floodgates)
- [Released drafting adapters](https://huggingface.co/MisakiTaro0414/floodgates)
- Paper license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The source PDF is an external versioned reference. Run
`python papers/floodgates/fetch_paper.py` from the repository to retrieve the
original unmodified PDF and produce a SHA-256 receipt. Binary retrieval from
the authoring environment failed, so no local PDF hash is asserted here.

## Research note

Floodgates predicts a suffix of masked positions with a shallow shared
decoder, assisted by adapters confined to drafting. It reads the last-loop
prefix KV memory left by the full-depth target. A target evaluation of the
draft accepts the matching prefix and supplies the next target token.
For greedy decoding, exact token equivalence follows by induction if the
target evaluation at each draft position equals ordinary target decoding,
with identical position IDs, masks, cache boundaries and tie-breaking.
Sampling additionally needs the correct proposal conditional distribution
and rejection correction. Greedy parity alone does not certify sampling.

The authors report model/task averages up to 7.5x greedy acceleration
(Figure 1; A100 80GB); the numbers are external hardware measurements.
Sections 4.2–4.4 investigate early-depth supervision, deep cache
conditioning, additional draft passes and batch serving. Reusing deep
prefix memory is a drafting operation; replacing the target's own cache
by that memory would generally change its conditional predictions.

### Relevance to our projects

- **MoLT:** GSF-02 tested selective symbolic receipt retention; the strict
  prefix model in `experiments/molt_goblin_v1` tests a different recurrence.
  Floodgates suggests a read-only transport from a verified deep prefix to
  speculative slots. Do not conflate it with either of those mechanisms.
- **Aleph/Baby Goblin:** Direct A applies 24 blocks (10 core blocks twice,
  four shell/answer blocks). A one-core-pass drafter with unchanged shell
  and answer blocks applies 14 blocks per slot. This is a layer-count
  ratio, not a latency estimate. The 10M-position checkpoint is a runner
  pilot, not a competent model.
- **BPD:** test whether a train-only predictive subspace (identified by
  future-token readers or causal interventions) improves draft acceptance
  when used to initialize or constrain the adapter. Compare ordinary LoRA
  and random/shuffled directions at equal rank and updates. Selection on
  a final evaluation panel would invalidate the claim.
- **LSA/RLT:** train final-only and all-depth supervisors at matched
  exposure and applied block work. For elastic drafting, track the draft
  loop clock and verify with the unchanged full-depth target. Every
  additional speculative pass must be charged.

For k proposed tokens and A accepted prefix tokens, one target verification
emits A+1 tokens (except at a generation limit). An idealized latency model is
E[A+1] * C_AR / (C_draft(k,E) + C_verify(k+1) + C_cache).
The denominator depends on batch, sequence length, precision and hardware.
The acceptance curve P(A>=j) matters more than per-position marginal accuracy.
A reduction in serialized forward calls can coexist with higher total
token-block visits.

## Companion gym

See HybridTRMLDT's `research_gym/benchmarks/floodgates_gym/` (GSF-10).
First stage: train compact tied causal targets on synthetic streams, freeze
their weights, train small masked-block adapters with first/deep prefix
conditioning, test cache rollback and exact greedy parity. The CPU pilot
does not reproduce the seven published checkpoints, elastic drafting,
sampling or concurrent serving. Native KV work and teacher quality are
reported separately. Follow with rank/transport comparisons, BPD-guided
adapters and calibrated draft-length control; then register a bounded
researcher task for GoblinBench. No GPU allocation is implied.

## First companion results (GSF-10)

[CPU pilot](https://github.com/MoralityLabAI/HybridTRMLDT/actions/runs/46096326247):
two synthetic targets, eight trained adapters, 192 audited traces, zero greedy
mismatches. At matched rank 4, first/deep/null cache emitted 3.470/3.097/1.636
tokens per verification for the final-only target and 3.429/2.462/1.540 for
the all-depth target. First-loop conditioning won this small fixture. The
best arm still used 1.632 times ordinary serialized block-token work; actual
latency acceleration remains unestablished. The targets differ in truth
accuracy (95.49% vs 85.42% on development), so their draft acceptance is not
an isolated supervision effect at equal target quality.

[FG-05 prospective protocol](https://github.com/MoralityLabAI/HybridTRMLDT/blob/gym/gsf10-floodgates-20261128/research_gym/benchmarks/floodgates_gym/FG05_PROTOCOL.md)
adapts NC-03-style gradient covariance to frozen Q/V/O output perturbations
on masked future-token log probabilities. Compare guided, Haar and shuffled
orthonormal LoRA B initialization with zero A, plus ordinary LoRA. Both factors
remain trainable, every physical update starts at zero, and discovery roots
are distinct from fit and evaluation. Run the same rank-four comparison for
first/deep prefix memory on two fresh synthetic target checkpoints. This is
BPD-inspired initialization, not a native BPD decomposition or evidence of
published-model acceleration. See [HybridTRMLDT PR #49](https://github.com/MoralityLabAI/HybridTRMLDT/pull/49).


### Elastic refinement study (FG-06)

[FG-06 prospective protocol](https://github.com/MoralityLabAI/HybridTRMLDT/blob/gym/gsf10-floodgates-20261128/research_gym/benchmarks/floodgates_gym/FG06_PROTOCOL.md)
tests additional frozen shared-block passes on one-pass-trained adapters.
All policies begin with the same adapter-active masked-block computation.
Subsequent visits use the pretrained block, either repeatedly reading one
verified prefix depth or cycling the already verified per-loop cache bank.
RLT-style coefficients 1/3, 1/2, 1 are paired with mean-matched constant
damping 5/12, 5/12, 1. Reverse cache cycling preserves the forward cycle's
depth multiset. These are inference-only forward controls, not RLT training
or learned MoLT depth attention. The primary work endpoint counts actual
serialized draft and target block calls, with token-block visits and logical
prefix reads separate. Extra passes may increase the cost even when they
improve acceptance. Exact greedy parity is mandatory.

Two new synthetic target seeds, four adapter fits, 32 inference cells and
768 planned dedicated trace episodes. Results pending CPU audit.

## Citation

```bibtex
@misc{suleymanzade2026loopedmodelsfastmultitoken,
  title={Looped Models are Fast Multi-Token Predictors},
  author={Suleymanzade, Ayhan and McLeish, Sean and Kim, Sehoon and Kim, Jinwoo and Bosselut, Antoine},
  year={2026},
  eprint={2611.33896},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2611.33896v1}
}
```
