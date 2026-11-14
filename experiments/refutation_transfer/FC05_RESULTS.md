# FC-05 model-proposed positive Lean proofs and exact negative certificates

**First full held-out CI:** [#43399759225](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43399759225),
13 November 2026. Both jobs passed. The test job compiled **318 model-selected
positive proof traces** in Lean 4.19.0; two inspected endpoint
theorems reported zero extra axioms. All released negative
certificates passed independent exhaustive Boolean checking.

The full public result ledger is in
[Compactification-Theory FC-05](https://github.com/MoralityLabAI/Compactification-Theory/blob/feat/refutationbench-fc05-positive-proofs/research/refutation_bench/results/fc05_dual_certificates_20261113.md),
with a [post-hoc saturation diagnosis](https://github.com/MoralityLabAI/Compactification-Theory/blob/feat/refutationbench-fc05-positive-proofs/research/refutation_bench/fc05/POSTHOC_SATURATION.md).

## Frozen source and training

- Registration committed first:
  `e6ab5716e3316441adad131b087a50d1ca0f2734`.
- FC-02 training source:
  `7aca2bc3fc5586c5b7997fba868bd179db761121`.
- FC-05 first held-out source:
  `e2fd9f6a0bc7466475541a7ee2afd1320b4e0366`.
- FC-05 canonical dataset SHA256:
  `faced2fce3ecc176a86871211ec013c1a8deb1a83f3b5e89f9cb650c979dd722`.
- Source CI checked 384 cases, compiled 192 valid reference Lean
  proofs and verified 192 negative examples.
- Each seed trains a freshly initialized 11,521-param LR-01
  recurrent trunk on train-only source (350 steps), then a frozen-trunk
  4,705-param learned proof-rule head (250 steps on 1,728
  training-only exact proof prefixes).
- Stopping threshold from FC-02 dev: 0.70 (seed17), 0.55 (seed29).
- New test is loaded only in the dependent evaluation job,
  after SHA256 checkpoint freezing.

## Certificates from actual model actions

The learned head may select at most ten applicable original rule
indices. Its produced action list is replayed independently, with
each antecedent required to have been proved beforehand. A failure
becomes an abstention. Successful traces are rendered into Lean
theorem statements from their **own exact action sequence**.
No source oracle is used to correct or finish the model's trace.

Negative refutations remain gated by exact complete Boolean models.

| Seed | Test depth | Positive Lean proofs /96 | Negative exact countermodels /96 | Abstentions |
| --- | --- | ---: | ---: | ---: |
| 17 | Same | 96 | 96 | 0 |
| 17 | Deep | 87 | 90 | 15 |
| 29 | Same | 96 | 96 | 0 |
| 29 | Deep | 39 | 87 | 66 |

Across 768 seed-case decisions, **687 certificates** passed exact checks.
The 318 positive term count can include different proof traces for
the same underlying test proposition across seeds.
No false certificates were issued.

## Learning comparison: negative result

The random-legal and immediate-goal baselines obtained **exactly
the same certified coverage**, with eight paired-discordance counts
of zero. The fixed ten-step budget covers all legal novelty
actions in these tiny generated graphs: with d≤8, n=d+4 atoms,
one initial fact and an unprovable guard, at most d+2≤10
novel consequences can ever be derived. Thus any legal rule
ordering eventually derives a valid goal within budget.

At identical certified coverage the learned policy spends more
compute, e.g., seed17 same-depth **2,790 recurrent calls**
versus **990** for either non-neural rule-order baseline.
No learned proof-step-selection superiority was established;
the extra head learns supervised rule preferences without a
test where they can improve success.

The first source validation test was corrected for reference
trailing actions before the successful first model run. The
strict model certificate verifier still rejects unnecessary
post-goal actions.

## Reproducibility

[GitHub Actions workflow](../../.github/workflows/refutation-fc05-dual.yml)
separates train and evaluation jobs. The run archived
`fc05-frozen-models` and
`fc05-dual-certificates-43399759225` (30-day retention).
The latter contains `FC05_model_proposed.lean`, all twelve
per-case prediction sets, the full JSON metrics and Lean
kernel log.

The symbolic finite Horn solver itself is 100% correct.
These results concern encoded propositional inference, with
no claimed transfer to reading or proving human mathematical
papers. The next preregistered trial needs a genuinely
binding proof-action budget and reachable distractors.
