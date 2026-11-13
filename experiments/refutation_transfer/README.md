# LR-01 × RefutationBench CI pilot

This experiment trains an **actual PyTorch model** with the tied graph-attention
cell from [LR-01](../lean_reasoner/README.md) and evaluates its output with
RefutationBench's existing scorer. The refutation benchmark remains authored
in the public \`MoralityLabAI/Compactification-Theory\` repository at
\`research/refutation_bench/\`. The training code lives here, beside its
imported LR-01 implementation, because LoopedTransformers is private and public
GitHub Actions runners cannot read its code without separate credentials.

## FC-05: dual-sided certified proof acquisition

The [FC-05 implementation and results](FC05_RESULTS.md) add a trained
4,705-parameter next-rule head to the 11,521-parameter LR-01 trunk,
while the exact checker validates the original model-proposed action
trace and Lean compiles each accepted positive proof term. The first
held-out CI compiled 318 positive terms, verified 369 negative
countermodels, and issued zero false formal certificates across the
four seed/split conditions. The proof head's coverage equalled
simple rule-ordering baselines because the ten-action budget
saturates every valid graph; the documented finding is operational
certificate generation, with learned proof selection still unproven.

## FC-04: validation-calibrated adaptive stopping

The [FC-04 report](FC04_RESULTS.md) records a two-job CI trial with
separate FC-02 train/calibration and fresh FC-04 evaluation checkouts.
The original 11,521-parameter LR-01 student executes **real early exit**
on exact verified counterexamples or two-checkpoint stable provisional
positives. Across two seeds and two public test depths, adaptive halting
certified 91–96 of 96 invalid cases with 5.0–8.4 mean recurrent passes,
compared with fixed12. Some provisional positives remained wrong, and
fixed4 was more certificate-efficient in one same-depth condition.
The [source test and preregistration](https://github.com/MoralityLabAI/Compactification-Theory/pull/107)
were committed before student evaluation. The result is a bounded Horn
transfer/compute finding, with no paper-level theorem claim.

## FC-02: larger certified curriculum

The [formal curriculum experiment](FORMAL_CURRICULUM.md) is a second, larger training track using the same original LR-01 TypedLoop cell and a newly trained 11,521-parameter reachability head. Its 1,536 generated tasks include 768 Lean-compiled proofs and 768 exact finite countermodels. Four trained loops perfectly learned the sampled training set, and additional inference-only recurrence improved public OOD depth-7/8 verdict accuracy to 85–88% at twelve passes across two seeds. Results and limitations are preserved in [the public outcome receipt](https://github.com/MoralityLabAI/Compactification-Theory/blob/feat/refutationbench-v0.1/research/refutation_bench/results/fc02_lr01_depth_extrapolation.md).

The original six-case textual verdict experiment below remains independently reproducible.

## Frozen input and scope

- RefutationBench commit:
  \`27a12f866204855e224c5f6bd00888ab41d6fa5c\`
- Dataset: 25 public cases from 11 provenance families.
- Train: six **synthetic** cases only, three source families.
- Development evaluation: 19 cases from eight different families, including
  all Nielsen/Semita and sourced historical paper analyses.
- A held-out test split does not yet exist. Development labels and answers are
  public; the CI report is a transfer pilot, not a blind generalization score.
- No LR-01 checkpoint is committed. The experiment imports its actual
  \`TypedLoop\` class from \`experiments/lean_reasoner/train.py\`, initializes
  weights from seed and adds a trainable hashed-text encoder, four-way verdict
  head, eight-way flaw head, and source-evidence pointer head.
- The input graph contains claim, citation, and verdict nodes connected by
  typed candidate/evidence rules. Word tokens are hashed with BLAKE2b into
  a 1024-slot embedding table. Source IDs, case IDs, family labels, and gold
  answer keys never enter model input embeddings.
- Multi-task training uses the train-only label, a decisive flaw label and
  source-evidence targets where available; source order is shuffled.
- Infer using one and four tied passes, two prespecified seeds, 120 steps each.
  A deliberately label-shuffled training control tests a cheap shortcut.
- Baselines include constant verdicts; evaluation calls the public scorer on
  the same development items and records verdict macro-F1, false refutations,
  and the supported-refutation annotation proxy.

The 1-pass and 4-pass arms share the same trainable parameter count. Their
compute is **not matched** because the deeper arm invokes the shared cell
four times per example. No dev-informed checkpoint selection is performed.
Only the learned weights and input-only public text enter the dev forward
pass. The scoring stage reads gold only *after* predictions exist.

### What this experiment cannot certify

A verdict classifier with a token/evidence pointer is not a proof-generating
language model. Its explanation is a standard warning that human review is
required. Evidence-ID matching is only the benchmark's documented annotation
proxy; it is not a Lean kernel check. The separate source cases themselves
are curated paraphrases and can telegraph the refutation. Claims of
mathematical reasoning compactification, new proof discoveries, hidden-test
transfer, or full semantic refutation ability require subsequent evidence.

## GitHub Actions

[Workflow](../../.github/workflows/refutation-transfer.yml) checks out
the public benchmark at its commit, pins PyTorch CPU runtime, runs unit tests,
trains the actual recurrent network, independently replays benchmark scoring,
and uploads \`report.json\`, response files and SHA-256-identified checkpoints.
The report includes training durations and peak RSS. Runner artifacts expire
after 14 days.

All model and dataset code runs without API keys or private remote services.
A change to public RefutationBench requires an explicit reviewed commit-pin
update here; the source-family split and input-only export invariants are
unit-tested.

## Reproduce locally

In a Python 3.11 environment with \`torch==2.5.1\` (CPU or CUDA build),
and a checkout of the pinned Compactification-Theory commit in
\`../Compactification-Theory\`:

\`\`\`sh
export RB_BENCH_ROOT="$(realpath ../Compactification-Theory)"
python -m unittest discover -s experiments/refutation_transfer -p 'test_*.py' -v
python experiments/refutation_transfer/runner.py \
  --benchmark-root "$RB_BENCH_ROOT" \
  --out /tmp/refutation_lr01 \
  --loops 1 4 --seeds 17 29 --steps 120 --threads 2 --label-shuffle
\`\`\`

Each checkpoint gets a matching data/model provenance record in
\`report.json\`. Predictions can be rescored by:

\`\`\`sh
cd "$RB_BENCH_ROOT"
python -m research.refutation_bench.expansion.suite evaluate \
  --split dev --responses /tmp/refutation_lr01/lr01_l4_s17_dev_predictions.jsonl \
  --out /tmp/refutation_lr01/independent_score_l4_s17.json
\`\`\`

## Next model-stage requirements

To train a **certificate-generating** student, create a new source-family-
disjoint training set with proof obligations and machine-checkable examples,
then add a verified decoder or external theorem-prover loop. Only then can
first-fault certificate cost be measured as a learned policy rather than a
keyword-and-pointer proxy. The existing 400M Lean-CGRT / LOTUS experiments
remain separate GPU projects.
