# FC-02: exact formal curriculum transfer with LR-01 TypedLoop

Date: 2026-11-12

This experiment imports the actual shared graph-attention cell in
experiments/lean_reasoner/train.py and trains a new **11,521-parameter**
reachability policy on the deterministic 1,536-problem curriculum in
public MoralityLabAI/Compactification-Theory. All claims are restricted
to bounded propositional Horn logic.

## Provenance and CI

- Certified generator source:
  MoralityLabAI/Compactification-Theory commit
  7aca2bc3fc5586c5b7997fba868bd179db761121
- Generated dataset SHA-256:
  8e86a41dd93ff162072799332ebbef4c7134951798eb4af7ebb4ab044bebd56f
- Lean 4.19.0: 768 positive examples compiled with explicit proof terms;
  the other 768 have independently checked finite countermodels.
- [Source proof CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43240137061)
- [Model training and OOD CI](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43241554303)
- [Public outcome receipt](https://github.com/MoralityLabAI/Compactification-Theory/blob/feat/refutationbench-v0.1/research/refutation_bench/results/fc02_lr01_depth_extrapolation.md)

CI installs CPU PyTorch, checks six learner unit tests (including
provenance/input-label isolation), then performs 350 AdamW training steps
for 1-pass and 4-pass models under two prespecified seeds (17, 29).
The train split contains 1,152 graphs at depths 2–4, development has 192
graphs at depth 5–6, and public OOD has 192 graphs at depth 7–8. No
held-out formal sample participates in gradients.

The student uses explicit graph incidence, fact marks, a goal mark,
shared TypedLoop recurrence and an atom-reachability prediction head.
A proposed countermodel assigns each atom according to its model output.
The independent checker verifies every premise and goal against those
exact bits. It never modifies an assignment. Positive proof terms are
reference artifacts compiled by Lean; the student does not generate
those proof terms.

## First experimental result

| Trained loops | Inference loops | Seed | Dev correct | OOD test correct |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 17 | 98/192 | 96/192 |
| 4 | 8 | 17 | 172/192 | 175/192 |
| 4 | 12 | 17 | 152/192 | 164/192 |
| 4 | 4 | 29 | 98/192 | 96/192 |
| 4 | 8 | 29 | 191/192 | 100/192 |
| 4 | 12 | 29 | 183/192 | 168/192 |

Trained four-pass models reached 100% accuracy on a fixed subset of
192 training cases. Testing with four passes collapses toward calling
all deeper claims false. At 12 inference passes, both seeds reach
85–88% OOD verdict accuracy with unchanged weights.

The **certificate success** remains a distinct statistic: with 12
inference passes, 46/96 and 56/96 invalid OOD examples, respectively,
received a fully verified model-proposed countermodel. All submitted
certificates are checked by an exact finite verifier, which permits
a selective policy to abstain on the rest.

The additional 8/12-loop inference settings were chosen after
observing the initial four-pass depth failure. This is an exploratory
follow-up on public OOD data, with considerable seed sensitivity. A
new sequestered test would be necessary for a confirmatory claim.

## Reproduction

Clone the private LoopedTransformers repo and the public Compactification
repo at the exact commits cited above. In Linux/WSL with Python 3.11
and CPU PyTorch 2.5.1, from LoopedTransformers root:

~~~sh
export RB_BENCH_ROOT="$(realpath ../Compactification-Theory)"
python -m unittest discover -s experiments/refutation_transfer \
  -p 'test_formal_student.py' -v
python experiments/refutation_transfer/formal_student.py \
  --benchmark-root "$RB_BENCH_ROOT" \
  --out /tmp/rb_formal_student \
  --loops 1 4 --seeds 17 29 --extra-loops 8 12 \
  --steps 350 --batch-size 48 --threads 2
~~~

Per-seed checkpoint files, prediction JSONL and a source-hashed
report.json are uploaded as short-lived GitHub Actions artifacts.
The model runner never downloads a pretrained checkpoint.

## Next experiments

A follow-up protocol should assign new graph families to a sealed
test split before looking at their labels. Hold depth constant
across graph families to distinguish depth extrapolation from
structural transfer. Add an untied-layer baseline matched on
compute. Upgrade the student toward a proof-step recommender whose
chosen action can be kernel checked, and test the capability on
raw-source rather than pre-curated benchmark evidence.
