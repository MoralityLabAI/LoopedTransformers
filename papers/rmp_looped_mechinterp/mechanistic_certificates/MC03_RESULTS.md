# MC-03: trained recurrent states with checked counterfactual continuations

2026-11-11. Code commit `5c95ed2bddcebf2ad7127028542af762921e1a80`.
[PR #9](https://github.com/MoralityLabAI/LoopedTransformers/pull/9) is stacked on
MC-02 PR #7. This report concerns new small CPU-trained models. The external
Q4/Q5 checkpoint panel has not been evaluated here.

## Frozen design and execution

The training recipe was committed at `92c1613c052dd1c04b8ac1b236f2713b0feaec5f`
before training. [MC03_PLAN.md](MC03_PLAN.md) records its budget and interpretation.
The complete unchanged LoopedChase and SEM-ECHO source modules were exported
from GitHub, checked against their Git blob and SHA-256 identities, and executed.

Two seeds, 201 and 202, each trained a 30,582-parameter LoopedChase: width 48, one
block, four heads, six graph nodes. Training supervised every visit over four
loops, with lambda=1. Each seed completed the fixed 3,000 AdamW updates at batch
64, without a budget stop. Local training took 39.27 and 39.35 seconds on one CPU
thread. No GPU or external model weights were used.

The 720 labeled six-node permutations were split by a fixed hash order into 480
training graphs, 120 calibration graphs and 120 test graphs. Calibration was
reserved and unused. Every start on every test graph gives 720 held-out items.
All starts and edge orders of a graph stay in its split. A fixed 64-item subset
supplies the intervention panel. Its indices were written before training.

Both seeds completed training before any held-out evaluation was read. There
was no best-checkpoint selection, hyperparameter change or discarded seed.

## Learned task behavior

| Measurement | Seed 201 | Seed 202 |
|---|---:|---:|
| Correct answer at training depth 4 | 720/720 | 720/720 |
| Correct answer at depth 16 | 718/720 (99.72%) | 693/720 (96.25%) |
| Every state correct through depth 16 | 716/720 (99.44%) | 691/720 (95.97%) |
| First visit with any held-out error | 13 | 8 |

Complete traces were checked against MC-01's exact finite-field relation. A
correct final answer can coexist with a prior state error, which explains the
difference between the last two accuracy rows.

The saved initial weights give depth-16 accuracies of 9.03% and 17.22%. Seed
202's initialization has 24 correct complete traces, all on fixed-point starts.
Its successful constant sequences therefore provide a useful trivial-path
control. Both initializations have zero successful cursor transplants.

## Counterfactual cursor transfer

At a selected visit, the donor cursor is copied into the receiver's cursor
position. Receiver context tokens and the fixed reinjected embedding remain
unchanged. The native head immediately reads the donor cursor. The expected
future was fixed from the receiver graph before replay, using MC-02's selection
and intervention code without modification.

| Measurement | Seed 201 | Seed 202 |
|---|---:|---:|
| Endpoint-correct receiver/visit slots | 320/320 | 316/320 |
| Selected intervention pairs | 32 | 32 |
| Full four-step counterfactual traces accepted | 29/32 (90.63%) | 16/32 (50.00%) |
| Predeclared equal-receiver primary rate | 91.07% | 51.85% |
| Distinct receivers in that primary | 28 | 27 |
| Random-edit future matches | 8/256 (3.13%) | 7/256 (2.73%) |
| Sham receiver continuations preserved | 32/32 | 32/32 |
| Constant-shift receiver continuations preserved | 32/32 | 32/32 |

The random comparison checks the four future labels even when its immediate
post-edit cursor differs from the donor. The transplant predicate additionally
checks that boundary. Random directions match the raw edit norm; post-injection
LayerNorm norms are retained separately. These are descriptive rates with shared
donors and repeated receivers, so no independent-pair confidence interval is
claimed. Coverage and visit-stratified counts remain in the raw receipts.

Both trained models preserve their sham and native-donor replay controls. The
trained donor-replay arms match none of the receiver-conditioned targets, as
expected for the selected graph-discriminating cases. Untrained donor replays
can accidentally match a target, and their full results remain available.

### A recorded nontrivial example

Both models succeed on selected pair 1 at visit 1. The receiver graph is
A=(1,0,3,4,5,2), with current cursor 4. The donor graph is B=(2,4,3,1,0,5), with
current cursor 3. The intervention predicts, and both models produce:

    3 -> 4 -> 5 -> 2 -> 3.

The donor's ordinary path is 3 -> 1 -> 4 -> 0 -> 2. The unmodified receiver path
is 4 -> 5 -> 2 -> 3 -> 4. The observed changed continuation follows A from the
transplanted cursor. The example was selected after measurement for illustration;
its target was fixed before the original intervention.

A post hoc cycle-length breakdown records successful transfers on nontrivial
cycles. Seed 201 passes 25/27 cases with cycle length above one; seed 202 passes
16/23. This breakdown is descriptive and does not replace the registered primary.

## Compiled observed traces

After reporting the full panels, the engineering test selects the first valid
natural trace and first valid cursor-transfer trace from each model. It recovers
labels from the saved logits and verifies capture/checkpoint file identities.

Official checksum-pinned Circom 2.2.3 compiles all four witnesses:

| Relation | Constraints | Witness wires including constant | Cases |
|---|---:|---:|---:|
| Six nodes, sixteen natural steps | 845 | 727 | 2 |
| Six nodes, four counterfactual steps | 257 | 223 | 2 |

All four WASM-generated observed witnesses satisfy the compiler's exported R1CS.
The independent checker rejects 1,896 increment-by-one single-wire mutations:
726 per natural witness and 222 per counterfactual witness. It uses the compiler
symbol map and reads the generated WTNS, preserving the public graph and boundary
semantics. The complete compiler artifacts and per-case receipts are retained.

This provides an executed path from trained neural states, through an
intervention with a predeclared symbolic consequence, to a compiled relation
checker. Honest neural execution is trusted. No ZK proof, STWO proof, garbled
verifier evaluation or cryptographic neural-inference proof was generated.

## Independent reproduction and artifact audit

[GitHub Actions run 43032063213](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43032063213),
job `130561519539`, passed the complete checkout: all 81 tests normally and again
under `python -O`, both fixed training runs, and the compiled-witness checks.
CI used Python 3.12 and PyTorch 2.10.0+cpu. The local runtime used the same Torch
version with Python 3.11. The exported initial and final checkpoint files are
byte-identical across the two environments. All saved clean/replayed arrays and
all task/causal metrics also match exactly.

The downloaded CI artifact `13860584885` has SHA-256
`a4c6404713fbc6c21b524b8b57fe644082166ab263e15426bcdfd422f9e9af46`.
Its identity was checked before inspection. Local independent audits reloaded
all four initial/trained checkpoints, recomputed 46,080 clean visit/item outputs,
and replayed all 1,524 intervention arms. Saved logits independently reproduce
the reported labels. The same audit passed on the CI evidence. All 1,896 compiled
mutation rejections were also recomputed after downloading the artifacts.

Final checkpoint identities:

- Seed 201: `983b3e30628f6683d05b59aa68b8cf087b815e69a8505a7c9575e258286bf762`.
- Seed 202: `dc597b7d994c66d97c8645b7613fe5b7d492a7a1e09cda4d38164ddcb398504c`.

Local frozen-manifest SHA-256:
`350aa5267d809def499ca1c975b16d45ab2a0dd04e836e4a5dedc93e3bf752a4`.
The source-pinned training recipe, complete loss histories, optimizer/RNG states,
checkpoints and replay tensors are retained in the downloadable evidence bundle.
GitHub lists the CI artifact expiry as 2027-02-09.

## Interpretation and limits

The measured success cases support the chosen cursor variable as a causal input
to the receiver's learned continuation over the tested horizon. Seed choice
changes its cross-context transferability substantially. An alternative
context-dependent lift can remain relevant to the failure cases.

These are two small models with explicit per-loop supervision, on one finite
family of labeled permutation graphs. Six-node orbits have length at most six,
so sixteen-loop evaluation revisits short cycles. The study measures continued
execution and cursor transfer on held-out labeled maps. It supplies no claim
about arbitrary-length reasoning, hidden language-level CoT recovery, frontier
models, or the original Q4/Q5 panel. No universal causal abstraction or formally
verified neural numerical implementation has been established.

## Reproduce

From this branch's repository root, use an empty output directory:

```bash
python -m research_gym.scripts.mc03_trained_pilot --out /path/to/new-run
python -m research_gym.scripts.mc03_compile_observed \
  --circom /path/to/pinned-circom --run /path/to/new-run --out /path/to/compiled
```

The frozen per-seed wall-time cap is 240 seconds. A slower machine can emit
`BUDGET_STOP`; the receipt preserves its completed-update count and evaluates
that final state. The published local and CI runs completed every planned update.
