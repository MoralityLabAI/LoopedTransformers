# MIC-18: exact recurrent memory after a delayed query

This CPU calibration implements a fixed recurrent ReLU cell that stores J
binary facts as they arrive. A coordinate query arrives after the state has
been committed. The query head receives that state and the query index.
Training updates and trained checkpoints both equal zero.

## Task and memory requirement

For a word x in {0,1}^J, observations arrive in MSB-first order. After the Jth
update, query q asks for x[q]. Public phase supplies the address of the fact
being written. Each pair of different words differs at some coordinate, so
the corresponding delayed query distinguishes their required responses.
A deterministic zero-error controller therefore requires at least 2^J final
states, or J binary state bits. Parity has a two-state streaming controller.
These statements concern the declared task and information interface.

The pinned EvalMath MIC-18 evaluator replays controllers compiled from the
actual neural matrices. The generated transition tables belong to the
verification harness. The running cell uses fixed weights and carries its
binary vector between updates. Its inference code has zero codebook lookups.

## Fixed neural cell

With binary state h, incoming bit b, and public one-hot write gate g:

    h'[i] = ReLU(h[i] - g[i]) + ReLU(b + g[i] - 1)

The matrix implementation has 2J ReLUs per update. Its update weights and
biases use {-1,0,1}. A one-hot query gate q drives the head:

    answer = sum_i ReLU(h[i] + q[i] - 1)

`register.py` evaluates the integer affine matrices and activations directly.
The protocol checks every binary state, phase and input bit for J=1..8,
including states that are outside the clean prefix support of that phase.
It also replays every complete word and every final query.

### Physical budget

The register, padded parity and padded last-bit arms all allocate J binary
coordinates, execute J updates, use 2J update ReLUs and use J readout ReLUs.
Their dense matrix shapes match. The interpreter executes zero coefficients
as well. Each complete episode with one query costs 2J^2+J ReLU evaluations
and 6J^3+4J^2+J coefficient products. Padding ensures the declared dense work
and allocation agree across the arms; their reachable states still differ.

The compact parity control carries one bit and uses two update ReLUs per
observation. Its cost is reported separately. Padded parity uses coefficient
-2 in its XOR update. The register requires coefficients of magnitude at
most one. Both updates use exact integer arithmetic on the finite domain.

Every program reports static coefficient slots, nonzero entries and dynamic
state allocation separately. A register update has 6J^2+5J dense coefficient
slots, including biases. The query head adds 2J^2+2J+1 slots. These counts
describe the logical architecture; Python object storage and source-code
bytes are outside that measure. Public phase has J+1 possible values through
commitment and would cost ceil(log2(J+1)) bits if stored. The one-hot write
gate has J coordinates. It supplies an address and carries zero private facts.
Query index and its one-hot expansion are public after commitment.

## Bridge to LS-NEURON-12

[LS12 at its pinned source](https://github.com/MoralityLabAI/LoopedTransformers/blob/75aeb9ae914636a4d70b576e331fbfb72c0daf7d/mechinterp/ls_neuron12/README.md)
constructs finite automata with explicit neural width, code and disturbance
assumptions. Its 64-state binary construction has 24 GELUs, and its primary
15% recovery gate failed. MIC-18 supplies a task whose future responses force
2^J distinguishable states, then exhibits an integer ReLU implementation with
J binary coordinates and 2J update units. A state-count lower bound becomes a
storage requirement only under a stated precision contract.

This receipt concerns the supplied matrix construction. Global minimum
neural width, finite-precision GELU behavior, hardware rounding guarantees,
pretrained checkpoints and learned circuit recovery require their own work.
The binary state contract here is exact; this calibration introduces no
perturbation radius.

## Evidence and replay

`PROTOCOL.json` fixes widths and checks before the main execution.
`SOURCE_LOCK.json` pins source bytes and provenance; `RESULTS.json` records
the exact counts and resource accounts. `paths.jsonl.gz.b64` contains every
word trajectory and query output plus both parity endpoints. Each replay
checks the decoded archive contents, which avoids relying on a particular
gzip implementation. Matrix truth tables are independently compared with
coordinate overwrite and XOR oracles during every replay.

Run from the repository root with Python 3.10 or newer:

```sh
python -m unittest discover -s mechinterp/mic18_online_memory/tests -v
python mechinterp/mic18_online_memory/run.py --check
```

The CI workflow uses Python 3.11 and standard-library dependencies. No model
weights are downloaded. The inherited EvalMath mechanistic-resolution
guidance classifies this census as an exact finite result. Independent
trained development roots, statistical power and a sampled sham quantile are
absent. `population_gate` records that boundary explicitly.

`CORE_LICENSE.txt` retains EvalMath's Apache-2.0 license for the byte-identical
`finite_memory.py` copy. The register cell and adapter code were authored for
this port. The adjacent BPD port reuses the same original register bytes for
an exact donor-and-mask calibration.
