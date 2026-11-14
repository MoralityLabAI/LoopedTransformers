# AF-02: addressed continuation in the native TypedLoop cell

AF-02 tests whether a compressed carried state preserves branch status and the
effects of interventions during repeated computation. It trains both preview
orders and adds waits to the AF-01 candidate-selection gym. A streaming adapter
imports the repository's original `TypedLoop` class from checksum-pinned source.
The AF-01 GRU provides a second architecture under the same data and training
schedule.

The experiment was registered in [PROTOCOL.json](PROTOCOL.json) before AF-02
optimization or scored evaluation. Registration commit:
`7b3e292e93e977918ed4903e459cdf9085c04d49`. The protocol's SHA256 is
`f6c3a939a09183c265466d506de985ec1a92c76d2d57f50cdd5f3f984e224c41`.
This subtree builds on AF-01 in PR #34. The separate FC-12 tactic-policy branch
has its own experiment and evidence.

## Scope and mathematical objects

Each input contains previews of two candidate branches. A branch is valid,
dead, or tainted. Tainted candidates depend on an unauthorized extra fact.
The policy chooses the smallest valid branch, or STOP when neither branch is
valid. The host also accepts a legitimate proof from the other valid branch.
Those two outcomes have separate scores.

The host binds the original theorem, candidate traces, preview order, schedule,
budget, and certificate identity. Lean checks submitted proof terms against the
original theorem. The neural model receives public F9 receipts and depth
metadata. Its assigned task is to remember previews and choose a candidate;
the full Horn-chain derivation belongs to the host and certifier.

For event \(x_t\), each of its two recurrent visits computes

\[
u_{t,k}=F_\theta(h_{t,k-1},x_t),\qquad
z_{t,k}=E_\phi(u_{t,k}),\qquad h_{t,k}=D_\phi(z_{t,k}).
\]

The teacher carries \(u\) directly. Both codec arms freeze the trained teacher
and insert the codec after every call, including every wait and goal visit.
The task codec fits action cross-entropy and teacher KL. The joint codec also
fits status labels, composed-prefix continuations, and a decoded commutation
loss. The two codec arms share train-only PCA initialization and sampling RNG.

Native tokens use canonical order `[A0, A1, Goal, R0, R1]`. Public rules are
`A0 -> Goal` and `A1 -> Goal`. A preview's supplied address routes its payload
to the corresponding atom and rule. The payload writer itself receives zero
in the address coordinates. Wait and goal events write the goal atom. The
original cell then updates all tokens through the public graph. This supplies
a concrete branch-binding mechanism whose use can be measured.

## Frozen data and fitting

All nine status pairs occur in both orders. Every canonical input has two
renamed host certificates. Training samples canonical inputs uniformly;
different sequence lengths execute in separate buckets, with bucket losses
weighted by their sampled row counts. Each reported metric gives equal mass
to each distinct public input. The two host copies supply certificate checks.

| Split | Depths | Event schedules | Canonical inputs | Host cases |
|---|---|---|---:|---:|
| train | 2, 3 | PPG, PWPG, PPWG | 108 | 216 |
| dev | 2, 3 | WPPG | 36 | 72 |
| depth_context | 4, 5 | PPG, PWPG, PPWG | 108 | 216 |
| delay2 | 2, 3 | PPWWG | 36 | 72 |
| delay4 | 2, 3 | PPWWWWG | 36 | 72 |

`P` reveals the next branch, `W` waits, and `G` supplies the final goal marker.
The action budget is `3 + number of W events`. `P` and `W` each cost one; `G`
is an observation; the final submission or STOP costs one.

[corpus_manifest.json](corpus_manifest.json) binds all generated inputs and
host cases. Its input-hash sets are disjoint across splits. Primary training
accesses train and dev only. It uses seeds 17, 29, and 43, with 400 teacher steps
and 300 steps per codec, batch size 64, AdamW at 0.003, weight decay 0.01, and
gradient clipping at 1. The runtime is Torch 2.5.1+cpu, float32, and two threads.
Checkpoints come from the fixed final step. Dev determines eligibility.

A source qualifies only if task accuracy and each branch's status accuracy
reach 95% separately for both orders on both train and dev. Every seed remains
in the evidence, including an ineligible source.

## Resources

The adapters differ in width, storage, and arithmetic. Equal optimization steps
define the registered comparison. The table counts a three-event input with
two calls per event.

| Quantity | GRU | Native TypedLoop adapter |
|---|---:|---:|
| Teacher parameters | 4,425 | 5,590 |
| Stored public buffer floats | 0 | 12 |
| Codec parameters | 292 | 296 |
| Decoded working coordinates | 32 | 80 |
| Carried codec coordinates | 4 | 20 |
| Float32 bits at codec boundary | 128 | 640 |
| Teacher dense MACs | 23,904 | 88,562 |
| Teacher plus codec dense MACs | 25,440 | 92,402 |

The native codec shares one 16-to-4 encoder/decoder across atom tokens and a
second across rule tokens. Retained code arrays precede decoding. Compression
reduces the carried representation while adding computation at every visit.
Dense MAC estimates exclude nonlinearities, indexing, framework overhead, and
Lean. Timing records use one warm-up followed by one block of three forwards
for each actual sequence length.

## Interventions and controls

The patch boundary follows the second visit of the first actual preview.
Donors share the first branch, public depth, schedule, and graph layout.
Deduplicating observed prefixes leaves three donor prefixes per stratum.

The whole-state intervention replaces the actual carried code, decodes it
once, and resumes the recipient's remaining events. Native runs additionally
patch the addressed atom/rule pair, the other pair, or the goal token. A self
patch tests replay. The artifact records the token mask, patched code, decoded
initial state, resumed trajectory, raw logits, and direct composed-history
output.

Counterfactual labels belong to the composed public history. These records
have their own causal scores and earn no original-host proof or attack credit.
Whole-state semantic scores deduplicate composed histories; localization uses
the exhaustive changed-prefix pair roster.

Clean runs cover all 18 architecture/seed/arm combinations. Teacher controls
erase the branch address or zero carried state between calls. Payload-fixed
companions test whether action choices and audit slots follow supplied branch
addresses. The exact address-erasure reference is retained in
[address_erasure_reference.json](address_erasure_reference.json). An address-blind
policy has at most 50% action accuracy on exactly-one-valid inputs.

All 12 physical permutations of the three atom slots and two rule slots move
the public graph, routing, and readout interpretation consistently. Each input
retains its worst logit discrepancy. This check tests the adapter's transport
of public structure.

## Registered judgments

Three registered hypotheses use the following requirements:

| Hypothesis | Required evidence |
|---|---|
| Protected benefit | On depth_context, an eligible joint codec gains at least 0.05 exact status accuracy over its task codec, loses at most 0.02 task accuracy, and adds no tainted attempts. |
| Delayed preservation | In every held-out family, an eligible joint codec remains within 0.02 of its teacher's task accuracy, reaches at least 0.95 exact status accuracy, and adds no tainted attempts. The teacher must reach 0.95 exact status accuracy in that family for the comparison to apply. |
| Addressed causal preservation | For native depth_context changed-prefix pairs, an eligible teacher must reach 0.95 clean exact status accuracy and an addressed-versus-other patch selectivity gap of at least 0.20. The joint codec must retain first-branch target accuracy, other-branch retention, and the selectivity gap within 0.05 of that source. |

An applicable failure makes the aggregate fail. Every required stratum must
pass for an aggregate pass. Remaining cases are inconclusive. The CI exit code
reports engineering integrity; the scientific verdicts remain in the receipt.

## Local inverse diagnostic

For the native teachers, 18 depth-4 PPG histories supply the retained state
after both previews. Frozen float32 states and weights are lifted to float64.
The update \(F\) is one goal-writer/native-cell visit. Horizons one and two
across three seeds yield 108 Jacobians.

Each token's tangent is orthogonal to `1 / LayerNorm.gamma`. With five tokens,
the combined tangent has 75 dimensions when all gains exceed the registered
minimum. For bases \(Q_{in},Q_{out}\), the diagnostic uses

\[
B=Q_{out}^{\mathsf T}JQ_{in},\qquad
\widehat\delta=Q_{in}B^+Q_{out}^{\mathsf T}
\left[F^q(h+\delta)-F^q(h)\right].
\]

The pseudoinverse cutoff is `1e-6`; a second rank uses `1e-8`. Four seeded
tangent directions use absolute L2 amplitudes `1e-4` and `1e-3`, shared across
horizons. Records include predecessor displacement error, forward-return
error, readout agreement, ranks, and singular values. Controls include zero
recovery, the least singular tangent direction, and a constant 0.125 shift
before LayerNorm. Independent centered finite differences check all 80 ambient
coordinates as well as the four registered tangent directions. Their retained
matrix provides a numerical cross-check of the tangent spectrum.

This evaluates local recovery around a known reference. The saved ambient
Jacobian and all perturbations allow an independent numerical audit. LayerNorm
removes a constant pre-normalization direction, which constrains the scope of
claims about reversing the loop.

## Reproduce

Run commands from the repository root using Python 3.12. Install the CPU build
of Torch 2.5.1, the requirements here, and Lean 4.19.0. The workflow pins the
official Linux Lean archive checksum.

```bash
python -m unittest discover -s experiments/activation_funnel_af02/tests -p 'test_*.py' -v
python -O -m unittest discover -s experiments/activation_funnel_af02/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af02.train --out artifacts/af02/checkpoints --threads 2
python -m experiments.activation_funnel_af02.evaluate --checkpoints artifacts/af02/checkpoints --out artifacts/af02/evaluation --threads 2
python -m experiments.activation_funnel.certify all --episodes artifacts/af02/evaluation/test_episodes.jsonl --outcomes artifacts/af02/evaluation/outcomes.jsonl --out artifacts/af02/kernel --lean lean --timeout 120
python -m experiments.activation_funnel_af02.inverse --checkpoints artifacts/af02/checkpoints --evaluation artifacts/af02/evaluation --out artifacts/af02/inverse
python -m experiments.activation_funnel_af02.audit --checkpoints artifacts/af02/checkpoints --evaluation artifacts/af02/evaluation --kernel artifacts/af02/kernel --inverse artifacts/af02/inverse --out artifacts/af02/independent_audit.json
```

Each output directory must be fresh. The trainer retains 18 Torch checkpoints,
18 named NumPy exports, and all 2,592 train/dev prediction rows before scoring.
Evaluation retains 10,800 original-host outcomes and 29,160 counterfactual
records, along with all trajectory and layout arrays. The independent replay
uses NumPy/SciPy implementations and the exported arrays. Artifact rosters,
source hashes, oracle arithmetic, Lean receipts, and numerical controls gate
the final audit.

Implementation map: `gym.py` supplies the fixed task and data contracts;
`model.py` provides streaming adapters and codecs; `train.py` freezes fitting
evidence; `evaluate.py` stores held-out trajectories and judgments; `replay.py`
implements independent arithmetic; `inverse.py` measures local recovery; and
`audit.py` checks the retained experiment. The original AF-01 certifier provides
the Lean boundary.
