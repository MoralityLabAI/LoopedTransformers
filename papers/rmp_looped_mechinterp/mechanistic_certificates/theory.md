# MC-01: source binding across compiled latent computations

2026-11-11. Handwritten proofs with executable finite controls. Existing causal
abstraction and compiler-correctness ideas are attributed below. No novelty claim
is made for semiconjugacy, the pointer relation, or automata minimization.

## 1. Two executions and the information connecting them

Fix the complete neural state h, including all token positions and caches, a
program/input x, a transition F_x and a decoder alpha. A fixed input can remain
an external parameter if every comparison preserves it. Any changing clock or
scheduler belongs to the complete state or the charged input interface.

The proposed symbolic machine has transition delta_x and readout g. A neural
readout G is part of the proposed explanation. For each declared operation u,
including intended interventions, require

    alpha(F_{x,u}(h)) = delta_{x,u}(alpha(h)),
    G(h) = g(alpha(h)).                                      (1)

Both equations must hold on a forward-invariant domain D. Restricting D to a
natural trajectory establishes only that restricted claim. If an intervention
leaves D, its continuation has escaped the certificate's domain.

RC-01 already derives the minimum affine-binary quotient from the closure of
observable rows under all allowed operations. Its equation BA_u = D_u B is an
exact instance of (1). MC-01 places a trace verifier downstream of a candidate
quotient and makes its source-linkage obligation explicit.

## 2. Exact relation and deterministic soundness

Let A be an N by N public Boolean matrix with exactly one 1 in each row. Let
s_0,...,s_T be private one-hot states and p_tij private product wires. Over a
prime field with p>N, impose

    A_ij(A_ij - 1) = 0,           sum_j A_ij = 1,
    s_ti(s_ti - 1) = 0,           sum_i s_ti = 1,
    p_tij = s_ti A_ij,            s_(t+1),j = sum_i p_tij,
    s_0 = initial,               s_T = final.                 (2)

The initial and final vectors are public. The verifier fixes N and T and checks
all public inputs against the requested statement. In MC-01, N<=32 and T<=64;
both supported prime fields exceed N.

**Proposition 1.** A satisfying assignment to (2) contains exactly the unique
T-step pointer trajectory for its public A and initial state, ending at final.

**Proof.** In a field, b(b-1)=0 implies b is 0 or 1. A row sum lies between 0 and N
as an integer. Since p>N, equality to 1 modulo p implies exactly one 1. Thus A
encodes a total function and each s_t a single node. If s_t selects i, the product
constraints force p_tij=A_ij and all products from other rows to zero. The next
state is exactly A's selected row. Induct on t from the constrained initial
state. The last equality binds its result to final. Conversely, the actual
trajectory and its products satisfy every constraint. QED.

The two-field checks verify this relation on finite examples. An M31 check has
no STARK proof attached. Compiling Circom requires a separate source-to-R1CS
parity gate, since the intended equations alone do not verify a compiler.

**Proposition 2 (trusted-source transfer).** Suppose an external trusted capture
supplies h_0,...,h_T from the declared F_x. Let a_t=alpha(h_t), with the initial
boundary bound to x. If the verifier accepts (2) and independently checks
s_t=onehot(a_t) for every t, the observed labels follow the pointer machine.
If G=g composed with alpha holds on these states, the same statement applies to
G's observed outputs. Extending it to counterfactual continuations requires (1)
on the corresponding intervention-closed domain.

**Proof.** The equality to captured labels and Proposition 1 identify each a_t
with the unique symbolic state. Substitute this identity into the readout
factorization. Equation (1), when it holds on a forward-invariant domain, gives
the same identity inductively after every declared intervention. QED.

The implementation's q0 is the task's input boundary. Its q1..qT are the captured
native head's argmax labels. Input anchoring supplies no claim that the all-zero
initial residual already represents the start node through that head.

## 3. What a verifier wrapper can establish

Write V(x,w) for acceptance of the abstract relation. Suppose a compiled or
encoded checker V' preserves its verdict on encoded inputs. Correct recursive
verification or correct garbled evaluation then establishes the same acceptance
event. Additional proof-system errors must be accounted for at their layers.

**Proposition 3 (source indistinguishability).** If a verifier sees only (x,w),
and two neural histories H and H' supply the same (x,w) while differing on a
claimed source property P, no decision function of that verifier transcript can
correctly distinguish P on both histories. Applying any further deterministic
wrapper to the transcript preserves this obstruction. A randomized wrapper has
the same output distribution for both identical input transcripts.

**Proof.** Equal inputs give equal outputs for a deterministic function. Conditional
on each random seed, the same equality holds; averaging gives equal distributions.
Since P differs, at least one history receives an incorrect common verdict. QED.

The executed control uses A=(2,2,1), start=0 and T=3. Its true symbolic trajectory
is (0,2,1,2). The different observed sequence (0,1,1,2) has the same final answer.
Submitting the oracle trajectory passes (2). The externally supplied observation
comparison rejects it. A hash provided by the same party as both files cannot
settle which history happened. The current importer trusts independently pinned
SEM-ECHO captures and states that trust in its output.

A cryptographic extension needs a relation that binds the model and input,
reproduces specified numerical inference, applies the specified decoder, and
connects those outputs to the abstract trace. A circuit identity or a model hash
introduced as an unconstrained public signal does not add that relationship.

## 4. Successful traces can still track an unused register

Let h=(a,b) be two bits. Natural execution flips both and starts on the diagonal
D={(0,0),(1,1)}. Take alpha(h)=a and actual output G(h)=b. On D, the symbolic rule
q -> 1-q agrees perfectly and the readout factorization holds.

Now permit the low-level intervention a:=0 while preserving b. From (1,1) it
reaches (0,1), where G=1 and g(alpha(h))=0. The proposed explanation fails on the
intervention closure. MC-01 detects the readout defect on the expanded domain;
a diagonal-only audit reports domain escape as soon as the intervention is added.

This control specifies why a valid decoded trace still needs causal experiments.
The separate positive control reads a itself and permits writes only to the
nuisance coordinate b, preserving both equations on every state.

## 5. Representation changes and loop-dependent codes

Let E_t be a bijection of the complete concrete state at visit t. Define

    F'_t = E_(t+1) composed with F_t composed with inverse(E_t),
    alpha'_t = alpha_t composed with inverse(E_t).

**Proposition 4.** For every h, the commutation defect after this change of
coordinates equals the original defect at inverse(E_t)(h), under any fixed
metric on the abstract states. Exact commutation is therefore invariant.

**Proof.** Substitute the two definitions into alpha'_(t+1)(F'_t(h)); the adjacent
inverse(E_(t+1)) and E_(t+1) cancel. The other branch becomes
 delta_t(alpha_t(inverse(E_t)(h))). These are precisely the original two terms.
QED. Interventions and readouts must be transformed through the same E_t.

Consequently a changing latent code can defeat one fixed decoder even when an
appropriate visit-dependent decoder exists. Its clock and decoder description
must enter the resource ledger. Cryptographic wire encodings provide one exact
example of semantics carried in unfamiliar coordinates. This statement supplies
no evidence that a learned transformer uses cryptographic encryption.

## 6. Finite error and what held-out measurements justify

For a random length-T trajectory, let B_t be the first violation of the chosen
commutation/readout obligation at step t. If, for each t, its conditional
probability given that all previous obligations held is at most epsilon_t,
then

    Pr(any violation by T) <= min(1, sum_t epsilon_t).         (3)

The first-failure events are disjoint. Each has probability at most epsilon_t
because its conditioning event has probability at most one. Summing proves (3)
without an independence assumption. A sharper product bound follows when the
same conditional-hazard bounds apply at every step.

A mean one-step defect measured under a calibration distribution does not supply
these conditional bounds on later deployment states. Tests at held-out depths
must measure the visited distributions. Overlapping pairs from one trajectory
remain dependent; semantic-echo's item-level aggregation is retained.

## 7. Description costs and experimental contract

A decoder that runs the entire symbolic program from x and the loop number can
manufacture a correct trace without reading h. Charge its computation and its
description. The initial MC-01 decoder is the model's native head, with no oracle
query inside it. The exact oracle is reserved for scoring and constraints.

A useful certificate budget includes decoder description, symbolic transition
description, intervention-lift description and live abstract state. RC-01's
intervention-relative lower bound applies when its finite-state hypotheses hold.
The one-hot arithmetization expands state bits for convenient constraints; its
wire count is not an intrinsic minimum description of the hidden algorithm.

The target hypothesis is compact, causally faithful transition semantics across
held-out depths. This step provides its source-aware audit and counterexamples.
Fitting an abstraction, measuring causal patch predictions on real Q4/Q5 traces,
and proving neural inference are distinct outstanding experiments. A native-head
rejection only rejects the fixed one-hop-per-loop candidate.

## Sources and prior implementation

- Geiger, Lu, Icard and Potts, Causal Abstractions of Neural Networks (2021),
  https://arxiv.org/abs/2106.02997 . Causal abstraction and interchange interventions.
- Circom: constraint generation and exported constraint format,
  https://docs.circom.io/circom-language/constraint-generation/ and
  https://docs.circom.io/circom-language/formats/constraints-json/ .
- Circom 2.2.3 official release and asset digest,
  https://github.com/iden3/circom/releases/tag/v2.2.3 .
- StarkWare, STWO/Cairo overview,
  https://github.com/starkware-libs/stwo-cairo . The repository announces migration
  to starkware-libs/proving. MC-01 has no STWO dependency or executed STWO backend.
- Compactification-Theory at 59d8a37919676469e3790bd66dfae33e5a3c1f44:
  research/recurrent_circuits_v1/THEORY.md and research/algernon/circuit_mirror/README.md.
  RC-01 owns the minimum affine quotient; Algernon-02 owns the earlier pointer
  relation and neural mirror experiments. MC-01 makes no new claim about them.
- LoopedTransformers at 80d2ce8d073c6b634eab56f94c3f24c1a3c3a9e1:
  papers/rmp_looped_mechinterp/semantic_echo and
  research_gym/scripts/loopscope_semantic_echo.py . Full-state capture contract.
