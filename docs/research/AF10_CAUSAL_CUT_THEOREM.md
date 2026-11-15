# AF-10: a structural causal-cut theorem for recurrent Horn proof state

**Status:** new mathematical and mechanistic analysis of frozen AF-09 source. Its numeric replays are diagnostic, since the held-out graph families had already been consulted in AF-05–09 and during selection of this intervention. The structural theorem does not depend on those observations.

## 1. The coupling AF-06 through AF-09 were optimizing against

The source-hash-verified `TypedLoop` first updates each rule token with **all** of its head and premise atom representations, then returns messages from that rule to every incident atom. For a downstream Horn conjunction with both Q0 and Q1 as premises, the shared rule receives both Q codes and is subsequently returned to both premise atoms. As a consequence, the Q0-to-Q1 neural Jacobian block need not vanish, even when the two upstream Horn proofs are logically independent.

This is a property of a perfectly valid graph-attention architecture. The model's learned causal invariance has previously been implemented as an approximate training loss. AF-10 tests a structural alternative.

## 2. The causal-cut operator

For a model with compressed atom codes `z_A` and rule codes `z_R`, define the actual downstream recurrent computation as

\[
  F(z;\mathcal G)=E\circ C_{\theta}\circ D(z;\mathcal G),
\]

where `E` and `D` are the actual tokenwise encoder and decoder and `C` is the original native cell with the public downstream graph. Let `P` select exactly the two derived-subgoal atom slots in the **actual carried code**, with all physical permutations resolved using public role positions.

Write `z^0` for the carried code after the upstream computation and any donor transplant. Define the *cut recursion*

\[
  z^{t+1}=(I-P)F(z^t;\mathcal G)+Pz^0,\qquad t\ge0.
\]

The implementation restores `Pz^0` after each native cell and code encoding, then decodes the restored codes before the next native call. This prevents the decoder's transient values from reentering the subsequent state. Every step executes the **same original TypedLoop cell**, preserving differentiability of the downstream goal path.

### Theorem A: exact no-cross-talk transport

For every choice of weights, graph, donor, recurrence length, and initial state, the cut recursion satisfies

\[
 Pz^t=Pz^0\qquad (t\ge0).
\]

Suppose receiver and donor share the same semantically untouched subgoal `Q_(1-j)` and the transplant changes only `Q_j` at the cut. Then the protected subgoal's code equals the recipient's source code for both clean and patched trajectories **at every recurrence step**. Its logit vector under the original affine atom decoder and status head is identical in both trajectories. Thus the protected-Q readout is invariant under any selected-Q transplant, and its cross-subgoal derivative vanishes identically.

**Proof.** The equality holds at `t=0`. Multiply the recurrence by `P`, use `P(I-P)=0` and `P²=P`, and induct. The clean and patched runs have the same protected entry of `Pz^0`; the role-selected atom decoder and Q head operate exclusively on that token. This gives exact equal logits. The map from selected initial Q code to protected Q code is constant, hence its derivative is zero wherever the readout is differentiable. ∎

### Theorem B: compositional prefix-state sufficiency

Let `q^r` be the original receiver Q0/Q1 Horn truth and `q^d_j` the donor's changed selected-Q truth, with `q^d_(1-j)=q^r_(1-j)`. Let `q*=(q^d_j,q^r_(1-j))` in address order, as defined by the exact independent Horn interchange oracle.

If the recipient prefix predicts `q^r_(1-j)` correctly and the donor prefix predicts `q^d_j` correctly, the addressed cut **must** predict both entries of `q*` correctly after every downstream iteration. This follows immediately from Theorem A and the identical role-selected affine Q readout. The downstream goal prediction remains a separate calculation and may be incorrect even when the proof-state interface is correct. ∎

Let `U` denote the event that either required prefix Q prediction is wrong, and let `V` denote a wrong patched goal prediction. The exact error decomposition is

\[
\Pr(V)=\Pr(V\cap U)+\Pr(V\cap U^c)
\le \Pr(U)+\Pr(V\cap U^c).
\]

This is useful because the first term measures *upstream semantic inference*, while the second isolates *goal-rule computation conditional on correct input proof state*. Under the causal cut, the intervened protected-Q transport error is structurally zero. This sharpens the previous attribution of accuracy losses to a mixture of code drift, wrong initial meaning, and goal readout failure.

### Consequence: AF-08/09 margin certificate simplifies

For the protected Q, `delta z=0`. AF-09's affine code-margin lower bound therefore reduces **exactly** to the prefix clean true-class margin:

\[
m_y(z^{t}_{Q_{1-j}})\ge m_y(z^{0}_{Q_{1-j}})-K_y\|0\|=m_y(z^0_{Q_{1-j}}).
\]

Equality holds. A positive source margin certifies correct protected-Q classification at all downstream steps, while a wrong or uncertain prefix remains wrong or uncertain. The cut is not a proof of Horn truth, independent Lean theorem discovery, global recurrent reversibility, or correct downstream goal semantics.

## 3. Tests and reproducibility

The source-bound AF-10 evaluator uses original frozen AF-09 `status`, `locality`, and `semantic_margin` checkpoints for seeds 17 and 29; exact SHA-256-pinned training-artifact ZIPs and Torch/NumPy checkpoint receipts are verified. The test runs all 5 existing AF-05 data families, 3-/5-step suffixes, both changed-status donors per case, and the original Horn oracle. Every paired record retains predicted Q and goal logits, the exact role-selected code, and the source/counterfactual labels.

The independent NumPy verifier reconstructs each protected and selected Q readout as

\[
 R(D(z))=W_R(W_Dz+b_D)+b_R
\]

from the saved model tensor arrays and checks stored logits against those numbers, in addition to recomputing the exact Horn donor and truth roster. Checkpoint-bound Torch replay samples verify the actual output arrays. Tests also evaluate the original-cell Q-to-Q Jacobian and gradient flow through the downstream goal.

The preselected data have been examined in earlier AF campaigns. Changes in their accuracy are **exploratory mechanism diagnostics**, not preregistered out-of-distribution superiority results. A scientific efficacy test of training with the latch will require fresh held-out graph families and a separate frozen registration.

## 4. Why a frozen latch can reduce goal accuracy

The frozen model was fitted with ordinary recurrent feedback. Enforcing a typed latch at inference changes its computation. Its downstream goal head may depend on Q-code dynamics, intermediate rule state, or recurrent self-interactions. Accordingly, a model can gain protected-Q correctness and lose goal correctness, or the reverse. The AF-10 evaluator records both measurements and conditional goal accuracy given correctly inferred prefix Qs.

## 5. Next controlled experiment

Train identical qualified AF-06 source checkpoints under a typed causal-cut downstream stage, with explicit upstream prefix Q status loss and the exact composed Horn goal loss; compare against a matched recurrent-call baseline. Fix a new, deeper graph family before optimization and use checkpoint-to-data hash closure so previously examined AF-05/06/07/08/09 holdouts remain diagnostic only. Report original proof-target Lean checks separately from neural latent-state behavior.
