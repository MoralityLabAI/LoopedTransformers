# NBR-02A: activation-dependent sparse write reconstruction

Exploratory follow-up to completed registered NBR-01/01B (PR #355).
The 144 boards have already been evaluated; this is a replay of their
archived native activations/Jacobians, not a new confirmatory test.

## Motivation and control

NBR-01 passed and NBR-01B failed in both held-out families. Reuse of
a fixed direction accounts for the observed temporal directional
stability. Test whether the actual activation coordinates nevertheless
provide a useful sparse representation of the write.

At one captured MLP call/token, let a be the vector of SwiGLU scalars,
W the down-projection (columns are w_i) and J the matrix whose i-th row
is the centered nine-logit derivative with respect to a_i. The physical
write is W a and the local first-order transported write is a^T J.
An orthogonal coordinate transformation Q gives b=Qa, W'=WQ^T,
J'=QJ. Both sums and the activation L2 norm are unchanged. This is a
coordinate comparison of one observed write, not an alternative trained
MLP: rotated coefficients need not have independent SwiGLU gates.

Compare native coordinates against eight fixed signed/permuted
block-Hadamard rotations, block width 256 (1536 coordinates total),
seed 20261125021. Rotations are fixed across boards, steps and calls.
These controls preserve the entire physical write and its transported
first-order sum, the number of directions and coefficient L2 norm.
Column norms can change; rank each basis by |a_i| ||w_i||. This selection
does not use labels or logit derivatives. A second diagnostic ranks by
|a_i| ||J_i||; native shuffled physical scores supply an equal-k
unstructured selection null. Report decoder-column norms and sparsity;
verify full-write and derivative-sum parity for every rotation.

## Outcomes

Primary descriptive endpoint: ACT3, L0, k=64; bounded reconstruction
loss ||x-x_k||^2/(||x||^2+||x-x_k||^2) for both physical and transported
writes, using the physical ranking. Smaller is better; undefined if
both norms are zero. Paired native advantage is mean rotated loss
minus native loss. Report per-board intervals and both eval/rep strata
separately, plus calibration. A high physical advantage alone does not
establish task importance; a transported advantage alone does not
establish a finite effect. Secondary curves k=16,32,64,128,256 at
ACT3/6/12, L0/H0, with transported ranking. Report absolute errors,
relative retained coefficient L2 and cancellation diagnostics.
Compare with native shuffled selection at k=64.

Also report eligibility and outcome-class counts at ACT3/6/12 for the
model-blind median blank site, stratified by source difficulty. Use
clean saved predictions and solver answers only for this descriptive
diagnostic. This determines whether a later independently registered
recovery experiment has enough active states. Existing evaluation boards
are now development data for further experimental design.

The replay uses no new forward passes, finite edits or model training.
It tests a different question than Anthropic's concept detection,
which requires labeled concept datasets. Do not claim native basis
specificity from this exploratory comparison. A fresh roster, frozen
decision rule, SAE/PCA controls and real equal-norm interventions are
required for a prospective NBR-02 causal claim.
