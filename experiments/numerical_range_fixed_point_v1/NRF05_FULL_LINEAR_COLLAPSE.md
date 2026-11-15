# NRF-05: full-linear-observation no-go on twelve trained loop checkpoints

Date: November 15, 2026. The original NRF-04 certificate established full closure for nonempty **coordinate** observations. NRF-05 strengthens this to **every nonzero linear observation**, including the rank-eight task-supervised subspaces in the frozen NRF-04 semantic experiment.

## Exact checkpoint results

All twelve SHA-256-pinned tied residual-loop checkpoints meet the following stronger hypotheses:

- W1 is nonsingular over the rationals and reals, witnessed by full rank over F_67108859;
- W2 is nonsingular, witnessed over both F_67108859 and F_65521;
- every entry of W1W2 is exactly nonzero, witnessed by all 16,384 entries of the scaled integer product being nonzero modulo F_67108859.

That is **196,608 nonzero matrix-product residues**, with twelve independent checkpoint-pair evaluations. See [nrf05_full_linear_collapse.json](nrf05_full_linear_collapse.json) for the complete determinants, SHA-256 provenance, and matrix-product support receipts.

The exact binary32 scaling is integer arithmetic: every stored weight is an integer multiple of 2^-149. Reducing those integers modulo the prime witnesses real nonzero determinants and nonzero entries of W1W2. Finite-field evidence is one-way: a failed modular determinant test would require a different modulus or a more general check.

## Full-state-space theorem

For any one of these saved matrices, define the real map

    F(x) = alpha*x + beta*W2*tanh(W1*x+b1) + beta*b2,

with nonzero beta on all R^128. Because W1 is invertible, every activation-gate diagonal diag(sech^2(W1*x+b1)) can vary independently over (0,1]^128. Since W2 is invertible, a nonzero observation row activates at least one rank-one gate. Nonzero entries of W1W2 then propagate that rowspace to *every* row of W1 in a second gate step. As W1 is invertible, the smallest exact observation-invariant rowspace has dimension 128.

Thus **every exact deterministic quotient by a nonzero linear observation on all of R^128 requires rank 128**. The global uniform differential fiber-leakage floor in MathCan's [FULL_LINEAR_GATE_COLLAPSE.md](https://github.com/patrickdugan/MathCan/blob/research/crouzeix-fixedpoint-switching-20261115/research/compactification/FULL_LINEAR_GATE_COLLAPSE.md) is strictly positive for every proper projector, with an explicit expression in singular values and the smallest nonzero interaction entry.

This theorem is about *exact, global, linear* factorization. It leaves approximate reduction on data-supported regions, nonlinear observations, and task-conditioned stable quotients as distinct research questions. It does not prove that any observed local leakage causes capability degradation.

## Reproduction

From the private repository root, with Python 3.12+ and PyTorch:

    python -m unittest tests.test_nrf05_full_support_modular -v
    python -m research_gym.analysis.nrf05_full_support_modular

The verifier calls the existing NRF-03 SHA-256-pinned loader and compares its computed modular determinants and every product-support count with the committed NRF-05 reference. All model weights remain unmodified.

The linked GitHub Actions workflow currently fails before runner setup on this private repository, so hosted verification is pending. The existing connector calculation independently decoded the original twelve checkpoint byte streams and computed the modular witnesses with integer-exact JavaScript arithmetic (all intermediate products below 2^53 for modulus 2^26-5); the Python script supplies a separate implementation suitable for local replay.

## Research continuation

Run a bounded **reachable-state** or teacher-conditioned quotient test: freeze an observation dimension and estimate the worst upper envelope of leakage on the actual hidden-state distribution, with held-out task performance and matched random subspace controls. A global exact quotient is ruled out under the established assumptions; a useful approximate quotient can still exist on a narrow distribution. A subsequent intervention should train toward low *regional* leakage while preserving held-out task utility, with an independent seed and precommitted acceptance criteria.
