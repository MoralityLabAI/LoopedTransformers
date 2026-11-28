# MOLT-17: independent exact finite-dose checker

Replay the MOLT-16 development roster with the same frozen teacher and
controller recipe. Export the actual binary64 cut residual, legal head,
LayerNorm parameters, support chart, raw displacement, radius, target
winner, numerical dual weights and primal coefficients as hexadecimal
dyadics. Bind the export to the teacher, factors, controllers and roster
hashes. This is an exactness audit of the same panel, not a new independent
performance experiment. Parent numerical status is never authority.

## Exact arithmetic

The standalone checker uses Python fractions and integer square roots;
it imports neither Torch nor NumPy. It treats exported finite binary64
values as exact real constants. The dose limit is R+1e-8, matching the
parent numerical tolerance, and it verifies the raw displacement fits
that enlarged physical ball. Numerical QR charts need not be perfectly
orthonormal: the checker uses the exact Gram inverse of the exported Q.
If centered columns are dependent, it uses the conservative centered
ambient envelope instead. Centering contracts physical displacement,
so the denominator envelope uses reach <= R+1e-8.

For squared nonnegative rational x, calculate
k=isqrt(floor(x*2^(2b))). The dyadic k/2^b and (k+1)/2^b enclose
sqrt(x); use equality when exact. Every square-root bound is checked by
rational squaring. b=80.

For a nonnegative dual lambda, form the exact weighted necessary
threshold L and centered ambient contrast g. Over the physical ball in
range Q the maximum of g dot d is R sqrt(S), where

    S = (Q^T g)^T (Q^T Q)^(-1) (Q^T g).

A strict obstruction follows from L>0 and L^2>R^2 S. No approximate
orthonormality, numerical dual convergence, or approximate square root
is assumed. Primal verification independently reconstructs the exact Qc,
checks its physical squared norm and every strict legal contrast with
outward rational denominator enclosures. A contradiction aborts.
Unverified cases stay unresolved. The complete paired chart roster is
required, and all witnesses and rational comparisons are retained.

The certificate applies only to the declared immediate legal winner
and the exact real-valued LayerNorm readout formed from the exported
constants. It does not certify that a floating-point execution has
identical outputs, the origin of the export, unrestricted vocabulary
argmax, future suffix accuracy, or a hard logical deduction. The proof
is an ordinary mathematical argument plus executable exact arithmetic,
not a Lean theorem. In particular the replayed source and witness export
must be trusted or separately audited. No new model training effect,
confirmation, or promotion claim.

The workflow repeats the parent fitting recipe, exports witnesses in
a bounded process, then checks them in a separate dependency-free
worker under the existing address-space/CPU and sampled RSS guards.

    python -m pytest -q tests/test_exact_carry_certificate.py
    python -m lsa.exact_carry_certificate --problems problems.json --out exact-receipt.json

The standalone verifier can be used by DataGoblin as a conditional
decision-admission check with source identity verification kept separate.
