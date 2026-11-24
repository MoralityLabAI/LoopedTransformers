# NAN-BANKGAP-100 joint BF16 rounding of nearly equal fit actions

Use the fixed BANKDOSE99 COMP14 CAL and exposed DEV archives. Work on
the same native donor input, 1% nominal residual dose, shared SVD
initializer and two fitted physical operators. Evaluate reachability
for both signs in both worlds before any new native forward.

Use the original independent rounded rays as a custody control. Also
apply SHELLREPAIR90's complement fill independently to both full fits,
and once to the shared SVD carrier. It holds the jump coordinates on
one side of the dose jump and fills the remaining dose off those
coordinates. For the shared-carrier arms, transport only the small
T1/T2 offsets relative to the SVD action onto the common rounded
carrier. Keep the ideal T1/T2 midpoint fixed for the gain ladder
1, 2, 4 and 8. Gain 1 targets the original differential. The higher
gains amplify that differential and have separate denominators.

Compare separate nearest rounding and joint rounding at each gain.
Joint rounding enumerates the nearest BF16 value to each requested
endpoint and its immediate finite BF16 neighbors. At each coordinate,
minimize squared error in the T2-minus-T1 differential over the nine
endpoint pairs, breaking ties by smaller total distance to the two
unrounded endpoints and then fixed traversal order. The summed
minimum is the exact differential-error bound within that finite
stencil, conditional on the selected carrier. Mark a stencil
incompatible if its error lower bound exceeds the largest normalized
error allowed by the 3% differential norm and 0.98 cosine gates.
This certificate is confined to the declared stencil.

Per-sign qualification requires both absolute writes to clear the
3% nominal-dose and 0.98 direction gates, 3% pair dose spread, and the
T2-minus-T1 displacement itself to clear 3% relative norm and 0.98
signed cosine against the intended differential. A root qualifies only
with both worlds and both signs. Each world also requires the four
signed writes to fit within the same 3% dose spread. Exact state
aliases remain visible.
Only ordinary floating-point roundoff is covered by the numerical
slack used for the stencil certificate.

The replay sees model states and physical factors, with no answer or
head inputs to selection. It uses no pretrained forwards or formal
tests. The exposed archive can commission an instrument and report
partial coverage. Subsequent causal readouts require native parity,
matched sham and head controls, and an independent confirmation
roster. Preserve the original ARG-03 seal.

The first replay triggered a follow-up analytic coordinate check.
For a unit intervention direction u and required cosine c, the
projection of a feasible unit displacement onto coordinate j lies
between c u_j plus or minus the square root of
(1 − c²)(1 − u_j²), with the corresponding endpoint replaced by −1
or +1 when the cap contains that axis. Vary the allowed radius over
its interval, add the native coordinate and widen for float64
roundoff. An interval containing no BF16 value proves no write can
meet both absolute gates. This necessary coordinate condition is
independent of the search stencil. A nonempty interval provides no
existence guarantee.
