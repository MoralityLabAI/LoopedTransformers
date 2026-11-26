# MOLT-08: conditional nonlinear error gates

Development-only continuation of MOLT-07. The measured parent run
[45851078656](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45851078656)
passed at source head 358100c5aee52f358e6d5a898110af3fcf11f726.
At dose 1, stack-rank 1..5 recovery counts were 0,16,112,125,128 out of
128 crossed model/world observations. Terminal SVD counts were
0,1,2,6,10 and random counts 0,0,0,1,8. Rank 5 had near-zero linear
error (mean 6.588e-15) and substantial nonlinear error (mean L2 1.00263,
max L2 7.53104). Correct bit recovery is not exact nonlinear state
reconstruction. The 6,144 raw rank rows have SHA-256
2c94d34bf1659cd394ab1df44f7e840c2b8da35fd5168a42ef41db1d812cdde7.
The parent cache panels at 0/4/16/64 KiB had 0/40/160/256 hits out of
512 queries, 512/472/352/256 map constructions and wall seconds
0.31498/0.31336/0.27642/0.24673, preserving every replay output. These
are subsecond descriptive timings under fixed exact-repeat locality.
Lifetime peak RSS was 428,630,016 bytes for the whole run, not each arm
or an enforced process-tree cap.

## Uniform curvature and a sharper segment bound

Fix a reached state z and its full stacked future observer F. Let d
be the original restoration displacement, p its rank-k orthogonal
projection, e=d-p, and a=max(||d||,||p||). If ||D²F|| <= M throughout
the closed ball around z of radius a, then integrating the derivative
on the line from z+p to z+d gives

    ||F(z+d)-F(z+p)|| <= (sigma_(k+1)(O)+M*a)*||e||.

Indeed O=DF(z), ||Oe|| <= sigma_(k+1)||e||, and
||DF(z+p+t*e)-O|| <= M*a on that segment. This also gives zero exact
projection error at full rank, unlike the sum-of-two-Taylor-remainders
bound. The original displacement must be present at ingestion to
measure ||e||; a forecast-only allocator can use the more conservative
radius upper bound instead.

In the strict predecessor-chain fixture each reachable next state is
tanh(W h+b_j), with fixed b_j. Its Euclidean Jacobian norm is bounded
by ||W||_2 and its bilinear Hessian norm by c||W||_2² where
c=4/(3 sqrt(3)) bounds |tanh''| globally. Starting L=1, M=0, each hop
updates M'=||W||M+c||W||²L² and L'=||W||L. Stacking linear readouts
q_j of reachable positions gives a bound sqrt(sum_j (||q_j||M_j)²).
The computed spectral norms and floating-point comparisons are
numerical; this is a real-arithmetic inequality, not an interval-rounded
machine certificate.

## Directional interval enclosure

A second, less conservative gate bounds the difference along the
specific segment. At ingestion, parametrize h(t)=z+p+t(d-p), t in [0,1].
For each of 1/4/16 equal subintervals, take coordinatewise endpoint
min/max for h and the exact constant initial tangent e. Propagate a
box for both h and its directional derivative along the fixed causal
chain: linear maps split positive/negative coefficients, tanh is
monotone, tanh' extrema follow distance to zero, and interval
multiplication encloses the chain-rule derivative. Linear readout
derivative boxes integrate to an enclosure [l,u] of the actual
F(z+d)-F(z+p), by summing each derivative interval times its segment
length. The exact cone has five output rows; everything outside it
is independent of the intervention. This procedure is specific to
the declared strict chain, not a general attention or native BPD bound.

An approximate nonzero readout q is sign-stable if its whole interval
[q+l,q+u] remains strictly on the same side of zero. One conservative
scalar alternative uses max(abs(l),abs(u)) across all five outputs.
The implementation uses ordinary float64 and a disclosed 1e-9 margin;
there is no outward-rounded interval arithmetic. Every enclosure and
accepted sign decision is independently checked against the actual
full restoration on the development panel. No formal or hard LDT
certificate is authorized, even when all checks pass.

## Fixed panel

Four fresh model seeds 137,139,149,151, 512 updates; 8 length-8 and
8 length-12 shared fresh development inputs. Sweep rank 1..5 and
doses 0.25,0.5,1, retaining all 960 candidate rows, and choose the
smallest eligible rank per gate. Compare linear-only diagnostic
(which may false-accept), global curvature, and symmetric/signed
intervals at three subdivision counts. Full-displacement fallback
happens before the donor is discarded. The decision uses no target
label or actual full-readout result; actual full outputs and labels
are evaluation-only. An invalid real-arithmetic enclosure or admitted
sign change stops the run. Signed intervals carry 2*5 float64 error
endpoints; scalar gates carry one float64 allowance. Report coefficient
and bound payload bytes with full fallback 128 B, excluding the
812 B rank-5 map, metadata and allocator costs. With a rank-5 scalar
bound the payload is 48 B, so sharing the map across at least 11 such
records is necessary to amortize it (excluding metadata). Signed
rank-5 records occupy 120 B each, requiring at least 102 records.
Per-record bounds require binding to the specific displacement,
observer, epoch and map. The study stores diagnostics; it does not
implement a persisted trusted certificate envelope or remove the
original donor from live memory.

Candidate timings include projections, approximate suffix calls and
interval construction. Summed candidate work is a development estimate
for an early-stopping rank search, not a measured standalone runtime;
it excludes training, map construction, full fallback and receipt IO.
Do not infer a hardware speedup. This is a new fixed development panel
with crossed model/world dependencies, no protected evaluation or
population claim. No new native MoLT or BPD mechanisms are established.
Run strict EvalMath preflight and fresh registered independent units
before any confirmation claim. Use the operator's resource-cap wrapper.

    python -m pytest -q tests/test_molt_observer_error.py
    python -m experiments.molt_goblin_v1.mms_error_study --out /tmp/molt08.json

A useful next MMS contract is a per-record provenance-bound error
receipt, an explicit fallback cost in DataGoblin's allocation forecast,
and an independently verified outward-rounded or exact arithmetic
gate for any hard claim.
