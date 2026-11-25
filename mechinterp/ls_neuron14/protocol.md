# LS-NEURON-14: certificate-aware discrete-search pilot

## Frozen question

Does replacing point-loss selection with a numerical interval-enclosure score
increase the number of exact invariant certificates obtained under a matched
discrete-search budget? The primary comparison is `bound_single` against
`point_single`. The secondary comparison is `bound_paired` against
`point_paired`. Both comparisons belong to one family of size two.

The primary development gate requires at least 12 newly certified models among
24 assigned seeds and at least six more new certificates than the matched point
arm. Apply the same count gate separately to the secondary comparison. Report a
secondary pass without promoting it to a primary pass. This is a prospective
development pilot; operational count gates do not establish a population effect
above the scientific floor of 25 percentage points.

For this gate, a new repair requires an initial POINT_VIOLATION witness and a
final CERTIFIED verdict. An initial UNRESOLVED case remains in the denominator
and contributes zero to primary repair. Report certificate acquisition from
any non-CERTIFIED initial model as a separate descriptive count.

## Models and independent units

Use 24 fresh seeds, every integer from 445 through 468, at 15 percent tensor-RMS
Gaussian perturbation. The initializer, Torch generator seed `12010000 + seed`,
FP32 noise arithmetic, RMS floor 1/8, and initial quantization come unchanged
from LS12/LS13. Generate each initial model once and share it across all four
arms. The perturbation seed is the independent unit. Every transition case and
trajectory generated for that model belongs to the same unit.

Parent: LoopedTransformers commit
`88aef2c14c6b9d75775c2379c2ee5cc6ce5484a9`, draft PR 385. Supplied LS13 archive:
SHA-256 `02b988671c37e16391c25c44ac560f814e2709b6d1146ec2963d817961b0f316`.
The eight-GELU scalar mod4 cell has 54 recurrence parameter slots, including 52
integer codes in [-7,7], four positive scales, and two FP32 scalar offsets.
Codes alone are four-bit; scales and offsets remain FP32 at decode.

For p in {0,1,2,3}, let c_p = p - 2 and R_p = [c_p-1/8,c_p+1/8]. Inputs
(a,b,n) range over {0,1}^3, with target state (p+a+b) mod4. The certificate
requires transition error at most 3/32 across all 32 state/input cases, leaving
room for an arbitrary additive state disturbance of magnitude at most 1/32.
The final verifier uses exact GELU and exact beta=1/5 with stored FP32 weights
interpreted as dyadic rational numbers. The initial state is -2 and readout is
the nearest of the four centers.

## Fixed search and controls

Each arm uses one common initial fit, then exactly 16 rounds. A round uses two
nonlinear-scale anchors, current and original initialization. Both anchors
profile the same fixed proposal-slot family: 105 slots for single-coordinate
search, or 177 slots for paired search. The first slot is no-op. Illegal moves
become no-op and duplicate codebooks remain in the profiling budget. Shortlists
contain the no-op plus the four distinct non-no-op codebooks with the smallest
assigned score, with stable slot order breaking ties. Each shortlisted model
receives one continuous fit. The incumbent remains eligible at no fitting cost.

There are exactly 161 fitting calls per arm/seed, each capped at 40 function
evaluations, and exactly 3,360 single-family or 5,664 paired-family profiled
slots. The 6,440 function-evaluation cap is the same within each comparison.
Actual function evaluations and elapsed time are reported. These are equal
search-opportunity caps; the interval score adds arithmetic, and fitted models
can converge after different numbers of evaluations.

Single moves change any one of the 52 codes by +/-1. Paired search adds the
LS13 architectural proposals: copy one paired up-row to its mate, move both
paired up entries or biases together, move paired down entries oppositely, set
paired bias gap to two codes, or enforce opposite paired down entries. Search
does not consult the unperturbed template codebook. Paired moves use the known
ramp-pair architecture.

All continuous fits and coefficient profiles are the unchanged LS13 routines.
The fitter minimizes mean squared error plus ten times mean squared excess
above 3/32 at nine equally spaced points in each region. It fits four log
scales and the combined offset with an analytic Jacobian. Scales are bounded
between 1e-6 and 128, combined offset between -8 and 8, and down bias stays
fixed. SciPy least_squares uses max_nfev=40 and ftol=xtol=gtol=1e-9. Profiling
fits positive skip/down scales and an offset at fixed nonlinear scales; its
clipped least-squares proposal is not claimed to globally optimize the hinge
objective.

The common initial fit is accepted identically in all arms. After it, a new
state is accepted only when its assigned selection score improves by more
than 1e-12. All candidates are decoded through the actual FP32 products before
selection scoring. Bound-arm fits still optimize the point objective; this
experiment changes ranking and acceptance, keeping the fitter fixed.

## Numerical selection scores

Both selection rules evaluate the same 17 equally spaced points in each
radius-1/8 region, for 544 endpoint evaluations per model. The point rule uses
mean(e^2 + 10 max(|e|-3/32,0)^2) over those endpoints.

For the interval rule, split each region into 16 segments of width Delta=1/64.
Write g(z)=z Phi(z). Then g''(z)=(2-z^2)phi(z), and
g'''(z)=z(z^2-4)phi(z). The maximum of |g''| on an interval occurs among its
endpoints and any contained members of {-2,0,2}. For each neuron, compute this
local curvature maximum over its preactivation interval. On a state segment
[l,r], score

    max(|F(l,x)-c_target|, |F(r,x)-c_target|)
      + (r-l)^2/40 * sum_i |down_i| up_i,h^2 max_Zi |g''|.

The interval score is the maximum over every segment and every case. Linear
interpolation with a second-derivative remainder proves the corresponding
real-arithmetic enclosure. Float64 evaluation here has no directed rounding;
the selection score is a numerical proxy, and it cannot issue a certificate.
The curvature sum discards cancellation across neurons. Report disagreement
between proxy threshold and exact verification in both directions.

This comparison changes average-loss aggregation into worst-interval
aggregation while adding a curvature allowance. It tests that combined change.
There is no point-maximum-only control that isolates the curvature allowance.

## Data lock, final scoring, and audit

Hash this protocol, all inherited dependencies, experiment/audit code, evidence
registrations, and statistical scripts before generating seeds 445..468.
Commit the locked sources remotely before starting the cohort. This provides
source chronology and does not constitute external scientific preregistration.
Use only LS13 seeds 433..444 for implementation checks. Preserve the source
hashes and development-check receipt.

Finish all 96 searches before exact certification or trajectory scoring.
Apply the inherited verifier unchanged to all 24 initial models, all 96 final
models, and the constructive positive control. Preserve CERTIFIED,
POINT_VIOLATION, and UNRESOLVED separately. The primary new-repair count requires
a final CERTIFIED endpoint whose initial verdict was POINT_VIOLATION. Also list
any certificate acquired from an UNRESOLVED initial model. Initial failures and every
unsuccessful arm remain in the denominator. A violation witnesses failure of
the specified region invariant; it does not by itself prove reachability.

Replay initial and final numerical T8/T128 rollouts using the inherited
256-sequence recipe and its seed `12030000 + seed`; keep raw final and
all-prefix correctness vectors. These sampled FP32 diagnostics do not replace
exact certification. Do not use them to select a model or alter this protocol.

Audit all initial decodes, fixed proposal slots, profile score arrays,
shortlists, recorded fitted candidate states, accepted transitions, final
weights, budgets, and source links. Recompute every score and all final exact
verdicts. Record the scope of optimization replay; checking fitted states and
selection does not independently retrain SciPy's trajectories. Check score
mathematics through a separately implemented scalar calculation and dense
final-region sampling, while retaining the rational proof as the decisive gate.

No adaptive extension, weaker disturbance target, paid compute, native
transformer evaluation, or protected ARG readout belongs to this run.

## Evidence policy

Apply the frozen FSI-01 registrations for the two comparisons. Report paired
new-certificate discordances, exact two-sided McNemar fractions, Bonferroni
adjustment across both comparisons, marginal intervals, paired uncertainty,
and the 25-point effect-floor gate. Fresh seeds come from one synthetic
perturbation distribution around a hand-built model.

The official EvalMath preflight uses historical LS13 outcomes as context.
There is no direct LS14 effect pilot or empirical matched-sham calibration.
Its conservative support-bound input returns REDESIGN_OR_PILOT_ONLY. Retain
PILOT_ONLY for the empirical algorithm comparison. Treat rational invariant
certificates as model-specific mathematical evidence. Lean/kernel proof,
hardware rounding guarantees, and generalization to native transformers
remain separate work.
