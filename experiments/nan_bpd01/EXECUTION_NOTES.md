# NAN BPD 01 execution notes

## Source registration and grading lock

The prospective source registration is commit
`18b102e` on `research/nan-bpd01-phase-causal-fibers`. It precedes the first
pretrained checkpoint run. The registered protocol and configuration remain
unchanged. Source-world admission uses exact oracle certificates and public
token lengths only. Structural construction rejection counters are not model
eligibility exclusions. All six admitted pairs (two CAL, four DEV) are scored;
all eight DEV cases remain in the clean competence denominator.

The separate grading implementation is fixed before DEV correctness grading.
It minimizes mean CAL source-program negative log likelihood over the 27
position-temperature triples, with lexicographic ties. The positional-frequency
diagnostic uses Laplace-one counts from the same four CAL cases. Neither uses
DEV labels for fitting. The current intervention readout is loop one; the
delayed readout is loop two. Donor recovery is graded against the donor world's
own independently verified demonstrations and future, not the recipient label.

The empirical phase preflight uses its actual two-root binary-mean granularity,
1/2, as pilot resolution. The eight-case clean table's 1/8 case fraction is not
substituted for resolution of this two-root causal endpoint. This pilot cannot
resolve the 0.15 scientific minimum; confirmation sample size must be redesigned
with independent calibration and a separate readout.

The original forty-entry exploratory allocation omitted two frequency-versus-
native contrasts although the frequency control itself was specified. Before
DEV correctness grading, the actual family is conservatively expanded to 42
entries, including those contrasts. FSI familywise bounds and Holm use 42;
the original config/protocol retain the planned 40. Unavailable continuous
tests reserve p=1 only for multiplicity accounting, not imputed outcomes.

## Memory and telemetry amendment before NF4

The initial BF16 runner holds all eleven direction tensors at once. Before
NF4 execution, the implementation yields those same tensors one at a time,
in the same order, with the same fixed seeds and arithmetic. A byte-equality
fixture checks the eager and lazy direction plans. This is a storage change;
the model, head, contexts, dose rule, source cohort and acceptance gates do not
change. Separate source manifests retain the exact code used by each backend.
The resulting memory/runtime comparison includes different direction storage
and is not an isolated quantization treatment effect.

The initial sandboxed BF16 CIM calls are denied and their localized stderr
triggers Python decoding errors. They do not stop model inference. An external
permitted observer samples that process's Windows DedicatedUsage/SharedUsage.
The NF4 launch is outside that restriction; subprocess decoding also becomes
explicit and errors are retained. All reported driver peaks are sampled
allocation peaks, not exact residency or paging measurements.

## Readout record references

`ChunkedHead.score` compares each supplied state with the first supplied state.
Native one/two/open records therefore use native one as their reference.
Each ordinary intervention's current record compares with clean loop one;
its delayed record compares with clean loop two. The cached zero-write and
source-input records reuse native score records: their stored diagnostic norms
retain their original native-one reference. Do not interpret those reused
norms as zero-write effects. Zero-write has exactly zero state change and its
behavioral scores equal clean loop two. Full-donor continuation has an explicit
recipient-loop-two reference and a separate state-closure error.

## Instrument limitations retained

Global phase-channel edits can miss the 0.5% dose tolerance because BF16
coordinates round in discrete steps. Such edits remain in all receipts and are
marked ineligible; they cannot establish an equal-dose advantage. Context
suppression and CAL replacement can round to the same physical direction.
They remain reported as registered arms and do not count as independent
evidence. Continuous delayed response and exact current-head parity are
separate from verified task mediation and dictionary recovery.
