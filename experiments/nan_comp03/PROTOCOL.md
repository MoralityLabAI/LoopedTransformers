# NAN COMP 03 bounded answers and reciprocal examples

This prospectively frozen development pilot follows NAN-COMP-02's formatting
and reciprocal failures. It uses new source worlds and does not rewrite either
prior pilot. The original ARG-03 168-case test must remain sealed: do not read,
regenerate, sample, or use it for exclusions. No BPD transfer is enabled,
regardless of this pilot's result. Keep all admitted cases and all arms.

## Sources and fixed prompts

Identify one unary operator I(f)=f, N(f)=f(-x), S(f)=f(x²), R(f)=1/f from two
degree-four polynomial demonstrations. Verify predictions by executing that
operator on both demonstrations and a hidden degree-32 query. This is operator
identification with exact executable verification, not autonomous coefficient
generation. With four candidates, search@12 is a trivial structural ceiling.

Use two CAL roots and eight DEV roots, two counterfactual worlds per root.
DEV contains four worlds of each operator, with the same preassigned operator
pairs and dense/sparse source families as config.json. Input/query/family are
shared within a pair; independently verified target operators/futures differ.
No current target labels, queries, IDs, family or split appear in prompts.
The EvalMath unary oracle and independent finite geometric/convolution checker
must agree before admitting every source. Reject ambiguous examples, equal
futures, duplicate demonstrations, token-length mismatch, oversized prompts
and out-of-bound public coefficients, retaining every rejection count.

Exclude the ten input pairs in NAN-COMP-02's already exposed public fixtures,
using only that specifically named 20-world public file. Do not open any other
roster. Exclude current tutorial inputs. Deduplicate unordered input pairs and
ACTUAL rendered token prompts across the new CAL/DEV roots; query coefficients
are not a proxy for rendered uniqueness. Erased token IDs must be identical
within a root and distinct across roots; clean token lengths match within
pairs. Admit sources solely from these fixed source/tokenizer constraints,
not model outcomes. Source-only global rejection induces dependence: do not
assert iid roots or inferential pair-label exchangeability from unique IDs.

Balanced control: definitions plus one simple example for each operator.
Curriculum: the identical balanced block plus an explicit recursive coefficient
rule for the SERIES reciprocal, three verified reciprocal examples and the
worked coefficient calculation [1,1,-1,0,0] -> [1,-1,2,-3,5]. Extra R examples
can shift the prior; source-erased curriculum is the competing control.
Erasure redacts ONLY the current task outputs, retaining tutorial/input data.
Both formats, all examples and wording are fixed in prompt.py. No CAL fitting,
prompt selection, temperature fitting or training occurs.

## Bounded greedy decoding and parser

Use the pinned native chat template with thinking disabled and assistant
prefix `Operator:`. Unrestricted full-vocabulary greedy argmax at EVERY token;
no logits mask, forced operator, beam/candidate search, temperature or fallback.
Recompute the entire growing prefix through the two native loops at each step,
without KV caching or embedding reinjection. Stop at the first complete parser
match, first impossible grammar prefix, EOS, eight generated tokens or 64
decoded characters. All tokens and full-vocabulary score receipts are retained.

The anchored grammar accepts one uppercase I/N/S/R, bare or surrounded by
balanced **bold**, *italic* or `inline-code` markers, with optional final period.
Whitespace is ignored anywhere within those forms. All other text is rejected;
never scan explanation/prose for a letter. Unclosed wrappers at a bound fail.
Stop at the first complete valid PREFIX: later free-running prose is not
generated or measured. A bare letter may therefore terminate before a later
period. This is a frozen stopping policy, not an EOS-complete answer claim.

Keep four independent generation paths per world: curriculum, balanced,
erased curriculum, and an independently repeated erased generation. The two
erased paths are freshly computed per root and each reused within its identical
pair. Six independent B1 contexts are processed per root, sharing only decoder
layer loading. A GPU fixture must show exact hidden/argmax parity with serial
B1 prefixes for BF16 and actual NF4 over three growing-prefix steps before
pretrained inference. CPU tests cover parser rejection/termination and gates.

Report the old strict first-token alias rule on the curriculum's initial score
as a derived control, plus the frozen constant-I prior. Also report first-step
four-choice alias-marginal ranking, separately from bounded greedy accuracy.
Restricted ranking and parser success alone cannot establish competence.

## Operational competence gate and comparison family

In EACH backend require curriculum bounded-greedy verified future correctness
at least 12/16, BOTH targets correct in at least 6/8 roots, reciprocal correctness
at least 3/4, and mean parsed-accuracy advantage over erased curriculum at least
0.25. Identical erased inputs have at most one of two distinct targets correct;
their both-correct ceiling is exactly zero and case ceiling is 50%.
A joint gate requires BOTH BF16 and NF4. Do not revise any gate after inference.
A pass is a scoped development prerequisite on this unary condition, not a
passing original compound gate, BPD transfer, recurrence geometry, confirmation
or quantization equivalence. A failure is retained without another prompt run.

Nine comparisons are fixed: per backend curriculum parsed correctness minus
erased, balanced, curriculum strict first-token and constant-I correctness;
plus NF4 curriculum minus BF16 curriculum. Scientific minimum is 0.25 solve
fraction, aggregate resolution 1/16. The primary contrast is curriculum minus
erased; the intervention removes public source outputs, with no hidden-state
physical dose. Source root is the sampling unit; the two worlds, backends,
prompt arms and generated tokens are correlated observations on that unit.
One checkpoint, eight DEV roots, 16 worlds; two CAL roots/four worlds separate.

Freeze registration/config/parser/generator/cohort/decoder/grader hashes and
push before any new pretrained inference. Pre-data strict power planning has
no empirical pilot for this new endpoint: record its honest refusal, not old
task shams. Afterwards use the actual eight root-level effects and same-label
erasure-repeat deltas. Repeat noise is only numerical/format noise, not a
semantic-intervention tail. Eight roots cannot calibrate a 95th-percentile
sham bound at 95% confidence (at least 59 independent sham worlds required).
A provisional future 60-root plan is not a power guarantee.

Apply FSI-01 with no sealed confirmation readout, no established iid roots and
no established paired-label exchangeability. Withhold unconditional p-values.
Archive root-macro differences and all nine exact root-sign-flip sensitivities
under an imposed independent-root signed law; Holm includes all nine. Those
conditional sensitivities do not establish the law's assumptions. Bootstrap
intervals are descriptive. This is power-limited development evidence, even
if the operational gate passes; there are fewer than FSI's 20 confirmation
roots and no independently sealed new confirmation readout.

## Engineering bounds and publication

Reuse the existing snapshot 451ed48c3273ecef7ea8faaa43c31ce529763bb1 with
all locked checkpoint hashes verified before loading. CPU embeddings and CPU
BF16 full-vocabulary head in both backends; BF16 decoder streaming versus
NF4 decoder CUDA with double quantization and BF16 compute. Preserve native
two-loop recurrence. Head chunks 8192 rows; candidate batch one. Maximum
768 prompt tokens, eight answer tokens, 75% CUDA allocator cap, 2400-second
soft limit between roots per launch, at most one hash-bound resume for pending
fixed pairs. Preserve attempts, failure and allocation receipts.

Run BF16 then NF4 sequentially. No sources or prompts change between backends.
Report CUDA allocated/reserved, sampled Windows dedicated/shared allocations
and timing separately; shared allocation is not measured paging. Independently
audit sources, transcripts/parser stops, score marginals, all scalar grades,
root contrasts and nine-entry Holm accounting, and replay from stored Git
files without weights. Validate detokenization/prompt binding separately with
the pinned tokenizer. Push a scoped report with all evidence to LoopedTransformers.
