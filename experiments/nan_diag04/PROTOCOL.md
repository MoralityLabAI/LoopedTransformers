# NAN DIAG 04 reciprocal computation and label selection

Freeze and push this protocol, sources, parser, model driver, grader, independent
auditor, tests and source manifest before any new pretrained inference. This
is a fresh development diagnostic informed by NAN-COMP-03's failures, not a
confirmation readout or a redefinition of prior competence gates. The original
168-case ARG-03 test must remain sealed: never read, regenerate, score, or use
it for exclusions. No BPD transfer, fitting, training or hidden-state edits
are enabled, regardless of the result.

## Source population and unit

Two CAL roots and eight DEV roots, one pretrained checkpoint. A source root
contains two fresh degree-four polynomials with constant 1 and a1 in {-1,1}.
Dense sources use nonzero higher coefficients in {-2,-1,1,2}, including at least
one magnitude-two coefficient. Sparse sources have one higher coefficient ±2.
This expanded-coefficient condition differs from the earlier ±1 tasks; do not
interpret cross-study accuracy differences as paired improvements.

Exclude ALL individual input polynomials in the explicitly named exposed
NAN-COMP-02 and NAN-COMP-03 public fixtures, and all previously admitted inputs
in this new cohort. Do not inspect any other roster. Verify EvalMath's unary
oracle against an independent finite-series implementation on every source,
reference table and hidden degree-32 query future. Require I/N/S/R outputs to
be unambiguous and the R versus alternative futures distinct. Retain source
trial/rejection counts. No cases are replaced using model behavior.

Each root has an R world and a counterfactual alternative (N/S/I in the frozen
schedule), sharing inputs/query. World order alternates; DEV has eight R worlds
and eight alternatives. Root/case IDs, split and family never enter prompts.
Reference-code assignments rotate across roots (each code labels R twice on
DEV), while source-only reference order is shared between gold-label and neutral
code matching. Distinct rendered prompts are required across roots for the
scientific arms; literal-copy controls may repeat and are not independent data.
Global source-only rejection induces dependence; iid roots are not asserted.

## Factorized diagnostic arms

There are 24 independently computed paths per root:

* Free computation: ask for b1, b2, b3 and b4 separately for the first input's
  multiplicative inverse 1/f, without I/N/S/R labels or supplied results (4).
* Scaffolded computation: the same four questions with the exact coefficient
  recurrence, but no current intermediate or final answers (4).
* Numeric copy controls: explicitly supply each verified coefficient and ask
  the model to copy that integer; these require no arithmetic (4).
* Raw selection: identify I/N/S/R from two observed examples, for both worlds,
  using operator definitions without a reference output table (2).
* Gold operator-table matching: supply all four exact candidate output pairs
  with I/N/S/R labels and ask which matches the observations (2). Calculation
  is explicitly unnecessary; this is assisted matching, not unassisted solving.
* Neutral-code table matching: identical matching instructions and values, but
  entries use rotated A/B/C/D codes and no operator definitions (2). Apart from
  labels, the table-matching prompt template is identical to the gold-label arm.
* Operator-label copy and neutral-code copy controls: explicitly announce the
  current verified label/code and ask only to copy it (2+2).
* Output-erased raw selection and an independently repeated erased forward:
  same input/definitions, hidden current outputs, identical across worlds (1+1).

All gold information is intentional, explicit and LOCAL to assistance/copy
prompts. It never enters free/scaffold computation or raw selection prompts.
The model worker reads only pre-rendered public prompt strings, not host gold
labels. Every arm has a separate freshly embedded full prefix and no KV cache;
gold-assisted context cannot carry into another arm. No weights update.

The first inverse coefficient is fixed at b0=1. Full reciprocal computation
requires all four independently generated b1..b4 values to be correct AND the
finite product f*b=[1,0,0,0,0]. Report each coefficient separately as a descriptive
breakdown; do not count four coefficients as four independent roots. Selection
R/alternative accuracy and both-world correctness are distinct endpoints. A
source-blind constant R can get 8/16 worlds but never both within one root.

## Unforced bounded decoding

Full-vocabulary greedy argmax at every token, native two loops, no operator or
integer logits mask, candidate search, forced answers, fallback, temperature
selection, embedding reinjection or training. CPU embeddings and full BF16 head;
BF16 decoder streaming versus CUDA NF4 double quantization with BF16 compute.
Twenty-four B1 contexts share only layer loading. Require exact random-weight
grouped-versus-serial hidden/logit/argmax parity over three growing-prefix steps
before pretrained inference. Reuse the validated full-dimension chunked head.

Operator and code grammars accept one uppercase letter from their own alphabet,
bare or in balanced bold/italic/code wrappers, ignoring whitespace, with optional
period. They stop at the first complete prefix, impossible prefix, EOS, eight
tokens or 64 decoded characters. Later free-running prose is not measured.

Integer replies are requested in brackets. Accept canonical signed base-ten
integers in [-999,999], including +n and -0, inside brackets, balanced bold,
italic or code wrappers, bold brackets or code brackets. Spaces around signs
or delimiters are ignored, but digit groups separated by whitespace are rejected.
Unicode minus is normalized. Bare integers require EOS, trailing whitespace or
a terminal period: a bare digit token is NOT a complete answer. Thus 1 can grow
into 17 without a premature stop. A token-limit cut without a delimiter fails.
Reject prose, decimals, fractions, multiple numbers and incomplete wrappers.
Integer limit: 12 new tokens and 64 characters. No current answer is forced.

## Prespecified descriptive interpretation

Controls must be perfect on DEV: full numeric copy 8/8, both operator-label
copies 8/8 and both neutral-code copies 8/8. Otherwise label the overall pattern
readout_control_failure and retain all component results without attributing
failure to arithmetic alone. Lookup competence requires BOTH R and alternative
accuracy ≥6/8 in BOTH gold-label and neutral-code table arms.

Apply this ordered descriptive pattern rule, not a statistical mechanism claim:
1. Failed controls -> readout_control_failure.
2. Free/scaffold full computation ≤2/8, lookup competent, raw R ≤2/8 ->
   arithmetic_bottleneck_compatible.
3. Free ≤2/8, scaffold ≥6/8 and lookup competent -> scaffold_sensitive_computation.
4. Free ≥6/8, raw R ≤2/8 and lookup competent ->
   implicit_selection_gap_despite_explicit_computation.
5. Free ≥6/8, gold-label R ≤2/8 but neutral-code R ≥6/8 ->
   operator_label_interference_compatible.
6. Free ≥6/8 and both matching R arms ≤2/8 ->
   matching_or_instruction_difficulty_compatible.
7. Free ≥6/8 and raw R/alternative each ≥6/8 -> competent_on_separated_subtests.
8. Otherwise -> mixed_or_unresolved.

These patterns are not mutually exclusive causal theories. Prompts differ in
information, length and context, and independent scalar calculations are not
the same demand as implicit computation during label selection. Even a strong
pattern cannot prove a unique internal bottleneck or license BPD transfer.
All thresholds and every arm remain fixed after outcomes are observed.

## Statistics and resolution

Unit: source generator root; DEV n=8, CAL n=2, checkpoint n=1. Primary information
assistance contrast: gold operator table R minus raw R correctness. The physical
change supplies source-computed candidate values; there is no hidden-state
dose. Scientific meaningful floor 0.25 root-solve fraction; resolution 1/8.

Fifteen comparisons are frozen: per backend scaffold-full minus free-full,
gold-table-R minus raw-R, coded-R minus gold-table-R, gold-table-alt minus raw-alt,
raw-R minus raw-alt, and free-full minus raw-R (12); plus NF4 minus BF16 for
free-full, raw-R and gold-table-R (3). Per-coefficient and control tables are
descriptive, without additional inferential tests. Keep zero-effect roots.

Run strict pre-data planning with no invented empirical calibration for this
new endpoint; record its refusal and pilot-only scope. Afterwards archive actual
root assistance effects and same-label original/repeated erasure deltas. Repeated
numerical/format noise does not calibrate semantic assistance variation. Eight
sham roots cannot establish a 95th-percentile bound at 95% confidence: at least
59 independent sham worlds are required even with a sample-maximum method.

FSI-01 flags: preregistered true, sealed confirmation readout false, established
iid root sampling false, paired-label exchangeability false, minimum confirmation
roots 20. Withhold unconditional p-values. Archive exact root-sign-flip and Holm
sensitivities only under an imposed independent-root signed-symmetry law. At
eight roots the minimum two-sided conditional p is 1/128, so the first 15-entry
Holm threshold cannot reach 0.05 (minimum adjusted 15/128). This is deliberately
a power-limited diagnostic, not confirmatory testing. A provisional 60-root
future plan is not a power guarantee; do not unseal the old 168 cases.

## Engineering and publication bounds

Use existing snapshot 451ed48c3273ecef7ea8faaa43c31ce529763bb1, verify all locked
weights/code before loading, keep HF offline. Maximum 768 initial prompt tokens,
8192 vocabulary rows per head chunk, 75% CUDA allocator cap, 2400-second soft
bound between roots per launch, at most one hash-bound resume for pending fixed
roots preserving attempts. Run BF16 then NF4 sequentially. No new case or prompt
search after inference. Keep all raw prefixes, full-vocabulary argmax/normalizer
receipts, labels, grades, rejection counters and failures.

Use append-only telemetry to avoid the prior live-log replacement failure.
Report CUDA allocated/reserved and sampled Windows dedicated/shared counters
separately; shared allocation is not measured paging. Audit sources, prompt
bindings and detokenization, integer delimiters, multiplicative certificates,
labels, control outcomes, root metrics and all 15 statistics. Replay from Git
stored files without weights. Publish the scoped report and evidence in
LoopedTransformers, preserving every prior study unchanged.
