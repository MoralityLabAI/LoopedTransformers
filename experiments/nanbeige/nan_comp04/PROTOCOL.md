# NAN-COMP-04: late-evidence reciprocal computation, source-only gym registration candidate

**Stage:** implementation and source-only preflight. **No pretrained Nanbeige inference, BPD transfer, or held-out readout has been performed.** Freeze this exact protocol, code, CAL/DEV source hashes and parser contract in a separate registration commit *before* authorizing a model run. This is a new exploratory task informed by the already exposed NAN-COMP-02/03 development outcomes; it does not amend their failures.

## Motivation and mathematical identity

NAN-COMP-03 proved bounded generation is feasible on both BF16 and NF4, while reciprocal cases were 0/4 in each. A post-hoc analysis of exposed answer-token class scores found a narrow BF16 R-prior window. All 16 old source worlds were solvable from the first two nonconstant coefficients. Those facts motivate a task in which the decisive reciprocal evidence occurs substantially later and a separate target requires a numerical cross term.

For odd m in {3,5}, integers a,c in {-5,...,-1,1,...,5}, and f=1+a*x^m+c*x^(m+2):

- N(f)=f(-x) and R(f)=1/f agree in coefficients through x^(2m-1).
- At x^(2m), R(f)-N(f)=a^2.
- At x^(2m+2), R(f)-N(f)=2ac. The standalone R-query coefficient is 2ac.
- All claims are exact truncated formal-series arithmetic, verified independently by convolution with f.

The source generator creates two distinct demonstrations per root, a third independent query source, and four operator-world targets I,N,S,R. Four-way classification is measured at D=2m, N/R source-paired classifications at D=2m-1 and D=2m+2, plus N/R mid-degree output erasures. An additional separate numerical question asks for the coefficient of x^(2m+2) of 1/f_query. Each root produces 11 task rows. The root, operator truth, generator control keys and target future are excluded from model-rendered text. Identical N/R early and erased prompts must render to identical tokenizer IDs.

## Fixed source populations and admission

See config.json, committed with the implementation. Four CAL generator roots 184101–184104; sixteen DEV roots 184201–184216; forty reserved future confirmation root IDs 184301–184340 are named only. **Never generate confirmation data for this stage.** Exponents alternate m=3,5; parameters come from the declared ten signed values, with a second demonstration using -a and a separate drawn c. Input sources are drawn deterministically once from each fixed root with no model-outcome access or tuning. Duplicate source identities cause fail-closed error; no outcome-conditioned replacement. This produces pseudorandom root clusters, and independent inferential assumptions require separate justification.

Generate only CAL and DEV with the source-only cohort tool. Keep the public JSONL and host-only labels in distinct files. The model runner accepts the public file plus its hash receipt and must never import host labels or the EvalMath oracle. Validate the actual pinned tokenizer/chat template against prompt identity and paired token counts before any pretrained run. Source-only exact integer tests independently verify the admissible coefficient families.

## Native model protocols

Model: Nanbeige4.2-3B revision 451ed48c3273ecef7ea8faaa43c31ce529763bb1, two native tied passes. Backend sequence BF16 streamed followed by NF4 double-quantized on RTX 3050 4GiB; CPU-resident token embeddings and CPU-resident 166,144 x 3,072 output head, chunked full-vocabulary scan with 8,192 vocabulary rows. Compare both backends on identical source IDs and tokenizer sequence hashes. Keep unforced full-vocabulary greedy argmax on each generated token. Use the original pinned chat template, no logit mask, no beam/candidate search, no forced label and no model-side oracle.

Classification answer grammar: original NAN-COMP-03 anchored uppercase I/N/S/R with optional balanced markdown wrapper. Numerical answer grammar: signed integer enclosed in [square brackets], optional balanced bold wrapper, strict anchored match; stop only on a complete bracketed integer, EOS, impossible grammar prefix, or the frozen token/character cap. Report format and math correctness separately. Record the initial unrestricted top token, the full-vocabulary log-normalizer, operator class scores and every generated prefix for the classification arm. Model-side numerical output is parsed later by the independent CPU grader.

A root-scale dry-run and pinned-weight source/hash check must precede model inference. Run a pilot of exactly two fixed DEV roots before the full cohort, with the pilot marked engineering/development only. No new prompts, cohorts, labels, activation edits, or calibration adjustments may be selected from pilot outcomes under the same source registration.

## Endpoint hierarchy

**Primary DEV competence:** N and R both correct at D=2m within a root and the independently generated query coefficient 2ac exactly correct. Report separate root hit counts for both and the intersection. Gate thresholds from config.json: at least 12/16 N/R-both roots, at least 12/16 numerical query roots, and at least 12/16 R-world classifications. Because the current study is exploratory, meeting these thresholds alone does not license a neural-causal claim.

**Source-effect controls:** At D=2m-1 the N/R prompts are identical; their paired correct count is structurally zero. At D=2m the source difference becomes visible, while both input arrays stay fixed. At D=2m+2 the cross term is available. The output-erased paired prompt is also identical and has a 50%-per-root case-accuracy ceiling. Report N/R response logit margins, source-erased accuracy, the balanced I/S cases, and full greedy outputs. A CAL-only fixed R answer-prior control competes with the free model; a constant R strategy cannot satisfy both N and R within one root. If any candidate calibration is examined, freeze it on CAL before scoring DEV and report all candidate attempts.

**Reasoning/recurrence controls:** Record one-pass versus native two-pass scores on the same fixed prompts in a distinct pre-registered adapter extension. Score standalone 2ac coefficient prediction against fixed value priors and the simple exact formula solver. Source semantics and numerical targets remain distinct. The model can succeed through a simple learned multiplication rule; the experiment would then establish this bounded computation rather than general algebraic reasoning.

## Before any BPD transfer

The code must keep BPD_transfer_enabled=false. A later separately registered BPD gym requires sufficient clean competence, bidirectional N↔R target transfer, own-state online routing, strict present-readout parity, same-dose random-dark and task-direction controls, input-level source edits, and root-grouped future task endpoints. Preserve the BPD CQ-05 gauge and source-visibility controls and the BPD v015 full-operator-plus-residual closure audit.

## Statistical and provenance contract

Treat source generator roots as units. Root-paired tasks, depths, quantization backends, prompt variants and repeated decoder tokens are correlated measurements. Invoke EvalMath FSI-01 and mechanistic-power preflight before any claim. This 16-DEV-root pilot is below FSI's 20-root confirmation policy; the independent 95th-percentile/95%-confidence matched-sham calibration requires at least 59 independent sham worlds. The source family is development-informed from COMP-02/03, so old examples cannot become a fresh confirmation cohort.

Publish source fingerprints, all rejected/invalid run attempts, tokenizer audits, per-case host-only grades, all decisions and failure counts. The original 168-case ARG-03 test remains sealed, ungenerated here and excluded from ALL operations. Do not run --permit-heldout-test, update original frozen protocols or quietly substitute a benchmark that passes.
