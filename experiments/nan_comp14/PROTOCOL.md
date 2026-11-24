# NAN-COMP-14: fresh NF4 competence with matched BF16 controls

Prospective source-only development pilot. Freeze, commit and push this entire
source/parser/criteria/tokenizer lock before the first pretrained COMP14 call.
The prior FIT07 NF4 DEV5/8 failure and all old criteria remain unchanged.
This pilot does not end the user's continuous Nanbeige gym goal.

## Question and source construction

Can native NF4 answer a fresh source-conditioned single-hop query in both
counterfactual worlds, and does a fixed one-hop prompt recipe change the
observed first-token versus bounded-answer competence? BF16 is the matched
control, not a source-selection oracle. There is one pinned checkpoint:
451ed48c3273ecef7ea8faaa43c31ce529763bb1.

Eight CAL and sixteen DEV procedural roots, two worlds each. Each graph is a
permutation of A–P; swapping exactly two outgoing destinations changes the
queried answer while preserving row order, query and destination multiset.
The two targets differ. Targets balance all 16 labels within each split.
Queries, permutations, row order and demonstrations use the fixed hash-derived
root RNG. No draw is filtered by model performance. Only the named exposed
GYM05/FIT07/CTX08 public fixtures are inspected for source overlap. Neither
protected test data nor any original test roster is read, regenerated or used
as an exclusion list.

Each world has five arms, hence 240 contexts/backend:

- legacy: inherited instruction and three 1/2/3-hop demonstrations;
- focused: explicit one-edge instruction and three one-hop demonstrations;
- copy: explicit verified answer, intentionally supplied only in this arm;
- legacy_erased and focused_erased: remove query edges but preserve the
  corresponding examples/query; world-A/world-B prompts are exactly identical.

The two prompt recipes jointly change instruction and demonstration content;
their contrast does not isolate either change. They are both reported. No
post-readout prompt, parser, case, seed, threshold or token-budget revision.
Repeated copy and erased prompts are intentional controls, not fresh roots.

## Frozen readouts and new-condition screening criteria

Independent B1 full-prefix inference, two native loops, no KV cache. Each new
token is the unrestricted full-vocabulary loop2 greedy argmax. No answer-token
mask, constrained label selection, sampling, teacher forcing, repair or fallback.
Thinking disabled with the pinned native chat template and existing Answer
suffix. At most 6 generated tokens and 64 reply characters; stop at the first
complete anchored prefix, EOS or bound. The anchored A–P grammar permits bare
letters, a terminal period, balanced square brackets, emphasis or one inline
backtick pair. It refuses prose, code fences, multiple labels and incomplete
wrappers. Later prose after an already accepted prefix is not this endpoint.

Retain first-token outcomes from loops 1/2 on the exact initial prefix and
bounded loop2 outcomes. Invalid outputs count wrong; no root/outcome is dropped.
For EACH fixed template and EACH loop2 endpoint, the new screen requires:

1. Both worlds correct in at least 6/8 CAL and 12/16 DEV roots.
2. Every copy pair correct (8/8 CAL,16/16 DEV).
3. Every erased pair has byte-identical generated-ID sequences and zero
   both-world successes; any instability blocks source-specific interpretation.
4. Full registered backend coverage and the independent terminal audit pass.

These are development instrument screens, not statistical confirmation or
repairs to FIT07. Bounded success cannot qualify a first-token fitting task.
A future NF4 fit requires a separate prospective registration and a matching
readout/competence gate. No automatic fitting, donor transfer or original
algebraic gate reopening. All templates/endpoints remain visible, including
failed, invalid and incomplete arms.

## Observations, audits and comparisons

Archive the initial prefix's 40 physical vectors: loops 1/2, layers 5/11/17/21,
residual_after_layer, down_proj input/output and o_proj input/output. Local
arrays, initial/generated token IDs, score tapes and replies stay under runs/.
Full-vocabulary CPU/resident-head parity is checked on both first-prefix loops.
Original snapshot locks are checked before model construction; hash target
parameters before/after inference. Host gold never enters the model worker.

After actual worker termination, a separate host audit loads only the pinned
tokenizer, independently parses the bounded grammar, verifies textual graph
answers, every generated-prefix decode and score-ID binding, all frozen input
hashes, and every state archive hash/dimension/finite/coverage contract. It
does not replay a second full model forward. No partial outcome is promoted to
an audited result. Missing contexts remain missing; no neural retries.

After both complete audits, pair every identical INITIAL prefix in physical
residual coordinates. Report loop1/loop2 state cosines/relative L2 and loop2−1
displacement cosines/relative L2, with BF16 norms as denominators. Average two
worlds within root then roots equally, for all 40 split/arm/layer strata.
Undefined zero-norm quantities remain missing and counted. Ranges are
descriptive, not confidence intervals. Later generated histories may diverge
and are not paired. No SAE-index subtraction or hidden-state intervention.

## Resources and execution

NF4 first, then BF16, one large model at a time. Existing offline snapshot and
environment only: Torch 2.6.0+cu124, Transformers 4.46.3, bitsandbytes 0.50.2.
NF4's 22 decoder layers and unchanged BF16 head are resident on GPU. BF16 keeps
four decoder layers plus reusable staging and the same unchanged head there.
CPU embeddings retain context capacity. Chunk rows 8192, allocator fraction .85,
prompt limit 1536 (tokenization receipt currently max 389), backend between-prefix
budget 7200 s including load. OOM/time/error is retained; no automatic restart,
prompt truncation, lowered precision, reduced vocabulary or criterion change.

Set OMP/MKL/OPENBLAS/NUMEXPR thread counts 1 before numerical imports, use
Below Normal priority, serialize host audits, never touch the separate math
process. Driver dedicated/shared samples after load and at termination remain
distinct from CUDA allocated/reserved peaks; shared-machine timings are not
isolated throughput. No packages/weights are changed or paid APIs used.

Every model case/state archive and terminal receipt is written once. Only
append-only events are observed live; never ordinary-open atomically replaced
worker progress on Windows. Reports consume closed events/terminal audits.
The finite queue preserves every error, attempts both fixed backend conditions,
streams aggregate milestones to LoopedTransformers, and cannot retry a model.

## Evidence and resolution

Unit: fresh source root; two worlds, five arms, loops and tokens are correlated
measures. CAL n=8 roots/80 contexts, DEV n=16 roots/160 contexts per backend;
one checkpoint, two previously informed prompt candidates. Primary contrast:
source-conditioned both-world competence by fixed template/readout, with
matched BF16/NF4 description. Physical intervention dose: none. Scientific
causal-effect floor and matched-sham floor: uncalibrated. New screen 75% paired
root solve plus complete copy controls is an engineering eligibility rule.
Planned/actual formal comparisons: 0. No population accuracy interval,
significance, equivalence or causal-power claim. Prior studies were exposed;
this fresh readout becomes exposed development evidence after the pilot.

EvalMath's mechanistic-power/FSI gate therefore assigns pilot-only evidence:
this cohort cannot establish a 95/95 causal sham tail, donor mediation or
mechanism equivalence. Future causal claims need a meaningful effect floor,
matched physical sham controls, independent source roots and a new sealed
readout. The original 168-case ARG-03 test is not that readout and stays sealed.
