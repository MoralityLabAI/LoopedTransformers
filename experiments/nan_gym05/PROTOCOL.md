# NAN-GYM-05: wider-suite development and observational SAE towers

Freeze and push this source, configuration, protocol, complete source hashes,
public-only fixtures, host-gold hashes and engineering receipt before loading
the pretrained checkpoint. This is a new development campaign, not a change to
NAN-BPD-01 or NAN-DIAG-04. Their failed competence gates remain failed. The
original ARG-03 168-case test stays sealed: never open, regenerate, infer on it,
or use it for exclusions. No BPD transfer, causal model edit or LM training is
enabled by this experiment, even if a new task is easier.

## Sources and native endpoints

The actual wider-suite source modules are used. Do not replace them with easier
single-cell questions or report routing as question-answering accuracy.

* Pointer: 32 fresh CAL and 16 fresh DEV generator roots, three solved examples,
  unassisted queries at one, two, three and four hops over 16 nodes, plus a
  literal-copy control. Hops share one root/RNG/demonstration context and are
  correlated observations. No clean-correct filtering or outcome replacement.
* ARC-1/ARC-2: all three public default grid tasks in each upstream microbench,
  whole query-output grid, exact upstream reference correctness. These are the
  local synthetic grid microbenches, not an ARC-AGI evaluation or AI2 ARC QA.
* Sudoku: all three public default 4x4 puzzles, whole output grid. Preserve the
  upstream exact-reference primary metric and additionally report any legal,
  givens-preserving full solution. Some public puzzles can have multiple valid
  completions; do not conflate reference mismatch with invalid Sudoku.
* Storyworld: 16 fresh source-only viable starts using the exact existing
  environment, modeled rival and horizon six. Give its transition rules and
  target explicitly and request a full action plan, not a single easy action.
  Grade success by executing the plan under the native rival policy; any
  successful plan is correct, not only the source oracle's witness. This is a
  fully specified planning condition, rather than an interactive feedback run.
* Routing: two source-eligible distinct trajectories from each of the existing
  eight categories, including Intellect-3 logic/math. Scan at most 80 rows per
  environment, using source order; reject only repeated trajectories or prompts
  above the token bound, never reward/success/model outcomes. The label is the
  authoritative environment ID. This measures environment routing, not solving
  Intellect, GSM8K, ARC QA, MBPP or web-search tasks.

Each grid, story and route root also has a local explicit copy control. There
are 322 contexts/backend: 160 CAL and 162 DEV. DEV contains 57 root groups:
16 pointer, six ARC, three Sudoku, 16 storyworld and 16 routing. A pointer root
has four main tasks and a copy; each other root has main and copy. Main DEV
answers total 105, controls 57. One checkpoint, two quantization/placement
conditions; these are not two model replications. Public fixture families,
fixed ARC/Sudoku puzzles, shared generator rules and routing source dependence
do not establish independent representative root sampling. Report suites and
controls separately; no inflated all-context headline accuracy.

Public source tasks are exposed development evidence, not new sealed
confirmation. The routing corpus is existing local benchmark data. Its text,
token IDs, trajectories, raw predictions and hidden states stay local. Only
source hashes/row indices, sanitized IDs, counts and aggregate scores may be
published. Public-repo fixtures explicitly exclude the 32 routing contexts.
Do not open unrelated local holdouts. The model worker reads the hash-bound
combined PUBLIC prompt file only; host labels and source oracles are separate
processes. Gold copy answers are intentional, explicit and arm-local.

## Unforced bounded native evaluation

Existing snapshot 451ed48c3273ecef7ea8faaa43c31ce529763bb1, verified by the
ARG-02 lock. HF offline, Transformers 4.46.3, no checkpoint download, update or
training. BF16 staged decoder and NF4 double-quantized decoder are separate
conditions, both with CPU BF16 embeddings/head and 8192-row full-head chunks.
Every token is the full-vocabulary argmax, with no task-label mask, candidate
selection, forced answer, fallback or gold-conditioned ranking. The existing
head's I/N/S/R telemetry does not restrict task outputs.

Use fresh full prefixes, two native loops and no KV cache. Contexts share only
layer loading. Maximum eight B1 contexts/root, maximum 1536 initial tokens; do
not truncate prompts or alter task outputs to fit. Anchored bounded grammars
accept finite labels, complete rectangular JSON grids or one-to-six-action JSON
plans, with balanced bold/inline-code/closed JSON-code-fence wrappers. They do
not scan prose or repair grids/plans. An open fence cannot pass as a complete
answer. Caps are 16 label tokens, 192 grid tokens and 96 plan tokens; character
bounds are in each public schema. Invalid or truncated answers count wrong.
Stop at first complete prefix; complete free-running replies are a distinct
future Prime endpoint and must not be pooled with this native prefix readout.

Physical allocator cap 75%; soft backend limit 7200 seconds between complete
roots. Sequential BF16 then NF4, at most one source/hash-bound resume per
backend for pending roots, preserving initial failures and incomplete artifacts.
No simultaneous large model processes. Append-only telemetry records CUDA
allocated/reserved and sampled Windows dedicated/shared separately. Shared
allocation is not measured paging. Include load time and observation overhead
in elapsed time; no intrinsic throughput/memory comparison across placements.

## Read-only towers and algebraic baselines

Before any pretrained inference, compare hooks versus uninstrumented native
BF16/NF4 random-weight execution at eight B1 contexts over three growing-prefix
steps. Require exact hidden/selected-logit/argmax closure and exact observation
counts. Retain the existing full-dimension full-vocabulary head fixture.

Collect ONLY at the initial complete prompt prefix, before answer generation.
At zero-based layers 5,11,17,21 and native loops one/two, retain last-token
physical vectors at residual_after_layer, down_proj input/output and o_proj
input/output. Hooks return None and never replace states. Record arrays, site,
layer, loop, backend, source IDs and byte SHA; no clean-correct subset selection.
First-prefix single-pass full-head scores are diagnostics, not full single-pass
answer accuracy or equal-compute controls. Such causal comparisons remain a
separate gated protocol.

Extract rank-8/16 nested randomized-SVD weight baselines at those modules using
the backend's own effective operator (dequantize NF4). These are NOT trained
VPD/BPD dictionaries. Keep V[input,rank], U[rank,output], operator/factor hashes,
rank residual errors and float32 factor-plus-residual closure. Require relative
closure error <=1e-5. This algebraic certificate is not bit-exact native BF16 or
bitsandbytes-kernel parity, necessity, a state edit or causal fidelity. Do not
save/distribute the full operator residual. The source recipe is fixed before
data; decompose every registered module, not a selected good one.

For each backend/layer fit ONE small sparse autoencoder shared across loops,
using exactly the 32 CAL pointer roots x four non-copy hops x two loops = 256
vectors. Four layers x two backends = eight dictionaries. Mean/scale are CAL
only, with standard-deviation floor 0.001. Architecture: 3072 ->128 ->3072,
top-four nonnegative codes, stable tie ordering. Train 1000 Adam steps, batch
32, learning rate .001, normalized MSE; project decoder columns to unit norm
after each optimizer step. Fixed seeds and final checkpoint only. No DEV
early stopping, architecture choice, feature selection, LM fitting or reruns
based on outcomes. Preserve the untrained initial dictionary as a control.
Report dead/active features and CAL/DEV reconstruction instead of hiding fit
failures. Eight small dictionaries are a pilot tower, not a mature overcomplete
SAE atlas or semantic validation of individual features.

Cross-reference physical encoder/decoder rows and codes with the matched
down_proj SVD output directions, random output directions and optional *actual*
compatible VPD/BPD artifacts when those exist with verifiable provenance. Width
alone is insufficient: checkpoint/backend/layer/loop/site must match. down_proj
output is added immediately to the layer residual, providing an explicit
identity output-coordinate bridge; o_proj output passes through the subsequent
MLP and needs a Jacobian bridge to the layer-output SAE. Do not use that invalid
identity bridge, an 8192-wide right/input factor, a post-final-norm vector, or a
TRM/Qwen dictionary as a Nanbeige residual direction. Keep missing trained VPD
or BPD stages visibly pending; never rename SVD or reconstruction as trained
VPD/BPD causal evidence.

## Evidence gate, power and reporting

Unit: source root/family, as above. Physical causal dose: zero (observation
only). Causal primary contrast: none; no causal or equivalence claim. Formal
inferential comparisons: zero, no p-values/confirmation intervals. Retain all
suite/copy conditions, four layers, loops, ranks and backends. Main endpoints
and exploratory geometry remain separate. A normalized reconstruction error
improvement of 0.10 is a descriptive instrument-quality screen, not a mechanism
effect floor or a feature causality pass. No semantic matched-sham floor has
been calibrated. Run strict pre-data planning with no invented empirical pilot
and archive its refusal. Eight/16/32 roots or many features cannot certify a
95/95 semantic-sham tail; 59 independent worlds minimum for a maximum-based
bound, plus justified assumptions and fresh confirmation, would be required.

The pointer hop-one/copied-node readout screen (>=12/16 correct main, 16/16
copy) can suggest a task for a FUTURE prospective causal design. It cannot
retroactively pass the original algebraic gate or enable BPD transfer. Actual
donor-specific BPD behavior still requires verified clean competence,
state/operator closure, reached-state routing, rank/random/dose/gauge controls,
matched shams, an independently registered effect floor and an adequate *new*
sealed readout under FSI-01. The old 168-case test is not that readout.

Progress reports may grade completed roots in a separate host process, clearly
mark partial denominators and retain source order/coverage. No protocol,
threshold, hyperparameter or source changes may depend on those rolling
scores. Publish aggregate benchmark and tower reports to LoopedTransformers
with exact source/hash provenance and failures. Prime task adapters must use
the same graders and private evaluator fields, and accurately distinguish
offline SDK validation from actual model/hosted evaluation. Do not launch
billable external services without separately applicable authorization.
