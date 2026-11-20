# NAN-KV-12: explicit BF16 effective-weight NF4 execution

Freeze and push all sources before pretrained execution. The new numerical
policy explicitly dequantizes each unchanged double-quantized NF4 decoder
matrix to BF16 before F.linear, for both full-prefix and cached calls. The
output head remains the unchanged full BF16 vocabulary head. Embeddings stay
on CPU; all 22 NF4 decoder layers/head stay on CUDA. No model/package downloads,
requantization, parameter updates or changes to any previous condition.

This is motivated by exposed KV11 incremental errors and a prospectively
registered random-model fixture. On that fixture BF16 and explicit-effective
NF4 were internally byte exact; default NF4 had nonzero differences. Explicit
and default NF4 full-prefix outputs also differed. These are engineering
observations, not evidence of real-model equivalence, restored competence or
causal fibers. The new real-checkpoint condition may fail and is kept if so.

## Same public scope, independent arithmetic condition

Use every public GYM05 DEV context: 130 contexts, 41 heterogeneous source
families, including pointer/ARC1/ARC2/Sudoku/storyworld and all 41 public copy
controls. No outcome-based inclusion or replacement. All GYM05 and later
results, plus partial KV11, were exposed when this protocol was designed.
Private routing and the original 168-case test are never read, regenerated,
graded or used as exclusions. Do not import task gold into the model process.

For each context, both paths start at the identical full prefix. The explicit
policy full-prefix path supplies up to eight unconstrained full-vocabulary
greedy token readouts, stopping at its EOS. The cache receives exactly that
reference history, even after an argmax disagreement. This is not an
independent cached free-running reply after divergence, nor a complete-task
accuracy evaluation. Reset independent per-pass DynamicCaches at every source;
use two native full-stack passes, correct positions/masks/physical layer indices
and existing native normalization. No hidden-state or cache interventions.

## Checks and retained outcomes

Before mounting the new modules, hash all target parameter bytes/names. Reuse
the same parameter objects, mount before observers, then assert the digest is
unchanged. Record changed decoder leaf count and policy. Before DEV, the fixed
first public CAL source checks CPU/resident-head equality at both loops and
warms up capacity; its answer never selects the cohort.

Every initial cached prefill must be byte exact to the new full-prefix path at
both normalized boundaries, all 166144 vocabulary logits and all 40 physical
sites (layers 5/11/17/21, both passes, residual/down/o inputs/outputs). Stop on
prefill failure and retain it. Archive every later numerical error and argmax
disagreement without tuning thresholds or asserting an equivalence margin.
All raw paired logits/boundaries/physical vectors and token histories remain
local. The independent CPU audit recomputes every archived difference, shapes,
finite values, parameter/source binding, trajectory, expected cache lengths
and 180224-byte-per-token K/V storage. Final target bytes must be unchanged.

After the original KV11 queue actually exits, its existing BF16 and default
NF4 initial-prefix readouts are the matched historical references. Compare
all available common initial histories, not later policy-dependent tails.
Require exact source IDs/prompt IDs and archived SHA bindings. Retain all
suite/copy/loop groups of free-argmax agreement, full-vocabulary logit error,
normalized boundary cosine/relative L2 and normalized loop-delta relative L2;
also four physical pre-normalization residual site cosines/relative L2. All
groups/zero differences retained. Missing historical coverage remains missing.
BF16 need not be rerun because this policy changes only NF4 arithmetic.

## Resources, supervision and interpretation

Wait for actual KV11 supervisor/model termination, then one NF4 model process,
one numerical CPU thread, Below Normal and CUDA allocator fraction .85.
Temporary effective matrices are released per call, never persistently stored.
Record allocated/reserved peaks plus separate instantaneous WDDM dedicated/
shared samples. No physical-residency claim follows from tensor.device.

Budget 7200 seconds including load/warmup, checked between measured steps.
Preserve in-flight work and every failure; no automatic neural retries or
outcome-filtered restarts. Append-only events only, closed artifacts for audit;
never read a live atomically replaced progress JSON. Publish only named
aggregate report files at 32-context milestones and terminal outcome.

Timing includes both-loop probes and full vocabulary readout, excluding NPZ
compression; fixed reference-then-cache order/CUDA sync. This is a shared
machine, with the separate CPU math workload unchanged; no isolated throughput
claim. CPU audits are serialized and one-threaded at Below Normal priority.

Unit=41 source families, n_cases=130, checkpoint=1; token positions, copy arms,
loops and execution conditions are repeated measures. Primary contrast is
same-history cached versus explicit-policy full-prefix numerical execution;
historical initial-prefix geometry is secondary descriptive evidence. Physical
dose=0; scientific causal floor, matched-sham floor and causal power are
uncalibrated. Planned/actual formal inferential comparisons=0. Verdict is
development engineering only. No causal/competence/equivalence claim or donor
transfer, and no alteration of original failed algebraic gates.
