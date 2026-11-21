# Early storyworld audit without another model load

The full RAM28 NF4 gym is generating storyworld trajectories first. A separate
model-free audit is now frozen to grade **all 72 native story trajectories**
once that block closes. This supplies earlier behavior/state evidence while
the remaining full gym continues; it does not reduce the 1,549-baseline /
216 real A/B-pair scope or replace its canonical post-GPU SAE/BPD analysis.

Freeze commit: `fe58a7accb4998c6bce87b2441cf00bb13169a66`.
Audit seal: `8671e772f7166facca90ecd5fa74683a0908270fe7051e06b3c6fc45da482143`.
Eleven critical files are bound, with inherited RAM28/method sources unchanged.
The audit was frozen after initial RAM28 execution began but before inspecting
or grading generated text/states. The original story sources and inference
condition were already frozen before their neural execution. No new inference.

The fixed block contains 12 roots (3 CAL, 9 DEV), two counterfactual worlds,
main/copy/erased arms, and all registered query/event-count strata. The audit
retains the original bounded grammar and format/final/checkpoint/path/joint
endpoints, exact pinned prompt IDs, decoding/stopping order, prefix histories,
closed-record/archive hashes, all 40 BF16 vectors per prefix, native local-write
identity, and actual batch/padding checks. Correct final answers do not rescue
incorrect or malformed traces. Missing outcomes retain world/root bounds.

**Validation:** eight local tests passed with none skipped: source order and
controls, original grading AST, missing-outcome bounds, duplicate/collision
rejection, real Windows creation-bound handles, resource/exit boundaries,
paired-padding/no-serial-substitute checks, and synthetic uint16 archives with
corrupt-history rejection. The unchanged source oracle also passed its 144
original registrations and 120 textual replays. These are engineering/source
checks, not real model grades. The new audit has not run yet.

**Observed 2026-11-21 15:54 UTC:** its stdlib metadata follower authenticated
the actual existing model and supervisor handles and saw **6/72** closed-journal
entries. That is execution coverage only. The follower used **10.62 MiB private
RAM** and 19.09 MiB working set, with Below Normal priority. A simultaneous
device snapshot was 100% busy at 3,679/4,096 MiB; the existing model supervisor
sample showed about 6.83 GiB available system RAM. These are instantaneous,
shared-machine observations, not peaks, throughput or correctness claims.

The follower waits without importing a numerical library or starting another
model. At admission, the single CPU-only audit requires at least 5 GiB available
RAM, a 4 GiB reserve, a 0.75 GiB private cap, one numerical thread and Below
Normal priority. A two-second self-only watchdog enforces the limits; no Torch,
CUDA, fitting or model forward. If the model ends during overlap, the audit
stops itself to yield to the canonical serialized host, without automatic
retry. If the worker ends before 72 close, it waits for the actual canonical
supervisor/host to end before auditing the available closed story prefix.
Unclosed artifacts never become grades.

The follower writes successful aggregate artifacts once. This launch has
automatic Git pushing disabled; the active goal agent will verify branch,
artifacts and provenance before committing/pushing that result. The original
RAM28 worker and its registered terminal publication path are unchanged.

BF16 remains deferred under the host-RAM policy. No early SAE/controller,
intervention, cross-precision, final-parameter or faithful-reasoning claim is
made. Roots are units; worlds, arms, tokens and path/checkpoint steps repeat.
Effect/sham floors and power remain uncalibrated: zero formal comparisons,
`REDESIGN_OR_PILOT_ONLY`. All old failures/gates remain intact. The original
168-case test stays sealed and the continuous goal stays active.

[Protocol](../../experiments/nan_storyaudit30/PROTOCOL.md) ·
[Registration](../../experiments/nan_storyaudit30/registration.json) ·
[Engineering receipt](../../experiments/nan_storyaudit30/engineering_receipt.json) ·
[Full RAM28 condition](NAN_RAM28_PROGRESS.md) ·
[Previous matched BF16/NF4 audit](NAN_CTX21_RESOURCE29_RESULTS.md)
