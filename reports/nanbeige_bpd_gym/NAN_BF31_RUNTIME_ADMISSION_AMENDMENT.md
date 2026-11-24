# BF31 runtime admission amendment before inference

Source revision2 corrects an identity bug found **before any BF31 worker or
pretrained forward**. The original worker wrote `backend='nf4'` into its BF16
runtime record; the existing admission rule correctly requires `bf16`. It
would reject that record. This is a prospective finding, not a failed model
response or an observed CUDA inference result.

The corrected worker writes `bf16` and calls that unchanged admission rule
before importing the Torch backend. An independent AST check removes exactly
that literal correction and the early gate's import/call, then requires exact
equality to the archived original worker. Decoder/head/loop, source roster,
tokenizer/parser, generation, batching, budgets, BPD/SAE assets, numerical
criteria, grading and all other worker statements remain unchanged.

Only the authenticated unstarted BF31 queue was retired. Its actual process
handle ended; no BF16 runtime/start marker or worker-launch event existed.
The entire original waiting attempt, publication/predecessor metadata and
original critical-source snapshot were archived. The original freeze/worker
are also retained in the source tree and Git history. The NF4 model,
supervisor and story follower were unchanged; the math process was untouched.
The replacement queue uses a fresh artifact root and is activated only after
the revised complete freeze is pushed and checked. It waits on the same
creation-bound NF4/supervisor/follower handles, with one large model at a time.

Five new stdlib-only contracts passed against the actual worker entrypoint and
actual admission function with synthetic metadata. They prove the corrected
identity passes; missing predecessors, wrong PID and mismatched publication
are rejected before Torch; and unrelated worker changes fail equivalence.
There was no Torch/NumPy import, model loading, neural response or gold read.
Maximum sampled private RAM was **14.711 MiB**, minimum available **4.336 GiB**,
under the separate 128-MiB / 4.25-GiB-start / 4-GiB-reserve fixture policy,
one thread and Below Normal priority. These are samples, not continuous peaks.

The original **24-contract / full synthetic host fixture** proof remains
historical, with its original source hashes and receipt preserved. That full
fixture was not rerun for this amendment. The source-freeze procedure checks
all unchanged original hashes and explicitly verifies the worker equivalence;
only worker/freeze bookkeeping and protocol/amendment additions are allowed.
The existing 24-contract CPU suite is checked again by CI before the revised
queue is activated; the new five-test job installs no numerical package.

Revised source seal:
`fbd500eb3a34cd52300a1a47f6c4151811190103dfa4a62e82ed3e688312a3a0`.
Original source seal:
`15525236bbae38f664480bb17c17c77665887159ce8b6e8e31ff02b1d0ed944e`.

All **1,549 baselines / 216 genuine B2 A/B pairs** remain in the same order,
matched to the unchanged RAM28 NF4 condition. No task/answer selection, retry,
orphan rescue, refitting, donor transfer or competence-gate repair. The 3,962
coverage slots remain represented, with unverified outcomes unknown. Actual
BF16 CUDA capacity, correctness, memory and matched geometry remain unrun.
Formal comparisons0, `REDESIGN_OR_PILOT_ONLY`, original failures and the
168-case held-out seal remain. The continuous goal stays active.

[Amendment bindings](../../experiments/nan_bf31/RUNTIME_ADMISSION_AMENDMENT.json) ·
[Revised complete seal](../../experiments/nan_bf31/SOURCE_MANIFEST.json) ·
[Original seal](../../experiments/nan_bf31/SOURCE_MANIFEST_before_runtime_admission.json) ·
[AST/admission checks](../../experiments/nan_bf31/admission_amendment_tests.py) ·
[Full BF16 plan](NAN_BF31_PLAN.md)
