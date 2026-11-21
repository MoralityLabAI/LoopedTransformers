# NAN-PRIME-06: actual local Prime model rollouts

This is a separately registered full-reply development condition. NAN-GYM-05
native prefix-stop results and their calibration-only SAE recipe stay frozen.
Its BF16 outcomes were visible when designing this condition. These are exposed
development diagnostics, with zero formal inferential comparisons.

Use the official verifiers SDK at ed70053b6a7a4079d6976e03bdcd7d420356d594
in isolated Linux Python, its real vf-run, SingleAgentEnv, subprocess runtime,
null tool-free harness, trace recording and task reward. Every model request
goes to a Windows native checkpoint worker on this machine's WSL adapter.
No Prime cloud runtime, hosted inference, paid job, or fake SDK import is used.
The engineering mock returns the fixed invalid text MOCK_ONLY without a model;
mock traces are explicitly excluded from model results.

Run all 162 existing registered DEV contexts per condition: 105 main contexts
and 57 copy controls, grouped in 57 source roots. They come from the hash-bound
combined public prompt file, including 16 local routing trajectories. There is
no outcome-based selection, new corpus sampling, CAL inference or test access.
The original 168-case ARG-03 test remains sealed. Routing is environment-category
classification, not solving source benchmark questions. These microbenchmarks
and shared roots do not constitute a standard hosted benchmark or IID sample.

Run NF4 and BF16 separately, in that order, with only one large model resident.
Use the existing verified checkpoint 451ed48c3273ecef7ea8faaa43c31ce529763bb1,
Windows Torch 2.6.0+cu124 / Transformers 4.46.3 / bitsandbytes 0.50.2 environment,
CPU BF16 embeddings and output head, NF4 double-quantized resident decoder or
staged BF16 decoder. Never install the Linux SDK's newer Transformers into the
model environment. Native recurrence is two full passes, final norm each pass,
no KV cache, global full-vocabulary greedy B1 readout, CPU head chunks 8192,
and 75% allocator cap. No model download, finetune, edit or BPD transfer.

Accept exactly one unchanged registered user message. Render the pinned native
chat template with enable_thinking=False, add_generation_prompt=True, then append
the same newline Answer: suffix used in NAN-GYM-05. Reject system messages,
tools, unknown prompts, changed sampling or model identities. No host labels or
graders are imported by the worker. SDK task gold remains with the evaluator.

Unlike native prefix stopping, generate until the pinned EOS or the schema token
cap: label 16, grid 192, plan 96. Do not parse during generation. EOS is removed
from returned text; other special tokens and whitespace are preserved. The SDK
global max_tokens is 192 and the registered schema imposes its smaller local cap.
Score the complete final reply once with the bounded anchored native parser and
native reference/plan grader. No explanation scanning, answer repair, constrained
decoding, second answer or pooling with prefix-stop accuracy. Report format and
correctness separately, exact Sudoku reference plus alternate valid solutions,
and main/copy counts by suite and pointer hop. No new competence gate or causal
claim is inferred from software transport validation.

One rollout per task, max concurrency one, deterministic unshuffled source order,
one agent turn, agent setup timeout 120 seconds and rollout timeout 600 seconds.
Official episode/agent retry defaults remain zero; any inner transport retries
are exposed in the request/trace audit and never reused as independent cases.
Identical prompts may legitimately appear in several registered contexts;
generate anew for each occurrence and preserve a completion identifier, prompt
hash and multiplicity ledger. There is no response replay cache. Any transport
retry or excess request remains visible and is audited against official traces.
Stop after all registered requests or a 7200-second service budget, allowing an
in-flight bounded generation to finish. Preserve incomplete/error attempts;
never silently replace them. Report actual denominator and coverage.

Record full local request/reply, greedy IDs/scores and first-prompt physical
last-token states at layers 5/11/17/21, both passes, residual and o_proj/down_proj
inputs/outputs. These can be scored with the already CAL-fitted dictionaries in
physical coordinates; no SAE refitting or DEV-based dictionary selection.
Do not equate independently fitted BF16/NF4 feature indices. Weight-only SVD
remains a baseline, never a trained VPD/BPD result.

Freeze source/config/adapter hashes in SOURCE_MANIFEST.json and commit/push them
before this condition's first pretrained inference. Address/port, ephemeral auth
token, process IDs and output paths are operational metadata, not prompt choices.
Bind a private WSL adapter only, authorize that WSL guest and local access, reject
arbitrary request fields and inputs, expose no file/tool endpoint. An authenticated
controller-only shutdown endpoint permits cleanup after runner termination;
in-flight bounded generation finishes and an incomplete receipt is preserved.
All routing
text, traces, raw outputs, IDs/tokens, physical states and per-record codes stay
local. Publish aggregate scores, reconstruction/loop diagnostics, hashes and
source/transport audit counts only. Model endpoint completion and official runner
completion must both be verified before claiming coverage.
