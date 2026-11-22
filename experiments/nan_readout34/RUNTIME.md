# NAN-READOUT-34 paired-depth runtime

The original source registration uses 12 generated roots, two history lengths,
two counterfactual worlds per length, and 432 two-pass slots per precision.
The pinned tokenizer checked every prompt and canonical reply. The longest
prompt was 3,367 tokens. The longest final, single-checkpoint, checkpoint-array,
and packet answers were 7, 6, 13, and 52 tokens respectively. The common
96-token ceiling admits all four answer forms.

## Inkling-inspired depth comparison

Thinking Machines Lab's variable-loop study used the same checkpoint across
inference depths, observed depth sensitivity on serial and long-context
tasks, and measured answer quality and written reasoning separately. The
READOUT34 upgrade crosses **one and two physical decoder passes** on matched
final-only and full-packet prompts. The final-only arm tests bounded answers
without an explanation. The packet arm requires an independently checked
parent path as well as rooms. Packet paths are verifiable evidence of the
emitted answer. Internal reasoning and path faithfulness require their own
intervention studies. [Inkling 2 Mini variable-loop study](https://thinkingmachines.ai/news/inkling-2-mini/#monitorability-of-looped-transformers)

Every one-pass condition uses the same prompt, tokenizer, checkpoint and
frozen full-vocabulary head as its two-pass counterpart. One pass uses a
fresh phase-one cache throughout its own decoding trajectory. Two passes use
two separate phase caches throughout theirs. This is a depth intervention on
Nanbeige, with a shared head that was not separately calibrated for shallow
reading. The analysis therefore measures the behavior of that readout at
each depth. It does not attribute difficulty to one cause or equate Nanbeige's
phase truncation with Inkling's variable-depth training.

The one-pass comparison retains all 48 contexts and includes final compute,
final copy, erased final, packet compute, and packet copy. This adds 240 slots
per backend to the complete 432-slot two-pass design, for 672 per backend and
1,344 total. Identical erased twins remain in the roster. The two context
lengths and three queries are reported separately. All outcomes use the same
strict parse and oracle, with generation stopped on EOS, a complete reply, the
512-character cap or the 96-token cap in that order. A complete wrong answer
stays wrong. A separately verified stop and intact GPU terminal are required
for a scored result.

Report exact answers, both-world success, field accuracy, paired shallow-to-deep
repairs and regressions, both-correct, both-incorrect and unknown pairs. Also
report matched NF4-to-BF16 outcomes and, for verified rows, generated tokens,
phase-token visits and model-lane wall time. One phase-token visit counts one
physical decoder traversal on one token. Head work and host overhead are
reflected in elapsed time. Counts are descriptive and clustered by the 3 CAL
and 9 DEV roots, with zero inferential tests and no trained exit policy.

## Software evidence and device admission

The pure schedule, immutable reply closure and independent scorer are
exercised without pretrained inference. A bounded tiny-model CPU fixture runs
the production one- and two-pass cache paths, real Transformers dynamic KV
caches, and streamed full-vocabulary head. It verifies exact same-shape repeat
and first-phase boundary closure, incremental/full diagnostic logging, the
registered cache-capacity gate and exact head agreement on the tiny model.
The runtime verification receipt records the measured fixture scope and memory.

The actual RTX 3050 worker requires the source freeze on the research branch,
published runtime bytes, the original and queued handles to have terminated,
and the CACHE32 model and host terminal receipts. The new backend runs
sequentially by an exclusive GPU lane. The preflight tests each physical depth
through the maximum scheduled B1 position of 3,463, including an incremental
step, before generating a scored reply. It records incremental/full-logit
differences without making legacy full-prefix equivalence a gate. Pin and
verify target weights and quantization state before and after generation.
The old jobs' frozen sources and artifacts continue under their existing
registrations. Actual CUDA preflight and READOUT34 scores are pending.

The worker consumes the private tokenizer receipt under
`runs/nan_readout34/tokenizer01`. The run must use the pinned snapshot, approved
GPU venv, and Compactification-Theory adapter. Invoke
`python -m experiments.nan_readout34.runtime --help` for the required
arguments and choose NF4 first, followed by BF16 after the NF4 process exits.
Run `python -m experiments.nan_readout34.score --help` for the host-only
aggregate step after the GPU lanes close. Each attempt uses exclusive files,
with the data on the local artifact disk. Only aggregate results and custody
hashes are published.

[Source protocol](PROTOCOL.md) · [Pinned-tokenizer verification](TOKENIZER_VERIFICATION.json) ·
[Runtime software verification](RUNTIME_VERIFICATION.json)
