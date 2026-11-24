# NAN-READOUT-34 source-only registration

The native NF4 story block's DEV compute arm returned 1/18 correct final rooms
and 0/18 fully correct packets. Supplying the complete trace gave 14/18 final
rooms and 5/18 complete packets. READOUT34 separates executing the source from
reporting particular fields and copying the complete answer. Its greedy
decoding runtime, tokenizer verification and GPU admission are separate,
pending work.

## Fixed development roster

The independent sampling unit is the generated source root. The roster has
12 roots (3 CAL, 9 DEV), stratified by actual room, recorded belief and nested
recorded belief. Each root appears at 8 and 96 events with the same initial
layout and queried roles. The two worlds per length differ in exactly one
event destination, and the final correct room changes. This gives 48
root/length/world contexts. The context ladder changes the history and the
checkpoint positions together, so its contrast describes these two difficulty
conditions.

Each context has four readouts crossed with compute and copy, plus an erased
final-room control. The fixed roster contains 432 slots per precision, 108 CAL
and 324 DEV. NF4 and BF16 will use the same 432 slots from checkpoint revision
`451ed48c3273ecef7ea8faaa43c31ce529763bb1`. CAL and DEV are retained separately.
Repeated precisions, lengths, worlds, readouts and controls contribute no
additional independent roots.

| Readout | Requested fields | Matched diagnostic |
|---|---|---|
| Final room | Room after the last event | Standalone final result and the same field in the packet |
| Single checkpoint | Room at the middle checkpoint | Standalone result and the same checkpoint item in the array and packet |
| Checkpoint array | Three checkpoint rooms in order | Per-item and all-three exact results in array and packet |
| Complete packet | Three checkpoint rooms, final parent path and final room | Field and all-field exact results |
| Erased final | Final-room format with initial state and events removed | Final compute control, with both registered identical twin prompts retained |

Copy supplies the complete canonical answer for the requested readout. Compute
and copy use identical story and query text, with the copy instruction and
answer appended. Compare format completion and field correctness across those
arms with exact denominators. For each fixed split, query, event count,
precision, arm and readout, report registered slots, verified slots, failures
and unknowns. Format failure in a verified response is incorrect for all its
requested fields. Missing or unverified responses are unknown. For each
endpoint with `s` successes, `u` unknowns and `n` registered slots, retain
the bounds `[s/n, (s+u)/n]`. Also report both-world exact results per root and
length with unknown bounds, retaining a both-world failure as soon as either
world is verified incorrect. Include paired NF4/BF16 coverage before assigning
a descriptive matched difference. Report each checkpoint item and the
complete-array endpoint separately.

The source audit reconstructs all 21 view maps from rendered compute and copy
prompts, checks the one-event A/B edit, and checks the production stop parser
against an independent field parser. The only whole-reply normalization is
trimming ASCII spaces, tabs and line breaks at the edges. A valid reply has
the exact requested fields and an `END` suffix, and is at most 512 ASCII
characters. There is no answer extraction from prose. The fixed examples
show syntax and use the same example values across roots.

## Runtime status and evidence limit

The source manifest freezes the generator, literal prompts, roster order,
model-side parser, separate oracle/grader and source checks. The source audit
uses Windows counters, one numerical thread and Below Normal priority. Its
private-memory cap is 128 MiB with 3.5 GiB available for admission and a 3 GiB
reserve. It runs for at most 120 seconds with a self-only watchdog. It imports
no tokenizer, numerical package or model and reads no experiment replies.
Raw replies, per-slot grades and case IDs remain local in later stages.
The original 168-case ARG03 test is sealed.

A later runtime freeze must bind the chat template and exact pinned-tokenizer
inputs, confirm all canonical replies fit the common token budget, establish
one-model serial admission after the existing queue, and define the matched
worker, terminal integrity and validated reporting. The proposed generated
token cap is 96 for both precisions and all readouts. Source registration
leaves inference disabled.

The primary descriptive contrast is final-only compute versus final-only copy
on the same DEV worlds, separately at each length and precision. Secondary
descriptions compare shared fields across readouts, the erased final control,
twin success and matched precision. This is exposed development with one
checkpoint, 9 independent DEV roots, zero planned formal comparisons, no
physical intervention or dose, and no calibrated sham or meaningful causal
floor. No mechanism power is estimated. The verdict is
`REDESIGN_OR_PILOT_ONLY`. Verified compute competence on new development roots
would motivate separately registered physical contrasts and sham calibration
before any donor-transfer experiment.

From the repository root on Windows, regenerate and verify the persisted
source manifest with:

```powershell
& C:/Python312/python.exe -X utf8 -B -S -m experiments.nan_readout34.freeze
```

[Native NF4 story audit](../../reports/nanbeige_bpd_gym/NAN_STORYAUDIT30_SUMMARY.md) ·
[EvalMath FSI-01](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/finite_sample_inference.md)
