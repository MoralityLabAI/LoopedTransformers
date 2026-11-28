# MOLT-10 measured response-factor engineering results

[CPU run 45857821008](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45857821008)
executed at source head 75353ce65cd4cfb7c4f1541ebc964518bac7102a. Three
focused tests passed, fresh tiny teacher and native BPD fit passed, and
all four continuation arms and artifact upload passed. The teacher has
eight ordinary training updates; native factor fitting has two steps.
These are engineering parity receipts, not a competence experiment.

Two TRAIN anchors with 16 distinct native donor writes each. Float64,
five readout visits, hidden size 128, vocabulary 2048, native rank 8.
The largest teacher target decode error was 6.5061e-15.

| Arm | First gradient max abs error | Last gradient max abs error | Final parameter max abs error | Response-loss max abs error | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 0 | 0 | 0 | 0 | 0.485621 |
| Decode | 2.22045e-16 | 3.33067e-16 | 1.80412e-15 | 4.87649e-22 | 0.458565 |
| Gram | 2.22045e-16 | 4.44089e-16 | 2.25514e-15 | 2.06524e-22 | 0.485424 |
| Recompute | 2.22045e-16 | 3.33067e-16 | 2.25514e-15 | 1.88028e-22 | 0.603525 |

All training loss differences were reported zero (the existing trace loss
uses fp32 CE). The mean auxiliary loss was approximately 9.4364e-11 in
all arms. All ordinary validation predictions agreed and accuracy remained
5.88235%, equal to the prior constant baseline. Four updates per arm,
batch two, identical student schedule. Recompute performs two extra
teacher forwards per batch. Unit tests separately compare a nonzero
response objective and full gradients with changed student unembedding.

## Payload

| Representation | Bytes |
| --- | ---: |
| Dense response target archive | 2,621,440 |
| Normalized-state response archive | 163,840 |
| Native coordinates | 2,048 |
| Shared native chart | 8,192 |
| Separately retained centered teacher head | 2,097,152 |
| Separately retained teacher Gram | 131,072 |
| Dense plus native chart and coordinates | 2,631,680 |
| Factored plus native, head already resident | 174,080 |
| Factored plus native and separate centered head | 2,271,232 |
| Gram plus native, separate head and Gram | 2,402,304 |
| Recompute plus native, complete teacher excluded | 10,240 |

The response archive shrinks 16x in the declared exact representation.
With a separate centered head, the total listed payload drops by 360,448
bytes (13.70%). If the head is already resident, the incremental payload
is about 15.12x smaller. These counts exclude record metadata, optimizer
state, transient allocations and allocator overhead; the recompute option
requires the complete teacher. The actual comparison keeps all target
representations and parity snapshots resident simultaneously. It has not
demonstrated an RSS reduction or standalone speedup. The short measured
wall times are descriptive and include parity checking.

Strict archive break-even with separately retained head is 28 records
at these dimensions; head plus Gram requires 30. See the matching
[DataGoblin planner #95](https://github.com/MoralityLabAI/DataGoblin/pull/95)
for explicit residency and payload-cap accounting.

Next: isolate representations in separate capped processes, use a competent
teacher and enough distinct source-valid donor records, include actual
per-arm process-tree RSS and I/O, and separate engineering parity from
a preregistered training-effect experiment. No sealed evaluation or
promotion is authorized.
