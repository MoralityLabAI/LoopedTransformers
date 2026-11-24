# NAN-SERIAL-62 lookup depth pilot

[Thinking Machines Lab's variable-loop study](https://thinkingmachines.ai/news/inkling-2-mini/#no-cot-capabilities)
found that increasing pass count improved its N-Hop Lookup task even after
broader no-CoT gains began leveling off. SERIAL62 adds a controlled lookup
ladder to the Nanbeige gym. It reuses the matched one- and two-pass native
readout interface from READOUT34. These checkpoint passes use an identical
frozen head. Nanbeige has no separately trained shallow readout in this pilot.

## Matched workload

Eight independent generated roots are split into two CAL and six exposed
DEV roots. Each root reuses its two equally long paths across one, three,
and six lookups and across tables of 24 and 72 rows. A/B twins differ at
the starting edge and follow disjoint reached codes. Extra rows preserve
the short table as a subsequence. The same native tokenizer audits token
counts across the lookup ladder and twin worlds.

Both worlds have direct-answer and externally checkable trace readouts
and exact-copy controls for both forms. A direct erased control retains
the exact same prompt in A and B with different hidden answers. Trace
verification checks each reached code and reports final-code accuracy
separately. A valid written path warrants credit for that emitted path.
It provides no measurement of the model's internal algorithm.

Each backend receives a roster of 960 slots, pairing the same 480 prompts
at one and two physical passes. Each slot has 64 generated tokens.
Greedy stopping is independent of the hidden answer. Score paired depth
repairs, regressions, both correct, both incorrect, and unknown cases.
Report both-world joint success at the root level, sliced by hop count,
context size, and arm. The panels share roots. No formal inference or
learned exit policy is registered. Report phase-token visits separately
from reply length. The pilot has no BPD edits and no sealed-source access.

## Local verification

Run `python -m experiments.nan_serial62.verify --help` in the LoopedTransformers
checkout. The verification binds the fixed source, pinned tokenizer and
deterministic schedule; checks canonical answer round trips and exact
prompt token lengths between compute twins and across the hop ladder;
then replays a scripted-model fixture through the native-depth engine
interface. It retains only hashes and aggregate evidence in the tracked
receipt. The inherited tiny CPU fixture covers actual KV cache and streamed
head behavior. This stage makes zero pretrained calls. GPU execution needs
its own published lane registration, native preflight and terminal custody.
