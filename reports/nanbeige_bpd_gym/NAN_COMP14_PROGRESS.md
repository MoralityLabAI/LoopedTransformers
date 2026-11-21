# NAN-COMP-14: fresh NF4/BF16 source-conditioned competence

Eight fresh CAL and sixteen DEV roots; two permutation worlds with different answers per root. Two fixed prompt recipes, explicit copy and repeated source-erasure controls. All 240 contexts/backend use the same pinned checkpoint and initial prompts. Full-vocabulary greedy generation stops at the first anchored answer prefix, EOS or six-token bound. First-token and bounded endpoints stay separate; no old failed gate is changed.

| Backend | Status | Closed contexts | Audited contexts | Missing audited coverage |
|---|---|---:|---:|---:|
| nf4 | running_closed_case_counts_only | 81/240 | 0 | 240 |
| bf16 | pending | 0/240 | 0 | 240 |

Scores below appear only after independent terminal tokenizer/grammar/graph/state audits. Partial closed counts are operational evidence, not graded outcomes.

| Backend | Split | Arm | Endpoint | Correct worlds | Format valid | Both-correct roots |
|---|---|---|---|---:|---:|---:|

## Fixed new-condition screens

Each template/endpoint must solve both worlds in at least6/8 CAL and12/16 DEV roots, pass every copy pair, and retain byte-identical erased repeats with zero erased both-world successes. These are exposed development screening criteria. Bounded success cannot qualify a first-token fit; any later fit needs its own prospective registration and matching readout. No donor transfer is enabled.

| Backend | Template | Endpoint | New screen |
|---|---|---|---|

All first-prefix physical recurrence groups are retained in the JSON after both complete audits; later backend generation histories are not paired. There is no causal intervention, formal comparison, equivalence claim or independent sealed confirmation. Original algebraic and FIT07 gates remain unchanged. The original168-case test stays sealed.

NF4 retains all22 decoder layers and the unchanged BF16 head on GPU; BF16 retains four layers plus staging and the same head. CPU embeddings preserve capacity. One large model, one numerical CPU thread and Below Normal priority; the separate math process is untouched. Allocator peaks and sampled dedicated/shared counters are distinct.

Source seal: `b9a62f0eedaedc3eca4667412873d5c7151b6a4fb1f98b41bf1bc6978649043f`. Updated 2026-11-21T03:06:30.115120+00:00.

[Frozen protocol](../../experiments/nan_comp14/PROTOCOL.md) · [Source seal](../../experiments/nan_comp14/SOURCE_MANIFEST.json) · [Full aggregates](nan_comp14_progress.json)
