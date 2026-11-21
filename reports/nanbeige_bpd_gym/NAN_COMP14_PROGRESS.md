# NAN-COMP-14: fresh NF4/BF16 source-conditioned competence

Eight fresh CAL and sixteen DEV roots; two permutation worlds with different answers per root. Two fixed prompt recipes, explicit copy and repeated source-erasure controls. All 240 contexts/backend use the same pinned checkpoint and initial prompts. Full-vocabulary greedy generation stops at the first anchored answer prefix, EOS or six-token bound. First-token and bounded endpoints stay separate; no old failed gate is changed.

| Backend | Status | Closed contexts | Audited contexts | Missing audited coverage |
|---|---|---:|---:|---:|
| nf4 | completed_audited | 240/240 | 240 | 0 |
| bf16 | running_closed_case_counts_only | 121/240 | 0 | 240 |

Scores below appear only after independent terminal tokenizer/grammar/graph/state audits. Partial closed counts are operational evidence, not graded outcomes.

| Backend | Split | Arm | Endpoint | Correct worlds | Format valid | Both-correct roots |
|---|---|---|---|---:|---:|---:|
| nf4 | calibration | copy | bounded_loop2 | 16/16 | 16/16 | 8/8 |
| nf4 | calibration | copy | first_token_loop1 | 2/16 | 2/16 | 0/8 |
| nf4 | calibration | copy | first_token_loop2 | 16/16 | 16/16 | 8/8 |
| nf4 | calibration | focused | bounded_loop2 | 15/16 | 16/16 | 7/8 |
| nf4 | calibration | focused | first_token_loop1 | 1/16 | 16/16 | 0/8 |
| nf4 | calibration | focused | first_token_loop2 | 15/16 | 16/16 | 7/8 |
| nf4 | calibration | focused_erased | bounded_loop2 | 2/16 | 16/16 | 0/8 |
| nf4 | calibration | focused_erased | first_token_loop1 | 1/16 | 16/16 | 0/8 |
| nf4 | calibration | focused_erased | first_token_loop2 | 2/16 | 16/16 | 0/8 |
| nf4 | calibration | legacy | bounded_loop2 | 14/16 | 16/16 | 7/8 |
| nf4 | calibration | legacy | first_token_loop1 | 1/16 | 16/16 | 0/8 |
| nf4 | calibration | legacy | first_token_loop2 | 14/16 | 16/16 | 7/8 |
| nf4 | calibration | legacy_erased | bounded_loop2 | 1/16 | 16/16 | 0/8 |
| nf4 | calibration | legacy_erased | first_token_loop1 | 0/16 | 12/16 | 0/8 |
| nf4 | calibration | legacy_erased | first_token_loop2 | 1/16 | 16/16 | 0/8 |
| nf4 | development | copy | bounded_loop2 | 32/32 | 32/32 | 16/16 |
| nf4 | development | copy | first_token_loop1 | 4/32 | 4/32 | 0/16 |
| nf4 | development | copy | first_token_loop2 | 32/32 | 32/32 | 16/16 |
| nf4 | development | focused | bounded_loop2 | 29/32 | 32/32 | 13/16 |
| nf4 | development | focused | first_token_loop1 | 3/32 | 32/32 | 0/16 |
| nf4 | development | focused | first_token_loop2 | 29/32 | 32/32 | 13/16 |
| nf4 | development | focused_erased | bounded_loop2 | 2/32 | 32/32 | 0/16 |
| nf4 | development | focused_erased | first_token_loop1 | 2/32 | 32/32 | 0/16 |
| nf4 | development | focused_erased | first_token_loop2 | 2/32 | 32/32 | 0/16 |
| nf4 | development | legacy | bounded_loop2 | 29/32 | 32/32 | 14/16 |
| nf4 | development | legacy | first_token_loop1 | 1/32 | 23/32 | 0/16 |
| nf4 | development | legacy | first_token_loop2 | 29/32 | 32/32 | 14/16 |
| nf4 | development | legacy_erased | bounded_loop2 | 2/32 | 32/32 | 0/16 |
| nf4 | development | legacy_erased | first_token_loop1 | 0/32 | 30/32 | 0/16 |
| nf4 | development | legacy_erased | first_token_loop2 | 2/32 | 32/32 | 0/16 |

## Fixed new-condition screens

Each template/endpoint must solve both worlds in at least6/8 CAL and12/16 DEV roots, pass every copy pair, and retain byte-identical erased repeats with zero erased both-world successes. These are exposed development screening criteria. Bounded success cannot qualify a first-token fit; any later fit needs its own prospective registration and matching readout. No donor transfer is enabled.

| Backend | Template | Endpoint | New screen |
|---|---|---|---|
| nf4 | legacy | first_token_loop2 | PASS |
| nf4 | legacy | bounded_loop2 | PASS |
| nf4 | focused | first_token_loop2 | PASS |
| nf4 | focused | bounded_loop2 | PASS |

All first-prefix physical recurrence groups are retained in the JSON after both complete audits; later backend generation histories are not paired. There is no causal intervention, formal comparison, equivalence claim or independent sealed confirmation. Original algebraic and FIT07 gates remain unchanged. The original168-case test stays sealed.

NF4 retains all22 decoder layers and the unchanged BF16 head on GPU; BF16 retains four layers plus staging and the same head. CPU embeddings preserve capacity. One large model, one numerical CPU thread and Below Normal priority; the separate math process is untouched. Allocator peaks and sampled dedicated/shared counters are distinct.

Source seal: `b9a62f0eedaedc3eca4667412873d5c7151b6a4fb1f98b41bf1bc6978649043f`. Updated 2026-11-21T03:26:57.349851+00:00.

[Frozen protocol](../../experiments/nan_comp14/PROTOCOL.md) · [Source seal](../../experiments/nan_comp14/SOURCE_MANIFEST.json) · [Full aggregates](nan_comp14_progress.json)
