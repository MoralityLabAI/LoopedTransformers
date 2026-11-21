# Official Prime20 and matched native development execution

Sixteen reused DEV roots, two counterfactual worlds and three arms give96 registered official episodes per backend. Each complete reply contains one unconstrained greedy native loop2 token. Only independently audited official episodes receive grades; pending, missing and orphan replies stay explicit.

| Backend | Terminal execution status | Neural replies | Recorded official grades / 96 | Missing grades | Failed official episodes | Orphan replies | Exact native prefixes | Byte-exact vectors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 | completed | 96 | 96 / 96 | 0 | 0 | 0 | 96 | 3840 |
| nf4 | completed | 96 | 96 / 96 | 0 | 0 | 0 | 96 | 3840 |

A successful recorded grade means the episode completed and is auditable; its answer can still be wrong. The vector comparison includes both loops/layers5/11/17/21 at every registered physical site,40 vectors per prefix. This identity is measured against the same backend and initial prompt, without claiming cross-backend quantization or mechanism equivalence.

| Backend | Arm | Official graded / 32 | Official correct among graded | Missing | Both-correct roots / complete pairs | Whole-roster accuracy bounds | Native first-token correct / 32 |
|---|---|---:|---:|---:|---:|---|---:|
| bf16 | focused | 32 / 32 | 32 | 0 | 16 / 16 | 1.00000 to 1.00000 | 32 / 32 |
| bf16 | copy | 32 / 32 | 32 | 0 | 16 / 16 | 1.00000 to 1.00000 | 32 / 32 |
| bf16 | focused_erased | 32 / 32 | 2 | 0 | 0 / 16 | 0.06250 to 0.06250 | 2 / 32 |
| nf4 | focused | 32 / 32 | 29 | 0 | 13 / 16 | 0.90625 to 0.90625 | 29 / 32 |
| nf4 | copy | 32 / 32 | 32 | 0 | 16 / 16 | 1.00000 to 1.00000 | 32 / 32 |
| nf4 | focused_erased | 32 / 32 | 2 | 0 | 0 / 16 | 0.06250 to 0.06250 | 2 / 32 |

Bounds range over missing registered outcomes and are not confidence intervals. Native reference scores remain separately labelled; they never fill an absent official grade. Full coverage requires exact aggregate world and both-world-root agreement with the same-backend COMP14 first-token endpoint. The bounded six-token endpoint and later generation histories are not substituted.

| Backend | Server elapsed seconds | CUDA peak allocated GiB | CUDA peak reserved GiB | Head device | Target unchanged | SDK outer-whitespace trims |
|---|---:|---:|---:|---|---|---:|
| bf16 | 2104.25 | 2.36624 | 2.41016 | cuda:0 | True | 0 |
| nf4 | 1401.41 | 2.59343 | 2.68945 | cuda:0 | True | 0 |

Server elapsed includes loading, qualification, SDK waits, inference and closing checks; it is not GPU kernel time. Allocator allocated/reserved peaks are distinct from driver residency. NF4 keeps all22 decoder layers and the unchanged BF16 output head on CUDA; BF16 four layers plus a streamed stage and the same head. CPU embeddings, greedy B1, two loops, no KV cache. One numerical CPU thread/Below Normal protects the separate math process.

## Scope and preserved failures

This is exposed development execution validation on one checkpoint, not independent task replication or a new competence screen. Roots, worlds, arms, loops, native/SDK runs and backend conditions are correlated. Primary contrast:official complete one-token reply versus same-backend COMP14 initial-token response. Physical neural intervention dose0; scientific causal floor, matched-sham floor and causal power are uncalibrated. Planned/actual formal comparisons0, verdict REDESIGN_OR_PILOT_ONLY. New independent source/readout roots and calibrated causal controls would be needed for confirmation.

The pinned official local SDK uses subprocess runtime/null harness, one turn/no tools, temperature0/top-p1, no shuffle and max-concurrent1, with no hosted/paid inference. Terminal audit binds task/source IDs, raw assistant nodes, exact prompt IDs, usage/finish, both phase scores, frozen target hashes, full-head parity and every physical vector. The SDK may strip outer whitespace only in its last_reply accessor; the raw node remains bound.

Original partial PRIME06/PRIME-CTX10 grades remain missing; no lost response, old tail or original competence gate is repaired. There is no SAE fit, online component edit or donor transfer in this condition. Raw replies, tokens, states and per-record grades stay local. Actual process handles and authenticated health supervise live work without ordinary reads of live atomic progress or SDK trace files. Observation expiry never creates a fresh neural attempt. The168-case test stays sealed and the continuous goal remains active.

Source commit `4a0d4dae7f97fb5df3379f57f8b49edab970c67d`, report SHA-256 `a19af6bdb1dc708412e7d717ad1b14af58daa349a1c400c1557709548454f6ad`. Native comparator `9db0d4c973fe27662c6b75805d2cabdff9a10bc9`.

[Immutable official aggregate](https://github.com/MoralityLabAI/LoopedTransformers/blob/4a0d4dae7f97fb5df3379f57f8b49edab970c67d/reports/nanbeige_bpd_gym/nan_prime20_progress.json) · [Live terminal-results stream](NAN_PRIME20_PROGRESS.md) · [Native comparison](NAN_COMP14_RESULTS.md) · [Frozen protocol](../../experiments/nan_prime20/PROTOCOL.md) · [Presentation receipt](nan_prime20_results_receipt.json)
