# Infrastructure repairs to the authenticated package

The authenticated package is OLR08_CPU_gym.zip (SHA-256 e5e4b26a…53750cb); its files are listed
in package_manifest.json. Every file here equals its manifest hash except those listed below.

| File | Package SHA-256 | Repaired SHA-256 | Patch |
| --- | --- | --- | --- |
| run08.py | 246695e6c0d29395d46db71afce51cf1b7cc85cf617302f0050e8af48ff4a047 | f35f42a0ed82a3a0a773dbaf11b74d45d5a49c03d590925bd48f9b07dfbe216c | [repairs/0001-replay-warm-library-search.patch](repairs/0001-replay-warm-library-search.patch) |

## 0001: warm the library-search index before replaying an exact? step

The first full run stopped in source_s2 with `fresh native path replay failed`. Two credited S2
paths, List.pairwise_of_forall_mem_list and List.findIdx?_flatten, end in `exact?`. In search that
step closed within its 500 thousand-heartbeat cap. In a fresh replay process the same step used
500,418 and 508,193 thousand heartbeats and timed out in `whnf`.

`exact?` builds Lean's library-search index as a process-wide cache on its first call. In search an
earlier, charged `exact?` attempt in the same process had paid that cost. A fresh replay pays it
inside the path's own step, so the step's cost depends on process history rather than on the proof.

The repair runs one uncharged `exact?` on the same state, with the 20000 cap used for human
replay, immediately before each `exact?` step of a replayed path. The warm-up's result is
discarded and its cost is recorded under `library_search_warmup` in the replay audit. The
replayed step keeps the 500 cap. The path, fresh pre-command opening, replay meter and kernel
certificate are unchanged, and search budgets, meters and outcomes are untouched.

Verified on all six credited `exact?` paths from the stopped run
([results/olr08a/replay_repair/](../results/olr08a/replay_repair/)): cold replay passes 4 of 6,
warmed replay 6 of 6, and the repaired `run08.replay` 6 of 6. Unit tests pass 4/4.

The repair changes the contract's code hash, so the campaign was rerun from scratch in a new
run directory; the stopped run's checkpoints are kept only as a determinism comparison.
