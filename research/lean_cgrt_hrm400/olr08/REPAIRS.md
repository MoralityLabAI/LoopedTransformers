# Infrastructure repairs to the authenticated package

The authenticated package is OLR08_CPU_gym.zip (SHA-256 e5e4b26a…53750cb); its files are listed
in package_manifest.json. Every file here equals its manifest hash except those listed below.

| File | Package SHA-256 | Repaired SHA-256 | Patch |
| --- | --- | --- | --- |
| run08.py | 246695e6c0d29395d46db71afce51cf1b7cc85cf617302f0050e8af48ff4a047 | after 0001: f35f42a0ed82a3a0a773dbaf11b74d45d5a49c03d590925bd48f9b07dfbe216c | [repairs/0001-replay-warm-library-search.patch](repairs/0001-replay-warm-library-search.patch) |
| run08.py | f35f42a0… (after 0001) | after 0002: 598025d86e7b3189e54e4146382d311f6beea87bfe86f97f7a5d22b9d9bc3199 | [repairs/0002-premise-timeout-no-proposals.patch](repairs/0002-premise-timeout-no-proposals.patch) |

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

## 0002: an enumerator heartbeat timeout yields no premise proposals instead of aborting

The second run stopped in cal_s1. Under the uniform-cost S1 control, the calibration root
Array.finRange_succ (Array/FinRange.lean) reaches a state whose `premises` request exceeds the
server's own 5000 thousand-heartbeat enumeration cap (5,000,210 used). The package's
`enumerate_menu` raises on any premise error, so a single enumerator timeout aborted the whole
campaign. Fable's referee search08.py read a failed reply as an empty premise list, keeping
the state's fixed, local and definition actions.

The repair restores that reading for heartbeat timeouts only: a reply with `ok` false, an error
naming the heartbeat limit, and no `rw` or `apply` field gives the state no premise proposals.
The call is still charged, and it stays in the meter's RPC journal with `ok` false, so every
timeout is counted. Any other malformed or failed reply still raises. The repair applies
identically to every arm.

Verified ([results/olr08a/replay_repair/](../results/olr08a/replay_repair/)): unrepaired, the S1
search on Array.finRange_succ raises; repaired, it completes unsolved at the 300-step cap with
3 timed-out premise calls charged. Premise-call heartbeats for the 73 source roots common to the
first two runs were identical, so the timeout is a deterministic property of that state.
