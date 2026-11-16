# AF-23 Track A: root-tied reachability and interior STOP ablations

**Registered scientific verdict: FAIL (0/8 seeds pass).** **Source-locked hosted engineering and independent audit: PASS.**

**Official source-pinned run:** [GitHub Actions #43811695657](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43811695657), source request commit `e969f8bac728327fbdced8cd850d7f1f4dd1d9fc`. All ten jobs completed: one source lock, eight independently audited frozen-checkpoint seed jobs, and independent aggregation. The run produced artifact **`af23-track-a-aggregate`** (artifact ID **14200772185**), plus eight per-seed artifacts with original checkpoint digests, complete new episode traces and audit certificates.

**Registration:** AF-23 [`PROTOCOL.json`](../../experiments/activation_funnel_af23/PROTOCOL.json), committed first at [6956443](https://github.com/MoralityLabAI/LoopedTransformers/commit/695644364518e048a83e3fdb6f1d6b5e87d4e645). Full scoring was performed after the [pure-Python branching/STOP source preflight #43810373784](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43810373784). Original AF-22 scientific source `5c16b342323977b3045f34c9be4eb24113dc691e`, eight immutable checkpoints and registered AF-22 **FAIL** are unchanged.

## Design and comparisons

Three interventions use the **same original AF-22 frozen 6,036-parameter TypedLoop checkpoint** per seed. They replace only numerical inference scoring; every new action trajectory is freshly re-evaluated using the original network and host-legal state graph.

- **Root-tied reachability:** use the root's neural reach probability at each nonterminal node in the current two-step lookahead graph. Terminal goal-state boundaries remain exactly zero.
- **No interior STOP:** remove the STOP candidate from backups at intermediate nonterminal nodes. The root STOP action remains available. If an interior node has no enabled successor, its learned leaf cost remains the boundary.
- **Root-tied + no interior STOP:** combine both changes with no learned parameters or fitting.

All original scored controls are retained: Direct, VI-1, VI-2, first-legal fixed, and shortest-proof oracle. The registered gate requires the combined intervention to improve proof success by at least **5 percentage points versus fixed**, at both held-out depths in the same seed, with **≥90% correct immediate STOP** among unprovable goals. At least **six of eight seeds** must meet the full gate, with zero false accepted Horn proofs.

Each seed evaluates the original 416 cases: 32 held-layout and 192 cases at each of depths 4 and 5. Each principal depth has **80 provable** and **112 unprovable** goals.

## Complete pooled results

**Provable goals completed within the exact shortest-proof budget plus one firing.** Denominator **640** per depth (80 cases × 8 seeds).

| Policy | Depth 4 solved | Depth 5 solved |
|---|---:|---:|
| Direct | 202/640 (31.56%) | 108/640 (16.88%) |
| VI-1 | 246/640 (38.44%) | 102/640 (15.94%) |
| VI-2 | 307/640 (47.97%) | 147/640 (22.97%) |
| Root-tied reach | **131/640 (20.47%)** | **99/640 (15.47%)** |
| No interior STOP | **307/640 (47.97%)** | **147/640 (22.97%)** |
| **Combined root-tied/no-interior** | **131/640 (20.47%)** | **99/640 (15.47%)** |
| **First legal (fixed)** | **480/640 (75.00%)** | **512/640 (80.00%)** |
| Exact shortest oracle | 640/640 (100%) | 640/640 (100%) |

**Correct immediate STOP on unprovable goals**, denominator **896** per depth (112 × 8 seeds):

| Policy | Depth 4 | Depth 5 |
|---|---:|---:|
| Direct | 854/896 | 858/896 |
| VI-1 | 884/896 | 896/896 |
| VI-2 | 848/896 | 886/896 |
| Root-tied reach | **896/896** | **896/896** |
| No interior STOP | **894/896** | **888/896** |
| **Combined** | **896/896** | **896/896** |
| Fixed | 384/896 | 384/896 |
| Oracle | 896/896 | 896/896 |

Root-tied reachability (with or without removing interior STOP) achieves perfect initial STOP on unprovable cases, while losing most of VI-2's provable proof completions. The gains in unprovable STOP discrimination come with excessive stopping on provable goals. The no-interior-STOP arm produces the **same pooled completed-proof counts as VI-2**, so it offers no registered capability improvement against first-legal.

## Seed-level target intervention: solved out of 80 positive cases

| Seed | Depth 4 combined | Depth 4 fixed | Depth 5 combined | Depth 5 fixed | Full registered gate |
|---:|---:|---:|---:|---:|:---:|
| 17 | 46 | 60 | 37 | 64 | FAIL |
| 29 | 0 | 60 | 0 | 64 | FAIL |
| 43 | 0 | 60 | 0 | 64 | FAIL |
| 61 | 12 | 60 | 0 | 64 | FAIL |
| 73 | 23 | 60 | 26 | 64 | FAIL |
| 89 | 36 | 60 | 24 | 64 | FAIL |
| 103 | 0 | 60 | 0 | 64 | FAIL |
| 131 | 14 | 60 | 12 | 64 | FAIL |

All eight seeds passed the 90% correct-initial-STOP requirement on the unprovable roster (112/112 per depth). **None passed the provable-success comparison against fixed.**

## Independent engineering and audit

- **8/8** original AF-22 checkpoint ZIPs downloaded unchanged and their source/member SHA-256 digests verified against the original published manifest.
- Original network training source hashes and Torch/NumPy checkpoint weight parity checked before any new rollout.
- **8/8** independently audited held-out seed jobs passed.
- **50,861** independently reconstructed planner-node receipts from newly taken action trajectories.
- **216** sampled checkpoint Torch forward replays, with the original AF-22 numerical tolerance of `1e-5`.
- **Zero** false accepted Horn proofs.
- Every action's numerical backup, host physical-rule premises and goal-state transition independently rechecked. The eight-seed aggregate independently verified source/episode digests, complete arm rosters and the registered gates.

The original hosted result is archived in the run's `af23-track-a-aggregate` artifact. The report is a documentation-only addition, and scoring under a later Torch runtime must remain separately labeled.

## Mechanistic interpretation

The evidence isolates STOP calibration as a central failure for these compact predictors on this particular low-branching gym. In a positive Horn closure, exact eventual reachability is invariant under legal rule firing. Root-tied reachability is therefore mathematically appropriate when the root estimate is accurate, but **it propagates whatever error exists at the root**. In AF-23 Track A, its uncorrected low-reachability estimates systematically suppress proof continuation.

Removing interior STOP leaves proof-completion totals unchanged relative to VI-2. This suggests that the dominant limitation is already present in the root STOP comparison and learned boundary estimates, rather than being fixed by suppressing interior termination branches. The new controls do not demonstrate useful neural proof-search advantage over first-legal.

The independent **Track B** high-branching gym and registered first-legal/random/oracle controls are designed to test whether the same compact architecture can perform meaningful action selection when the task actually requires search. Track A remains a final **scientific FAIL**, regardless of Track B outcomes.

## Engineering amendment record

Three source-pinned CI attempts preceding the successful run failed on reproducibility plumbing and remain visible:

1. [#43810850510](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43810850510): shallow checkout prevented Git from resolving the source-base commit.
2. [#43810990064](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43810990064): original fixed/oracle episodes have null neural traces, requiring explicit zero-forward accounting.
3. [#43811391300](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43811391300): the audit initially used exact float-list equality for checkpoint forward replay; its bound was corrected to the AF-22 verifier's registered `1e-5` tolerance.

These failed jobs occurred before complete independently passing aggregation; no thresholds, test cases, model weights or intervention scores were changed in response to outcome. All original negative results and failure receipts remain preserved.

**Research status: AF-23 Track A scientific FAIL; hosted engineering/audit PASS.** Keep [AF-23 draft PR #116](https://github.com/MoralityLabAI/LoopedTransformers/pull/116) unmerged pending Track B.
