# LSA-02I: Explicit-cost native Lean branch reconsideration

**Registered November 24, 2026.** CPU-only development experiment, before a full Lean reasoning gym. [LSA-02H](../reports/lsa02h_recoverable_offpath_cpu_20261123.md) certified 93 new recovery states but found zero discriminating tactic forks. The next falsifiable mechanism is reversible proof search when an accepted branch ends without available tactics or cycles to a previously visited public state.

## Search semantics and fairness

The fixed model uses Plan B, 57,475 parameters, two AB recurrent passes and the pp96 typed/public-pretty view under 384 bytes. Its online selector sees only public Lean goals, syntactically available tactics, Lean's verdict on prior attempts and its own search history. The generator's proof hints, shortest-path graph, dead-branch annotations and correct tactics are never given to the model.

Each proof search stores a native Lean state identifier, the current accepted proof prefix and tried tactics on a stack. A **RECONSIDER** action pops one frame and restores the prior native Lean state, preserving its list of already attempted tactics. It is legal only after all tactics have been tried at the current state, or after the current public state repeats an ancestor. The final proof script omits abandoned branches.

Every Lean tactic attempt costs **one search-work unit**, including an accepted tactic whose continuation later fails. Each RECONSIDER costs **one unit**. The fixed total budget is **16 units** per theorem for every arm. Model inputs with more than eight remaining units use the encoder's existing capped remaining-budget feature of eight. Model calls, Lean tactic attempts, reconsiderations, successful proofs, and resource costs are reported separately. A legal but ultimately unprovable branch can only be identified using the post-scoring Lean graph.

## Frozen source and model training

Freeze a new Lean-CGRT G03 theorem suite at generator seed **20261202**, with 64 independent roots in each source split. Only the 64 training roots supply verified shortest-continuation supervision. Native Lean 4.19 must independently compile every reference proof. Reserve **32 disjoint calibration theorem roots** for development; leave ordinary and OOD test outcomes sealed.

Train two fixed model seeds, **173 and 179**, under identical pp96 input, two AB passes, 720 AdamW updates, FP32, width 32, four attention heads and the original 57,475 parameters. Duplicate each training fit on the *same physical GitHub CPU worker*, with one Torch thread, deterministic operator selection and disabled MKLDNN. Exact equality of tensors, training sample streams and optimizer traces is mandatory. No evaluation-root updates are permitted.

## Matched native policies

| Policy | Tactic selection | Search-work budget |
| --- | --- | ---: |
| frozen_greedy | Frozen Plan B highest-ranked available tactic, without stack rollback | 16 |
| frozen_dfs | Exactly the same frozen Plan B, with charged RECONSIDER for exhausted or cyclic branches | 16 |
| auto_dfs | Model-free public-order heuristic with the identical rollback machinery | 16 |
| uniform_dfs | Model-free root-seeded uniform tactics with the identical rollback machinery | 16 |

The first three policies are evaluated on the *same 32 roots within each model seed*. Uniform DFS has five fixed random streams, **17, 31, 47, 61, 83**, yielding 160 episodes on the same 32 roots. The random selector uses only the public packet, remaining menu, fixed root seed and remaining budget.

Every accepted tactic receives a real Lean 4.19 verdict. Every completed script must satisfy native kernel closure with no extra axioms and separately compile as a complete standalone Lean declaration. All action and RECONSIDER journals are independently replayed from fresh Lean processes with zero model calls. Replay checks the candidate menu, public state fingerprints, predecessor stack, exact work costs and surviving theorem script. The native shortest reference proof is computed only after all live policies finish.

After those model runs, a **post-scoring** Lean graph audit classifies accepted legal transitions into finite and unreachable successors. Its labels may never influence online decisions or fit updates.

## Registered outcomes and decision gates

The primary endpoint is **full-proof completion under frozen_dfs minus frozen_greedy**, paired within seed and theorem, with the common 16-unit budget. Treat the 32 unique theorems as the independent statistical unit. First average paired differences across the two seeds within each root, then compute a descriptive 2,000-draw theorem-root bootstrap interval.

**Development screen:** frozen DFS must gain at least **+0.05 absolute proof-closure rate** over greedy and be nonnegative at each seed. Report the auto and uniform DFS controls, effort/work costs, backtrack counts, tactical attempts, proofs recovered after dead branches and actual Lean attainable ceiling. Null and negative outcomes remain valid recorded findings.

The separate established **competence floor** requires 32 independent roots, at least **40% native certified full-proof completion**, and at least **+0.10 above the five-stream native uniform DFS baseline**. Even a positive result needs separately registered sealed confirmation, FSI-01 sham and source-ancestry checks. A pass does not directly authorize scaling, gym construction or Mathlib transfer claims.

## CI and custody

The [GitHub Actions CPU workflow](../.github/workflows/lsa02i-native-backtrack-cpu.yml) builds the frozen Lean teacher, trains both frozen checkpoints with duplicate receipts, runs two model evaluations and an independent random DFS control, verifies their script certificates and independent Lean replay, then performs the post-scoring source/graph audit and root-paired report. At most two model evaluations run concurrently, with bounded CPU/RAM and no GPU or paid inference.

Store every theorem/episode receipt, source hash, checkpoint SHA-256 and verifier certificate as Actions artifacts with fourteen-day retention. Commit a permanent research report after green CI. Ordinary-test and OOD family outcomes stay unopened.
