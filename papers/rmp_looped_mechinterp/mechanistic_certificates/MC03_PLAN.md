# MC-03 CPU-trained development pilot

2026-11-11. Frozen before this pilot's training or held-out evaluation. Base:
`ce51d20bafb14f1a933b0f510e8091fdb984af6b` (MC-02).

The original Q4/Q5 checkpoint files are unavailable in this runtime. This pilot
trains new small instances of the unchanged native LoopedChase class. Its results
must be labeled `MC03_CPU_TRAINED_PILOT`; they are not Q4/Q5 replication results.

## Fixed training contract

- Native model: n_nodes=6, width=48, one block, four heads; unchanged prompt
  injection and native head. No requested-depth token.
- Seeds: 201 and 202. Four training loops, per-loop supervision lambda=1.
- AdamW, learning rate 0.001, betas (0.9,0.98), weight decay 0.01, gradient clip 1.
  Warmup 100 updates, then cosine decay to 5% of the initial learning rate.
- Batch size 64, 3,000 updates per seed. CPU-only, one torch thread. A cooperative
  wall-time limit of 240 seconds per seed stops at a completed update and records
  `BUDGET_STOP` rather than silently dropping or restarting that seed.
- Enumerate the 720 permutations on six nodes. Sort by SHA-256 of canonical JSON
  containing the domain tag `mc03.graph_split.v1` and the permutation. First 480
  are training graphs, next 120 calibration, remaining 120 test. All starts and
  edge-order variants of a graph stay in its split. The native token layout is
  retained. Draw training graphs, starts and edge orders from seeded generators.
- Save initialization and final weights, configuration, optimizer/update budget,
  source hashes and complete loss records. No best-checkpoint selection. Use the
  final completed update for evaluation, including training failures.

This is an intentionally supervised positive-control learning task. The exact
transition labels are supplied during training. Neither the model nor its native
head receives an oracle at evaluation time.

## Evaluation fixed before results

Evaluate every start on all 120 held-out test graphs (720 graph/start items),
with a seeded edge order and up to 16 loops. Report accuracy by visit, exact full
trace validity, and first-error depth. Compare with the saved random initialization.
A depth-4 accuracy of 0.90 is a descriptive learning gate; all seeds and all
metrics remain reported regardless of this gate. Test outcomes never alter the
training schedule, split, hyperparameters or checkpoint choice.

Apply MC-02's unchanged cursor-transfer assay on held-out graph inputs at visits
1,2,4,8,12, with horizon 4, a 32-pair cap and eight random directions. Freeze the
input panel and pair manifest before replay. Eligibility depends only on current
endpoint correctness and the declared graph-conditioned targets. Retain donor
replay, sham and constant-shift controls. Report coverage and receiver-weighted
finite trace fidelity. Keep each visit's counts separate and retain all tensors.

A failed cursor-only transfer can result from a context-dependent distributed
code or from destructive interventions. Direction-matched controls and oracle
correctness support interpretation. The experiment does not identify which
explanation holds from failure alone.

## Execution and evidence boundaries

Fetch the full original model and SEM-ECHO dependency files through a source
artifact, verify their Git blob identities, and execute those complete modules.
Local excerpts are not sufficient for this pilot's source claim. Additive pilot
code will provide graph-disjoint data and call the existing native transition,
readout and MC-02 intervention functions.

CPU feasibility and code correctness can be tested without training labels or
held-out model scores. Changes required to fix implementation bugs must be recorded.
The two training seeds and their budgets remain fixed. No expensive cloud training
or external model download is authorized by this protocol.

The symbolic checker reuses MC-01. STWO proving, garbled evaluation, neural-source
cryptographic proofs and Lean checking remain separate work. A trained result is
empirical evidence for the specified model/task/intervention contract.
