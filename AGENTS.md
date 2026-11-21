# Agent instructions for LoopedTransformers research

Before running or interpreting AF, LS-SAE, SMC, PEF, NRF, ROC, LOOPSCOPE, or other recurrent-depth/mechanistic evaluations, consult the [EvalMath mechanistic-power skill](https://github.com/MoralityLabAI/EvalMath/tree/main/skills/evalmath-mechinterp-power) and the companion [FSI-01 statistical evidence gate](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/finite_sample_inference.md).

The experimental unit is the independently sampled source puzzle, graph family or trained checkpoint, as specified by the hypothesis. Multiple loop depths, SAE features, intervention doses or 64 generated variants from one source are correlated measures of the same unit. Before a large GPU run, set the minimum *meaningful causal* effect, compare it to matched sham/intervention variation and instrument resolution, plan independent roots for adequate power, freeze the test family and sealed readout, and keep all controls and raw paired outcomes. A stable probe, feature correlation or large continuous logit change alone does not establish mediation of correct reasoning.

Use the merged LoopedTransformers FSI-01 bridge at `research_gym/scripts/evalmath_fsi_gate.py` for paired correctness where applicable; future-logit, state-distance and multi-dose effects require their appropriate registered endpoint and cluster-aware inference. Keep finite Lean proofs and exact synthetic enumerations in separate evidence classes. Use development-only calibration and do not open held-out ARG-03-N test data.

## Authorized local Nanbeige work

The user explicitly requests continuous, autonomous RTX 3050 work. Resolve
routine choices from the current task and available evidence; do not interrupt
an active goal with an optional question whose first option is the obvious
intended default. Prime integration defaults to the official verifiers runner
on the local 3050 with the existing pinned checkpoint and environment.
Maintain a finite queue of genuine registered research jobs, one large model
at a time, while doing CPU audits and pushing sanitized report milestones.
An observation timeout does not mean a worker ended. Ask only for genuinely
essential missing information or required authorization after independent work;
preserve frozen protocols and the original 168-case test seal throughout.

A separate math process needs CPU capacity. Use one CPU thread and Below
Normal priority for model workers and CPU analyses; serialize substantial CPU
analyses. Prefer resident GPU weights/head when the registered condition fits
the 4-GB RTX 3050. Never stop or reprioritize the user's math process.
Set OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS and
NUMEXPR_NUM_THREADS to 1 before launching new numerical processes.

On Windows, never Get-Content/read_text/ordinary-open a live worker's atomically
replaced progress JSON. A reader can deny its rename and stop inference (seen
at PRIME-CTX10 NF4 reply 51). Poll authenticated health; audit immutable
snapshots/terminal files. FILE_SHARE_DELETE alone also failed the native
replacement fixture here, so do not assume it makes progress reads safe. Preserve failures,
register supervision changes before an unstarted model condition, and never
replay lost neural responses into official SDK scores.

The user clarified on 2026-11-21 that this continuous Nanbeige goal must remain
active when a finite pilot ends. Consuming the current registered queue is a
milestone, not completion of the broader goal. Prepare the next genuine,
bounded source-only stage and freeze/push it before inference; keep the 3050
working without inventing filler or relaxing old scientific gates.
