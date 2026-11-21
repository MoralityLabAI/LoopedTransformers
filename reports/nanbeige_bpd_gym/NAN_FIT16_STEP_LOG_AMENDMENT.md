# FIT16 step-log defect and separate execution repair

Both original FIT16 NF4 T1 and T2 conditions failed after two actual optimizer
updates each. Their second steps_local.json snapshot called the exclusive-create
receipt writer and raised FileExistsError. The original worker ended naturally;
both terminal and independent partial-checkpoint audits passed. Each partial
archive has finite arrays, correct factor shapes and optimizer counters2;
each steps log has1 record and1 missing logged loss. No loss or response is
reconstructed. No successful fit or new correctness result is inferred from
these partial updates. Original BF16 remains unrun under the amendment.

This was an integration error in the inherited logger binding. The tiny CPU
stock-trainer fixture had exercised sixteen optimizer steps but bypassed the
file-writing path. Its original narrower evidence scope is preserved.

Before changing supervision, [the amendment](../../experiments/nan_fit16_io18/SUPERVISION.md)
was pushed at `9689206dc4c9c390e189cdd7ffd19bc735ceb9a2`. Only authenticated,
task-owned FIT16/XREF17 supervisors were stopped. The actual original NF4
worker was left running its predeclared T2 condition. Original BF16 and XREF17
are **unrun under this amendment**; this prevents repeating the known writer
defect or analyzing unaudited live checkpoints. Original frozen sources,
partial updates, logs and checkpoints are retained. The math process was
neither stopped nor reprioritized.

The separate [FIT16-IO18 protocol](../../experiments/nan_fit16_io18/PROTOCOL.md)
and complete source were frozen/pushed before any replacement model execution
at `9dfb16c015dd072135e3c6cfcc7df56ce468e8f2`, seal
`7bb76c5c895848cf12643312609a584ca0797bc87e9a05e320b8cfa2d04f4830`.
The sole fitting-recipe change binds the repeated step-log call to FIT07's
existing atomic writer. Reverse AST comparison restores the original recipe
exactly. All numerical config values and public/host source fixtures are
unchanged; no partial checkpoint is warm-started. Three contracts passed,
including the actual selected writer over sixteen successive snapshots and
reproduction of the original second-write failure.

The replacement queue verified actual original NF4 worker termination, then
completed the original terminal audit/report and independent partial-checkpoint
check before starting the fresh NF4 worker. Matched BF16 is queued. Actual
optimizer counters and logged-step coverage remain separate; no missing loss
or neural response is rescued. All replacement T1/T2 fits start independently
from their original SVD/CI seed with the same
sixteen-step criteria, all controls, one-thread/Below Normal limits,2-GiB
saved-tensor CPU cap and85% CUDA cap. No model retry occurs inside the queue.

[The replacement stream](NAN_FIT16_IO18_PROGRESS.md) keeps this attempt separate
from [the original stream](NAN_FIT16_PROGRESS.md). Frozen XREF17 will not be
patched to accept replacement checkpoints. The separate
[XREF19 registration](../../experiments/nan_xref19/PROTOCOL.md), freeze
`a5fc0713503c934e29ebd7c7e2f8da7554d7127b`, binds replacement audits with unchanged
analysis/inventory/numeric settings. Its follower waits for actual IO18
supervisor termination; no numerical cache analysis has started.
All earlier failed gates remain unchanged, no donor transfer
is enabled, and the original168-case test stays sealed. The continuous goal
remains active.
