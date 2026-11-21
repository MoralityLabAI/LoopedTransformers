# FIT16 step-log defect and separate execution repair

The original FIT16 NF4 T1 condition failed after two actual optimizer updates.
Its second steps_local.json snapshot called the exclusive-create receipt
writer and raised FileExistsError. The live event records two updates; the
first-step log and preserved partial checkpoint remain local. The independent
terminal partial-checkpoint audit is still pending. No successful fit, missing
loss or new correctness result is inferred from this partial execution.

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

The replacement queue waits on the actual original NF4 worker, then runs the
original terminal audit/report plus an independent partial-checkpoint check.
Actual optimizer counters and logged-step coverage are reported separately;
no missing loss or neural response is rescued. Only then may fresh NF4 T1/T2
and matched BF16 T1/T2 start, each from its original SVD/CI seed with the same
sixteen-step criteria, all controls, one-thread/Below Normal limits,2-GiB
saved-tensor CPU cap and85% CUDA cap. No model retry occurs inside the queue.

[The replacement stream](NAN_FIT16_IO18_PROGRESS.md) keeps this attempt separate
from [the original stream](NAN_FIT16_PROGRESS.md). Frozen XREF17 will not be
patched to accept replacement checkpoints; their SAE analysis requires a new
registration. All earlier failed gates remain unchanged, no donor transfer
is enabled, and the original168-case test stays sealed. The continuous goal
remains active.
