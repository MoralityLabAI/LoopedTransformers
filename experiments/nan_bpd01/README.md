# Nanbeige phase conditioned causal fibers

NAN-BPD-01 is a development-only instrument study on the pinned Nanbeige4.2-3B
checkpoint. The source registration is `18b102e`; the isolated grading and
storage-only direction amendment are `5d21ddb`. Read [PROTOCOL.md](PROTOCOL.md),
[EXECUTION_NOTES.md](EXECUTION_NOTES.md) and [RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md).
The separate Dark Fibers addendum reports the results and blocked stages.
No original ARG-03 test records are opened or regenerated.

## Exact local commands

The active checkout is the isolated checkout below. The Compactification
adapter is its separately pinned ARG-02 worktree, not that repository's main
branch. Existing weights are used offline and checked against the ARG-02 lock.
The chat-template file additionally has SHA256
`82316b70213c65e5f820c03d5a11561ccda4055bda984cb770ed0d51fe497c56`.

```powershell
Set-Location 'C:\Users\patri\Documents\Codex\2026-11-17\yes-nanbeige4-2-3b-at-4\work\nan-bpd01\LoopedTransformers'
$GPU = 'D:\codex-artifacts\NanbeigeQ4Pilot-20261117\venv\Scripts\python.exe'
$CPU = 'C:\Users\patri\Documents\Codex\2026-11-15\coe-09-testing-whether-the-probe\work\.venv\Scripts\python.exe'
$Snapshot = 'D:\codex-artifacts\NanbeigeQ4Pilot-20261117\snapshot\451ed48c3273ecef7ea8faaa43c31ce529763bb1'
$env:HF_HOME = 'D:\codex-artifacts\NanbeigeQ4Pilot-20261117\hf_home'
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

& $CPU -X utf8 -m unittest discover -s experiments/nan_bpd01/tests -v
& $GPU -X utf8 -m experiments.nan_bpd01.validate_gpu --snapshot $Snapshot --out runs/nan_bpd01/engineering
& $CPU -X utf8 -m experiments.nan_bpd01.check_components --bpd-root '..\BPD' --out runs/nan_bpd01/engineering/components_fixture.json
& $GPU -X utf8 -m experiments.nan_bpd01.cohort --evalmath-root '..\EvalMath' --compactification-root '..\Compactification-Theory-arg02' --snapshot $Snapshot --out runs/nan_bpd01/cohort
& $CPU -X utf8 -m experiments.nan_bpd01.verify_sources --cohort runs/nan_bpd01/cohort --out runs/nan_bpd01/cohort/independent_verification.json

# Run sequentially; the 3B processes must not overlap.
& $GPU -X utf8 -u -m experiments.nan_bpd01.run --compactification-root '..\Compactification-Theory-arg02' --snapshot $Snapshot --cohort runs/nan_bpd01/cohort --backend bf16 --out runs/nan_bpd01/bf16
& $GPU -X utf8 -u -m experiments.nan_bpd01.run --compactification-root '..\Compactification-Theory-arg02' --snapshot $Snapshot --cohort runs/nan_bpd01/cohort --backend nf4 --out runs/nan_bpd01/nf4
& $GPU -X utf8 -m experiments.nan_bpd01.check_effective_nf4 --bpd-root '..\BPD' --out runs/nan_bpd01/engineering/effective_nf4_fixture.json

& $CPU -X utf8 -m experiments.nan_bpd01.grade --evalmath-root '..\EvalMath' --runs runs/nan_bpd01 --out runs/nan_bpd01/graded
& $CPU -X utf8 -m experiments.nan_bpd01.analyze --runs runs/nan_bpd01 --graded runs/nan_bpd01/graded --out runs/nan_bpd01/analysis
```

The actual initial BF16 run uses the `18b102e` direction-storage version;
subsequent NF4 uses `5d21ddb`. The first model launches inherited default CPU
thread settings, rather than the optional single-thread settings shown for
CPU checks above. Figures use the isolated analysis Matplotlib installation,
not a modification of the working CUDA environment.

If a launch reaches the 1800-second soft bound with pending fixed pairs,
repeat that same model command with `--resume`. It rejects changed source
hashes, preserves the first attempt and runs only the remaining pairs. It
never creates replacements. Receipt totals include both attempts. A completed
output directory is protected against an accidental fresh overwrite.

For memory telemetry on Windows, run `environment.ps1` and the separate
`watch_memory.ps1` observer with access to CIM. The observer takes the model
PID from its manifest and writes sampled DedicatedUsage and SharedUsage.
CUDA allocated/reserved peaks are reported by the model process itself.
Successful completion or zero observed paging is not a residency certificate.

## Artifacts and evidence

`receipts/` holds compact machine-readable receipts and paired outcome tables.
Full model-only JSON/activation tapes remain under `runs/nan_bpd01/` locally;
no weights or hidden-state tensors belong in Git. Each backend has a separate
source/quantization manifest and checkpoint verification gate. Source labels
are used only by the source constructor and independent grader.

The proposed BPD donor-recovery primary is unavailable until native clean
competence and state closure both pass. Unrun learned factors, gradients,
Gramians and routing are reported as blocked, never behavioral zeros. The
CPU rank-8/16 fixtures establish numerical implementation properties only.
