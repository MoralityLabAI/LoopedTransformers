# Native NF4 gradient prerequisite

Read [PROTOCOL.md](PROTOCOL.md). No optimizer steps or donor transfer. Sources
are the first two COMP14 CAL roots, both worlds, fixed focused recipe.

Before any GRAD15 pretrained call:

    python -B -S -m unittest discover -s experiments/nan_grad15/tests -v
    python -B -S -m experiments.nan_grad15.freeze

Push the freeze, then start the finite queue with the existing pinned tools:

    C:\Python312\python.exe -X utf8 -B -u -S -m experiments.nan_grad15.queue --after-pid <actual-COMP14-supervisor-PID> --python D:\codex-artifacts\NanbeigeQ4Pilot-20261117\venv\Scripts\python.exe --cpu-python C:\Users\patri\Documents\Codex\2026-11-15\coe-09-testing-whether-the-probe\work\.venv\Scripts\python.exe --snapshot D:\codex-artifacts\NanbeigeQ4Pilot-20261117\snapshot\451ed48c3273ecef7ea8faaa43c31ce529763bb1 --compactification-root ../Compactification-Theory-arg02 --bpd-root ../BPD --freeze-commit <pushed-full-commit> --publish

Set OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS and
NUMEXPR_NUM_THREADS to1 before launch. The queue waits on the actual upstream
process handle, then checks both original focused first-token screens and
refuses another known model worker. One large model, Below Normal priority;
the separate math process is untouched. Existing run directories are protected
against replay. Raw archives stay local. The original168 cases stay sealed.

Runtime source locks use exact registered Windows bytes. Linux CI validates
separately recorded exact Git bytes; it does not loosen the runtime check.
The continuous goal remains active after this finite prerequisite.

[Report stream](../../reports/nanbeige_bpd_gym/NAN_GRAD15_PROGRESS.md)
