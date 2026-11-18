# NAN-COMP-02: fresh source-only competence pilot

The frozen [protocol](PROTOCOL.md) and [config](config.json) define a new four-operator unary task. This is a development diagnostic following NAN-BPD-01, not a continuation of its compound-program gate. The original ARG-03 168-case test must remain sealed. BPD transfer is disabled regardless of this pilot's outcome.

Prospective registration: [2ec61819220d39db2dc3561d59708124e5703cf9](https://github.com/MoralityLabAI/LoopedTransformers/commit/2ec61819220d39db2dc3561d59708124e5703cf9), committed and pushed before reading checkpoint outcomes. [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) pins the inherited decoder, EvalMath, ARG-02 adapter, tokenizer and fresh source files. [Report](../../papers/shared_weights_dark_fibers/NAN_COMP02_COMPETENCE_REPORT.md).

## Replay the published evidence without model weights

The 20 synthetic worlds in fixtures are newly generated CAL/DEV sources, not the protected ARG-03 test. Receipts contain all 40 model-only case tapes, full-vocabulary normalizers, eight selected answer-token logits, unrestricted argmax IDs, class scores, prompt hashes, memory samples, grades and all nine comparisons.

    python -S -m experiments.nan_comp02.audit_receipts --out audit.json
    python -m unittest discover -s experiments/nan_comp02/tests -v

The stdlib audit uses an independently implemented finite-series checker and imports neither Torch nor EvalMath. Unit tests require Torch and NumPy. The CPU workflow pins Torch 2.6.0 and NumPy 2.2.6. The measured Windows inference environment used Python 3.12.5, Torch 2.6.0+cu124, Transformers 4.46.3, bitsandbytes 0.50.2 and NumPy 2.5.5.

## Optional full local reproduction

Use a new empty output directory. Never point these commands at any old ARG-03 roster. No download is needed or performed by the worker. The checkpoint revision is 451ed48c3273ecef7ea8faaa43c31ce529763bb1; all locked files are verified before loading. Set HF_HUB_OFFLINE=1 and HF_HOME to the existing pinned cache. Use EvalMath and Compactification-Theory checkouts at SOURCE_MANIFEST revisions; the ARG-02 adapter branch is required.

    python -m experiments.nan_comp02.validate_gpu --snapshot <pinned-snapshot> --out <new-runs>/engineering
    python -m experiments.nan_comp02.cohort --evalmath-root <EvalMath-checkout> --compactification-root <ARG02-checkout> --snapshot <pinned-snapshot> --out <new-runs>/cohort
    python -m experiments.nan_comp02.run --backend bf16 --compactification-root <ARG02-checkout> --snapshot <pinned-snapshot> --runs <new-runs>
    python -m experiments.nan_comp02.run --backend nf4 --compactification-root <ARG02-checkout> --snapshot <pinned-snapshot> --runs <new-runs>
    python -m experiments.nan_comp02.grade --runs <new-runs> --evalmath-root <EvalMath-checkout> --out <new-runs>/graded
    python -m experiments.nan_comp02.analyze --runs <new-runs> --out <new-runs>/analysis --plot

Run BF16 and NF4 sequentially. Keep the 75% CUDA allocator cap and 1800-second per-launch soft limit. Only one hash-bound resume is permitted for pending fixed pairs. On Windows the optional per-process GPU allocation sampler needs permission for CIM performance counters; these are sampled allocation counters, not a paging measurement. Local inference used D:/codex-artifacts/NanbeigeQ4Pilot-20261117/venv/Scripts/python.exe and the existing snapshot under that directory. Exact launch arguments are preserved in each receipt's commands field.

The pre-data strict planner has no empirical pilot and honestly refuses that input; actual eight-root source-use and repeated-erasure sham effects are separately archived. No historical-task shams were substituted.

## Readout limitations

There are eight DEV roots (16 worlds), two CAL roots (four worlds), and one pretrained checkpoint. All formats, loop depths and backends share roots. One CAL/DEV clean prompt and one erased root prompt coincide because the generator's public hash includes query coefficients that are not rendered. This is disclosed in the source audit; CAL was not used to fit or choose any format. The 16 DEV clean prompts and eight DEV erased root prompts are distinct. No source was replaced after inference began.

An accepted first-token operator must verify public demonstrations and the hidden degree-32 future. Four-choice ranking is diagnostic only. With just four candidates, search@12 is a trivial structural ceiling. Fresh repeated-erasure forwards calibrate numerical/format repeat noise only; they do not calibrate a semantic causal-intervention tail. Eight roots cannot supply the 59 independent sham worlds needed for a 95%/95% sample-maximum bound. No unconditional p-values, equivalence, recurrence geometry or BPD mediation are claimed.
