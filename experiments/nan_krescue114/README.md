# Running the CPU verification and scoring harness

The [registration](PROTOCOL.md) fixes the target site, contrast and development go/no-go rule. The harness reuses the frozen BOUNDRESCUE109 planner and K hook, RESCUEGROUP110 parity check, and RESCUEGPU111 closed-terminal host audit.

Run from the existing LoopedTransformers checkout with the audited runtime file bytes and original pinned interpreter. When the new package is in an isolated checkout, add that checkout's root to PYTHONPATH and keep the original research checkout as the current directory. This lets the frozen parent checkers verify the local runtime byte views. Replace each uppercase path or directory below with the corresponding local path. Each output directory must be new.

    python -B -S -m unittest experiments.nan_krescue114.test_score -v
    python -B -X utf8 -m experiments.nan_rescuegroup110.verify --snapshot SNAPSHOT --work STAGED_DIR
    python -B -X utf8 -m experiments.nan_rescuegpu111.verify --mode host --snapshot SNAPSHOT --source-dir FROZEN_SOURCE --work HOST_DIR
    python -B -X utf8 -m experiments.nan_krescue114.verify --snapshot SNAPSHOT --source-dir FROZEN_SOURCE --staged-proof STAGED_DIR/verification.json --host-fixture-dir HOST_DIR --evalmath EVALMATH_CHECKOUT --work VERIFICATION_DIR

The staged check uses a randomly initialized BF16 decoder with serial and grouped execution. The two host commands use the original tokenizer and separate CPU leases. The host fixture has a manufactured change in one closed root and exercises the same receipt audit as a native terminal. The verification output cannot produce a native go.

The planned native artifact is a closed grouped BF16 terminal from the first native use of the 16 frozen development roots. Once such a terminal exists, the CPU-only scorer independently checks source, qualification, the completed journal and physical evidence. It writes the aggregate decision and private root outcomes to a new output directory. The scorer has no model execution entry point.

    python -B -X utf8 -m experiments.nan_krescue114.closed --snapshot SNAPSHOT --source-dir FROZEN_SOURCE --terminal-dir CLOSED_BF16_TERMINAL --out OUT

Use no live progress file as input. Keep private root outcomes and source payloads local. The report and the verification receipt contain aggregate information only.
