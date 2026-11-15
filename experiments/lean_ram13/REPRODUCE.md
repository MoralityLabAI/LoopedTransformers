# RAM-13 reproduction: frozen original corpus and native Lean proof search

**Status:** first registered 12-theorem source-family proof comparison complete, 156 policy/theorem cases, 114/114 accepted cases independently compiled a third time. This is a negative recurrent-only result.

Public source was compiled with Lean4.35 at `patrickdugan/MathCan@f427e0ef4a61093304d12ca2c0dc943d8d2429e5` using [public MathCan GitHub Actions run 43721176783](https://github.com/patrickdugan/MathCan/actions/runs/43721176783). The Actions artifact was streamed into three SHA-pinned parts because the full portable toolchain exceeded file-transfer limits. The source and compiler binary hashes were rechecked locally before scoring. No private RAM08/RAM12 weights were uploaded to MathCan.

The running model and RAM12 registered first-run head checkpoints were loaded from `ram08-frozen-audited-train-43689874169.zip` and `RAM12_certified_native_states_and_rankers_20261115.zip`, both verified by their original SHA256 receipts. The posthoc RAM12 strict-rejection heads are excluded.

## Local exact reproduction

The archived first execution ran under `/mnt/data/ram13_local` with the source at `/mnt/data/ram13_portable/mathcan`, portable Lean at `/mnt/data/ram13_portable/lean/bin/lean`, the original RAM08 ZIP at `/mnt/data/ram08-frozen-audited-train-43689874169.zip`, and registered first-run RAM12 ZIP at `/mnt/data/RAM12_certified_native_states_and_rankers_20261115.zip`. The primary result code intentionally retains the absolute paths used in the original run, and the original evidence archive retains the source bytes and all generated Lean proof sources, logs and receipts.

The native persistent Lean server is the same RAM03 probe implementation used in prior RAM campaigns; its code is `ProbeServer.lean` in this directory. Compile it first using the pinned source and `LEAN_PATH` pointing to the pinned project `.lake/build/lib/lean` and the harness directory.

```bash
python test_grammar_replay.py --ram08 /mnt/data/ram08-frozen-audited-train-43689874169.zip
python harness.py --smoke
python harness.py --n 12
python secondary.py --start 0 --stop 12
python independent_audit.py --start 0 --stop 156
python build_receipt.py
```

The scorer refuses silently overwriting scored policy outputs. If an external terminal timeout interrupts one case before producing a receipt, preserve/quarantine the incomplete directory, then invoke `run_one.py` for that exact same frozen theorem and policy; the original per-case clock starts again, and the interruption receives zero scored result or model credit. The recorded run kept four such interrupted non-scored source workspaces outside the final case matrix.

The summary receipt `RAM13_RESULT_20261115.json` and individual `RAM13_CASES_20261115.jsonl` include original source and certificate hashes, exact policy budget, every failure reason and the registered negative result. The full original 114 × 3 proof source/log evidence is available in the separate signed SHA-manifested evidence ZIP supplied with the original research conversation; it excludes the private model weights and training examples.

Reproducibility caveat: this is procedural new-source-family transfer over twelve public MathCan theorems, with dependencies and thematic neighborhoods shared among them. Each accepted proof was separately verified by native Lean4.35. This report does not change the older RAM08 PEF or RAM10 registered tests, both still unscored.
