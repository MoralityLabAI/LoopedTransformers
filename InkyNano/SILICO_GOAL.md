# Goal: deliver InkyNano cloud artifacts for local LDT re-routing

Work in the PR #341 branch research/inky-nano-prep. Read GYM.md, SOURCE_AUDIT.md and REGISTRATION.md. Produce a hash-bound artifact package; do not train the LDT on paid cloud time.

Use one GPU only, 45 GPU-minute pilot ceiling. Start with metadata inventory and CPU tests. Pin Transformers and source checkpoint. Ensure adequate RAM/VRAM and disk; stop if the native full-model reference load exceeds budget. Check quantized extraction against the upstream conversion implementation and record native parity, including real activation samples if available. Record tolerances from repeat measurements before testing. Then run a bounded pilot with rank-256 nested factors for 16 experts (four physical layers), compare FP16 and packed INT4 on identical weights and inputs. Use the supplied exporter as a starting implementation; fix native loader, hook or format incompatibilities in the branch, with tests and receipts. A 45-minute cap is a maximum, not a target.

Prepare a public/synthetic roster with independent source families, 16 roots initially, then 128 train / 64 dev if the cost gate permits. Each JSONL row needs root, turn, family (independent source grouping, not broad task category), split, source_kind, source, prompt and continuation. Record task category in provenance. Keep teacher-generated continuations and reference-forced continuations separate. Capture prefix-verified native routing. Run gym_data.py to export whole-turn and rolling-window demand targets. Provide a small sample and shape/coverage checks first.

Example (choose tolerances from native repeatability, replace placeholders):
```sh
python -m unittest discover -s InkyNano -p 'test_*.py' -v
python InkyNano/inventory.py --remote --output inventory.json
python InkyNano/gym_export.py --inventory inventory.json --roster public_roster.jsonl --out pilot --rank 256 --format fp16 --parity-tol "$PARITY_TOL" --prefix-tol "$PREFIX_TOL"
python InkyNano/gym_data.py --roster pilot/roster.jsonl --trace pilot/trace.jsonl --out pilot
python InkyNano/gym_manifest.py pilot --exporter-commit "$(git rev-parse HEAD)"
python InkyNano/gym_manifest.py pilot --verify
```

Return pilot manifest, tensor inventory, spectral shard(s), source/activation parity metrics, native trace, causal demand arrays and cost/quality receipts. Include exact Hugging Face revision if user authorizes publication. Use the user's namespace, no secrets in source. Keep weights separate from lightweight routing data so the 3050 can download just the gym initially. Report estimated full atlas runtime and cost; only execute full --atlas with user approval and an independently bounded resource plan. Include nonexpert tensor extraction/repacking design and native restricted-router integration gap. Leave local controller training to the 3050.

Acceptance requires successful native execution, verified layer/pass coverage, prefix invariance, no split overlap, native numerical parity, rank errors, actual stored bytes and SHA-256 receipts. If any fails return a precise blocker with partial artifacts; do not claim success from synthetic CI.
