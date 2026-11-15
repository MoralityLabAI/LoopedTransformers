# BPD-LG-01 independent Lean kernel audit

Date: 2026-11-15. The private LoopedTransformers CI job failed before executing steps. An independent GitHub-hosted Ubuntu runner in EvalMath reproduced and checked the full generated positive theorem source using pinned Lean 4.19.0.

## Provenance

- Original source: MoralityLabAI/LoopedTransformers commit ebdd76e3acacb5d68d4d736d6dc49417f26f166f, experiments/bpd_lean_curriculum/generate.py.
- Original generator SHA-256: 3b474850e3aac9c6958be8c7d33ae16b2448fbf8ef6fac965d7d795fb2f830fe.
- Independent CI: https://github.com/MoralityLabAI/EvalMath/actions/runs/43701011945
- Kernel evidence archive: https://github.com/MoralityLabAI/EvalMath/actions/runs/43701011945/artifacts/14156104018
- Validator: MoralityLabAI/EvalMath/.github/workflows/bpd-pinned-kernel.yml at commit 66eda949a6cce4f17dece7cb3fd237aa148d8398.
- Compiler: Lean (version 4.19.0, x86_64-unknown-linux-gnu, commit 6caaee842e94, Release).
- Source hash checks, kernel compilation and individual #print axioms audits all succeeded.

| Split | Kernel proofs | Axiom-free | Generated SHA-256 |
| --- | ---: | ---: | --- |
| train | 128 | 128 | c688fc711817e33633c9065569e11cafcabfca665c29013c6b5704cec43fbdab |
| dev | 32 | 32 | bb2329e9745c966e529cbf8f591276228df707c04b0f6471e291602d8ed2607f |
| test | 32 | 32 | e2376bb1cef30eb78ab93dc13853b682a740bc6f3daa12560fd010fd23a8048f |

Total: 192 positive Lean theorems kernel-checked and 192 empty axiom inventories. The generator also emits 192 negative cases with deterministic finite countermodels, yielding 384 examples in all. Negative cases are not Lean proofs of unprovability. The exported JSONL keeps kernel_verified=false because generation itself does not run Lean. This independent audit applies only to the exact pinned positive source hashes.

The four tactic labels name a successful witness in the synthetic template grammar. Other tactics may prove the same theorems. The audit establishes neither neural reasoning improvements nor mechanistic identifiability for a trained model.

## Reproduce

Generate from the pinned source commit, compare the three generated source hashes above, then compile train/dev/test positive sources under Lean 4.19.0. Run #print axioms on each declaration; all 192 must report no axioms. The independent CI artifact retains the generated files, stdout/stderr and signed-by-context JSON receipt. The private CI workflow can provide a second replay once its runner is available.
