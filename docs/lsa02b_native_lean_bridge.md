# LSA-02B Native Lean Bridge: paired frozen block-order intervention

**Status:** executable exploratory development protocol, 2026-11-23. Retains the original [LSA-02B block-order protocol](lsa02b_block_order_protocol.md). This runner checks *whole-proof completion* against native Lean rather than predicting retrospectively stored one-call candidate labels. Any native CI test with a synthetically trained checkpoint is a wiring demonstration.

## Source audit and selection

The unmerged [PPT-G04 PR #281](https://github.com/MoralityLabAI/LoopedTransformers/pull/281) contains meaningful Lean 4.19 proof-checking evidence for Plan A/B with CPU-only online learners. Its [native pilot](https://github.com/MoralityLabAI/LoopedTransformers/blob/fdfb136a5ed8211325bd4b85bfb5cbb73bad8e2a/research/lean_cgrt_hrm400/PPT_G04_RESULTS.md) reports:

| Condition | Observed native result | LSA-02B suitability |
| --- | --- | --- |
| RAM-04 Micro B, 3 roots | 1 of 3 at depth 1/2/4 for all three message modes, 0 of 3 at depth 8/16 | Weak developmental competence, not an order experiment |
| RAM-05 persistent Micro A | 117 episodes independently replayed; source is native | Trains a different Plan A architecture, so cannot be loaded as Plan B |
| Seed-908 fresh Micro A | 3/12 SGD versus 4/12 frozen and retrieval at final dose | Negative/plausibility control, insufficient for a BA-superiority hypothesis |
| Archived G04 report | generated cohorts, episode ledgers, independent proof audits and curves | Public archive lacks a directly loadable \`ppt_gym.checkpoint.v1\` Plan B weight file |

The [PPT residual-transfer archive at commit \`3df3d5e\`](https://github.com/MoralityLabAI/LoopedTransformers/blob/3df3d5efaff77686bcc46190fe01c05dd053da2b/research/lean_cgrt_hrm400/PPT_RESIDUAL_TRANSFER.md) concerns a fitted two-slot proof policy with its own state, message and tactic grammar. Its single-seed reward transfer has a positive result on repaired directed proof roots, but the saved fit is **a different architecture and task contract**. Interpreting its scores as a pretrained \`PlasticProofTransformer\` AB/BA weight intervention would be invalid. Both archives remain read-only inputs to future, explicitly typed adapters.

The native bridge instead reuses the checked [PPT-G03 Lean 4.19](../research/lean_cgrt_hrm400/PPT_G03_RUNBOOK.md) mechanism already in \`main\`:

- \`ppt_gym.live_suite\` freezes independent generated theorem lineages, their generator source hash and each split file digest.
- \`ppt_gym.live_lean.run_episode\` executes public-menu tactics; Lean returns the actual successor state and kernel proof term.
- \`lcgrt.certify.certify\` recompiles successful scripts as independent declarations with an axiom whitelist.
- \`ppt_gym.live_lean.replay_episode\` re-executes every retained action against a fresh Lean state without consulting the model.

The initial runner only admits the **calibration** split. Locked \`ordinary_test\` and held-out \`ood_family\` execution must follow a separately registered competence and finite-sample gate. All benchmark ancestry checks outside the locally generated suite are pending.

## Counterfactual experiment

At each source theorem root \(r\), compare the **same immutable checkpoint** at the same recurrent depth and tactic budget:

\[
\widehat{Y}_r^{AB}
=\mathbf 1[\mathrm{Lean\ standalone\ certifies\ script\ selected\ by}\;AB],
\quad
\widehat{Y}_r^{BA}
=\mathbf 1[\mathrm{Lean\ standalone\ certifies\ script\ selected\ by}\;BA].
\]
The native reference \(\widehat{Y}_r^{\mathrm{auto8}}\) uses public action ordering with the same eight-tactic allowance.

Both neural arms apply identical named parameter tensors, embeddings, prelude, coda, and heads; two middle-block calls occur per recurrent pass. Their previous-pass memory uses the same symmetric message rule. We alternate which order executes first across roots and independently reset the Lean process state on every arm. Only the internal block dispatch order changes.

For \(N\) fully paired independent roots:
\[
\widehat\Delta_N = N^{-1}\sum_r(\widehat{Y}_r^{AB}-\widehat{Y}_r^{BA}).
\]
Report paired roots, win/loss/tie counts, verified closure counts, attempted tactics, model calls, and a **descriptive** root bootstrap interval. Each independent theorem lineage is one observation. The script retains every step, public action menu, cached goal, reject and proof-term hash. An aborted or partially paired root is excluded, recorded through \`timed_out\`, and cannot pass an architecture gate.

The primary purpose here is to establish whether frozen block order changes **native proof behavior at all**. A separate program must test whether training under one block order improves results. That follow-up needs at least three independent seed-matched AB and BA checkpoints trained on the same native-verified source split, plus verified developmental competence, calibrated matched-sham effect resolution, independent fresh roots and an [FSI-01](evalmath_finite_sample_gate.md) assessment.

## Evidence classes

\`native Lean 4.19\` means an actual \`lcgrt.lean_repl.LeanRepl\`, a pinned toolchain, real kernel terminal certificates, standalone file recompilation and model-free action replay. A row marked \`standalone_verified=true\` has all of those checks.

A model **trained on a synthetic fixture** may still run through *real Lean inference*. Such an outcome is evidence about the native execution interface and the chosen frozen weights, with \`scientific_eligible=false\` because the checkpoint was never shown competent on independent Lean training data.

\`FakeLean\` and its mock certifier remain unit-test tools. The test harness sets \`actual_lean_process=false\` and cannot create a \`standalone_verified\` success, even when its simulated status says that a proof closed. This prevents accidental mixing of simulation and native research outcomes.

The CLI refuses locked/OOD split scoring, explicitly checks checkpoint provenance if a checked training pack is provided, records zero query-label updates, and refuses any change to the frozen parameter bytes.

## Runbook

From \`research/lean_cgrt_hrm400\`:

\`\`\`bash
python -m unittest ppt_gym.tests.test_native_order -v

# Build the independent Lean 4.19 lgym server as in PPT-G03_RUNBOOK
python -m ppt_gym.live_cli freeze --out runs/lsa02b-suite \
  --seed 20261123 --per-split 24

# Checkpoint must be a PLAN B ppt_gym.checkpoint.v1 bundle, trained externally.
# Optional --training-pack PATH verifies a native checked-candidate source pack.
python -m ppt_gym.native_order run --suite runs/lsa02b-suite \
  --checkpoint runs/ppt-b-native/checkpoint.pt \
  --training-pack runs/native-verified-checked-pack \
  --roots 8 --checks 8 --depth 2 --max-seconds 360 \
  --out runs/lsa02b-native-paired

python -m ppt_gym.native_order replay --suite runs/lsa02b-suite \
  --session runs/lsa02b-native-paired --out runs/lsa02b-replay
\`\`\`

The new CI executes a two-root native Lean wiring smoke using an explicitly synthetic, 2-step-trained PPT-B checkpoint. Its report and independent replay receipt are archived as the GitHub Actions artifact \`lsa02b-native-lean-smoke\`. The native lean process may correctly close zero proofs; the wiring gate checks successful execution, authentic kernel provenance, matched root accounting, and replay fidelity rather than imposing a desired score.

## Next research gate

1. Produce or recover a **genuinely trained and custody-verified Plan B checkpoint**. The source ledger's native validity and one-call tactic completion competence must be tested on a disjoint development split.
2. Register a paired AB/BA training run with exactly matched widths, parameters, token and computation exposure, roots and stopping rules. Check the logits and gradients of equal initial weights before distinct schedules train.
3. Freeze prospective held-out theorem roots; use the independently compiled full-proof closure as primary outcome, and separately measure label-blind block commutators, first-action disagreement, and full-trace divergence.
4. Preserve both positive and null results. If the trained model stays below competence, record \`BLOCKED_INCOMPETENT_CHECKPOINT\` and retain the structural results.
