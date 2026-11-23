# LSA-02B: Block-order geometry in a Lean tactic specialist

Status: **implementation and prospective gate** (2026-11-23). No new Lean-certified model-performance outcome is reported here. The implementation is a frozen-weight intervention, a matched training switch, and a provenance-checked data interface. This is the first executable follow-up to [LSA-02](lsa_typed_architecture_calculus_v0.md).

## Exact architecture intervention

Use the existing [PPT-B](../research/lean_cgrt_hrm400/PPT_PLAN_B_INKLING_BLOCK_MESSAGE_MEMORY.md) middle blocks, \`middle_a\` and \`middle_b\`, whose weights recur at every pass. Hold input processing, prelude, coda, policy heads, parameter objects, number of block calls, step depth, initial state, and source menu fixed. The two dispatch operators are

\[
\mathrm{AB}(z,m)=B(A(z+G_A(z,m))+G_B(A(z+G_A(z,m)),m)),
\]
\[
\mathrm{BA}(z,m)=A(B(z+G_B(z,m))+G_A(B(z+G_B(z,m)),m)).
\]

The previous-pass message stays the symmetric average of both middle states in either order. The \`normal\` mode follows training and is the main frozen-weight intervention; \`zero\` removes cross-pass message content while executing the same gate calls. Training can now select either block order with \`--middle-order ab|ba\`. The default AB call remains unchanged. The trained weights are frozen during the paired probe.

For every public input, independently estimate
\[
C(z)=\frac{\|B(A(z))-A(B(z))\|_2}{\max(\|z\|_2,10^{-8})}
\]
at the state immediately after the prelude and before the first message gate. This is a **label-blind** geometric descriptor, measured without root outcomes or gradients. Record AB-vs-BA logit distance in both message modes and the per-root effect on the selected candidate. Commutator magnitude is a candidate explanatory descriptor of order sensitivity, not a certificate of superior accuracy.

## Data and custody

The primary intended source is a DataGoblin **checked-candidate pack**, imported by \`ppt_gym.data.load_checked_pack\`. Its private receipts certify previously checked full-goal-closing Lean tactics, and the loader rejects mismatched kernel verdicts, forbidden axioms, and altered manifest bytes. Reusing these receipts measures *retrospective tactic selection*, not new proof discovery.

Split roots by theorem lineage. Keep training, calibration, and locked test lineages disjoint. A scored root requires all candidate labels definitive and at least one one-call proof witness. Other roots remain in the output with their exclusion reason; report the denominator. The model sees goal text, candidate strings, and its own state. It never receives receipt labels. One selected candidate costs one decision; native search, multi-tactic continuations, and future proof generation are separate tasks.

The synthetic fixture remains an **engineering-only** path. The probe reports \`scientific_eligible=false\` for any synthetic fixture, even when label accuracy is high.

## Gate before locked evaluation

Use calibration first, with a checkpoint trained on the same source manifest. A candidate checkpoint passes the fixed competence gate only if:

- its source is \`lean_kernel_receipt_replay\` and its parameter checkpoint is bound to the same manifest SHA-256;
- the training metadata records positive completed training steps;
- at least **32 independent eligible witness lineages** are available;
- the checkpoint's trained-order one-call close rate is **at least 0.40** and exceeds the exact candidate-uniform baseline by **at least 0.10**.

The locked \`test\` split is inaccessible through the runner without the successful calibration manifest for the same checkpoint and source. For a genuine architecture-order comparison, train **three paired independent seeds** under the same data order, optimizer exposure, width, number of unique blocks, forward visits, and registered wall/compute limits; one set executes AB and the other BA. The currently available harness runs one checkpoint at a time and writes raw paired records; multi-seed inference and checkpoint-to-checkpoint claims need a separate registered aggregation stage.

The training-order comparison and the frozen-order intervention answer related questions. The former tests whether a specific schedule trains more effectively under a matched budget; the latter asks whether the already-trained function depends on block order. A frozen intervention cannot by itself predict which training schedule is preferable.

## Primary outcomes and failure boundaries

For each eligible independently sampled lineage, record AB and BA success under the **same frozen checkpoint**, resulting in a paired difference in one-call proof closure. Report root-paired 95% descriptive bootstrap intervals, winners/losses/ties, choice disagreement and exact random-menu baseline. Separately measure the label-blind commutator and its rank correlation with absolute logit displacement; the latter is a *mechanism-screening descriptive statistic*, not a proof-completion claim.

For three-seed matched training comparisons, register a minimum useful **+0.025** improvement in closed proof proportion relative to matched AB, with calibration-only variance and cluster-aware uncertainty using the [FSI-01 gate](evalmath_finite_sample_gate.md) before exposing test data. The first analysis must report both positive and negative root-paired effects, complete trained-order accuracy and model FLOPs. If no checkpoint clears competence, the outcome is \`BLOCKED_INCOMPETENT_CHECKPOINT\`, and any measured noncommutativity remains structural evidence only.

This follows the [LSAD v1 chance-level audit](../reports/checkpoint_lsad_architecture_discovery_v1_addendum_1.md): near-chance dispatch comparisons are inconclusive. It also keeps native certified proof closure separate from retrospective ranking, as specified in the [PPT gym runbook](../research/lean_cgrt_hrm400/PPT_GYM_RUNBOOK.md).

## Commands

From \`research/lean_cgrt_hrm400\` with CPU PyTorch available:

\`\`\`bash
python -m unittest ppt_gym.tests.test_lsa02b_order -v

# Engineering wiring only; result is always labelled synthetic
python -m ppt_gym fixture --out runs/lsa02b-fixture --per-family 9 --seed 31
python -m ppt_gym train --fixture runs/lsa02b-fixture --allow-synthetic \
  --plan b --middle-order ab --width 16 --heads 4 --max-bytes 32 \
  --steps 2 --depths 2 --max-seconds 90 --out runs/lsa02b-ab
python -m ppt_gym.order_probe --fixture runs/lsa02b-fixture --allow-synthetic \
  --checkpoint runs/lsa02b-ab/checkpoint.pt --split calibration \
  --out runs/lsa02b-order-smoke
\`\`\`

For native retrospectively checked packs, run:

\`\`\`bash
python -m ppt_gym.order_probe --pack PATH_TO_CHECKED_PACK \
  --checkpoint PATH_TO_TRAINED_PPT_B_CHECKPOINT \
  --split calibration --depth 3 --out PATH_TO_FRESH_CALIBRATION_RECEIPT
\`\`\`

Only after passing that gate and securing the registered matched training seeds, use the locked test route:

\`\`\`bash
python -m ppt_gym.order_probe --pack PATH_TO_CHECKED_PACK \
  --checkpoint PATH_TO_TRAINED_PPT_B_CHECKPOINT \
  --split test --depth 3 \
  --gate-manifest PATH_TO_FRESH_CALIBRATION_RECEIPT/manifest.json \
  --out PATH_TO_FRESH_TEST_RECEIPT
\`\`\`

Every output includes \`paired_roots.jsonl\`, \`manifest.json\`, source and checkpoint hashes, score denominators, and zero query-label training updates. The new CI workflow uses a synthetic fixture for program correctness, not research scoring. No existing frozen trial or checked receipt is modified.

## Next source requirements

A genuinely independent test needs a source pack with \`train\`, \`calibration\`, and \`test\` splits; paired AB/BA training checkpoints from three seeds; model and sampler hashes; retained checkpoints; and pinned Lean receipts for every positive closure. Archive calibration-gate manifests before opening held-out labels. If only calibration data exist, this project stops at a development/causal-swap analysis.
