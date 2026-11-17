# LS-SAE06: prospective causal transfer of singular modes into Sudoku predictions

**Registered November 16, 2026 before any LS-SAE06 heldout hidden states.** Prior source: LS-SAE05 found high-gain Jacobian singular input directions on two fresh Sudoku families (8/8 primary events each). Their output tangent energy mostly propagated beyond the original edited token. This experiment scores all 81 Sudoku logits and verified solution correctness, with a computation-matched logit-optimized direction.

## Original model, immutable source fit, new Sudoku

Use the unmodified original gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att file step_21700 SHA-256 **932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3**, in LoopScope02 PureSudokuTRM, FP32 PyTorch 2.7.1 CPU, four threads, math attention. Reuse the original *preheldout calibration-only* source-fit PCA 512→48 and top-four SAE48→96 NPZ from LS-SAE04 GitHub Actions run 43818817318, artifact 14203533086. Require exact NPZ file SHA-256 **073b7771676e44eb934c50bdf1e7639ffeadbdff12998979cfe4c43e42b2da25**, manifest SHA-256 **cb44b910f94210745b7a7df304e575d5a194ee8cd7bec51a88281fccc95bf33a**, array SHA **83dacde8f13926fc41fc48cb17bcfba8c399538754efb7104e41e94bd93797ec** and physical dictionary SHA **14dc1a8b4130a1cbbf215e97a4a0d1c4004d23ded319546b6bf35ddc3238f5eb**. No fitting or choosing weights.

Generate independent unique-solution noncanonical 9×9 Sudoku using the registered randomized MRV solver, positive alternating 2×2 rectangle invariant, and dig seed=full seed+1000000. New solution/input pairs must be distinct from ALL LS-SAE02/03/04/05 source, pilot and heldout families. Pilot-only: 2 30-hole at seeds **202626020000..202626020001** and 2 34-hole at seeds **202626021000..202626021001**. Registered full test: **8 30-hole at 202626030000..202626030007** and **8 34-hole at 202626040000..202626040007**. Record entire rosters and canonical JSON SHA. One source injection site per puzzle: median missing cell in visible mask, sorted index floor(n_missing/2). Engineering pilot is excluded from registered full gates.

## Clean source and four equal-norm arms

Capture true native z_H and z_L after one full ACT step. Edit exactly ONE source z_H board token (512 dimensions), retaining the other 96 z_H tokens, full z_L, embeddings and all frozen model weights.

Select directions before inspecting correct solution digits:
1. **STATE_POWER:** original LS-SAE05 exact *four JVP and four VJP updates* maximizing the full 97×512 hidden z_H norm after two further ACT steps, seeded by original literal LS-SAE05|family|seed|power-start. Score one further forward JVP; record four complete JtJ power transcripts.
2. **LOGIT_POWER:** exactly four JVP and four VJP updates maximizing the *81×9 centered digit-logit* norm at the same 2-step horizon, seeded with first 63 bits of SHA256 literal LS-SAE06|family|seed|logit-power-start. Center each cell's nine logits by their own mean to remove softmax-invariant additive offsets. Preserve all four normalized 512D source vectors and full reverse cotangents. This has a matched four-step structural search budget. It uses clean predicted logits, not task answers or labels.
3. **SAE:** strictly highest positive top-four source feature's 48D physical decoder column, lifted by original 512×48 PCA and normalized; gap to runner >1e−8, else label-blind abstention. Do not require SAE feature reencoding stability after intervention.
4. **RANDOM:** independent isotropic unit 512D Gaussian seeded SHA256 literal LS-SAE06|family|seed|random-final. Same physical dose as every eligible arm.

At horizon ACT3, score full hidden JVP norm and centered all-81-logit JVP norm for each selected direction. A source direction and its code are frozen before inspecting heldout task labels or clean task correctness.

## True finite physical perturbations and verified solutions

Use physical L2 radii exactly **0.05** and **0.20**. For each of four arms and both radii, insert **both signs ±r** into the same source token of native z_H after ACT1, and run actual frozen recurrence to **ACT3** (two further ACT steps) and **ACT6** (five further ACT steps), with an original unperturbed clean trajectory. Archive all 81×9 digit logits at every horizon, sign, arm and radius; infer no missing logits from selected target-only scores.

For each horizon and variant, independently derive from logits:
- centered-logit L2 effect: [||center(logits(+)−clean)||F+||center(logits(−)−clean)||F]/(2r), symmetric secant ||center(logits(+)−logits(−))||F/(2r), and positive/negative count of argmax-changed cells across **all 81**;
- blank-cell correct prediction counts against unique exact Sudoku solution (mask from original visible input), given-clue disagreements, full 81-cell exact match, number of 27-row/column/box conflict repetitions, and validity of the predicted complete grid against original givens;
- correct-to-wrong and wrong-to-correct transitions on blank cells, split between original edited goal token and all other cells;
- truth-digit-minus-strongest-clean-nontarget margin per missing cell, preserving the clean rival separately at each horizon and in all edit arms, plus paired ± truth-margin changes.

The full 81×9 centered logit JVP at ACT3 is independently computed from the native model and compared to the symmetric finite 0.05 secant, with a recorded numerical residual; nonfinite or unsupported JVP is an execution error. All labeling and scoring occur after source directions are selected without target answers.

## Preregistered scientific gate and inferential scope

Primary question: does a STATE_POWER singular direction change more **discrete missing-cell digit predictions** at ACT6 under equal-dose physical edits than an independent RANDOM direction?

For direction u at radius **r=0.20** define F(u) as one half of (number of originally blank cell argmax predictions different from clean under +r) plus (same count under −r). Define paired D=F(STATE_POWER)−F(RANDOM), with F(missing/failed STATE_POWER)=0 and all original test puzzles included in denominator. No choosing edit sign based on task output.

Per fresh 8-puzzle family, compute **2000** puzzle-ID bootstrap resamples with replacement from D, random.Random(202626130+len(family)); sorted 95% percentile endpoints indices **50** and **1949**. A family passes if **≥6/8 puzzles have D≥1.0** and the lower paired mean 95% bound is **strictly positive**. Both families must pass for overall registered success. Preserve all original raw counts, no threshold/split changes after test.

Separate descriptive outcomes include blank-cell *correctness gain* averaged across both signs versus clean, verified valid exact Sudoku solves (must match all original clues and the unique solution), and the LOGIT_POWER computation-matched benchmark's finite 81-cell logit effects at ACT3 and ACT6. A strong logit effect without verified correctness improvements is an output-sensitivity finding rather than a reasoning-capability claim. Source SAE and source RANDOM arms are included with identical physical edit radii.

## Independent receipt audit and limitations

A separate standard-library replay must verify original checkpoint/calibration NPZ/manifest SHA, fresh complete-grid Sudoku overlap and uniqueness, frozen median visible blank, all 512D unit direction norms, four STATE_POWER and four LOGIT_POWER normalized JtJ transcripts, all original 81×9 logit arrays and centered differences, prediction flips, valid Sudoku units and clues, true-digit clean-rival margins, paired 2000-draw puzzle bootstrap and both original scientific gates. It checks saved numerical statements and does not independently rerun pretrained Torch forward passes or certify exact floating-point arithmetic.

A positive primary gate establishes that optimized full-state singular directions can causally alter more distributed task predictions than one equal-dose random direction on fresh puzzles. It does not establish improvement in Sudoku accuracy, general mechanistic semantics or cross-checkpoint transfer. A negative result limits this particular dose and horizon, not all source-state causal influence.
