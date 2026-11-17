# AF-33: position-balanced learned STOP on new Horn goal motifs

**Status: prospective, preregistered before new graph generation, fitting and heldout scoring.** The first frozen scientific registration is \`14893fb6cb8ecffea44c345cbf0547b48e798ba3\`, [PROTOCOL.json](PROTOCOL.json). The parent AF-32 source-verified results remain **STRICT_RULE_SELECTION_PASS (8/8)** for learned legal-action selection against fixed equal-certificate controllers, and **LEARNED_STOP_STRICT_FAIL (0/8)** for certificate-free termination.

AF-33 directly tests whether balancing the STOP classifier's TRAIN/DEV exposure to middle Horn premise-swap locations 1/2/3 corrects the position-specific false-continuation transfer in AF-32. The exact original eight-feature legal-action ranker, ten-feature STOP summary architecture, 500/600-step optimization budgets and eight original initialization seeds remain registered, and no heldout tuning is permitted.

## Source cohorts

Four source splits, each **96 reachable/unreachable matched pairs** (192 cases), with 48 pairs per logical motif family and exactly **32 pairs at each premise-swap position 1, 2, 3** in TRAIN, DEV and both HELD depths. Every positive shortest proof is exactly 5 or 6 legal firings as registered; each negative requires inaccessible supported atom 10. The paired positive and negative graph members share all initial known facts and marked role locations, physical atom/rule permutations, entire directed per-node degrees, root legal actions and exact seven-anchor physical-ID-preserving directed induced radius-two incidence signatures. Only two unary Horn premises change.

Train and DEV share the two-node supported inaccessible cycle structure, with their own distinct unpermuted logical motifs. HELD depth 5 and 6 balance **48 two-node / 48 three-node cycles** each, with *new goal conclusion topology*: the positive goal-concluding rule has a second, root-known premise on heldout graphs, versus the unary positive goal conclusion in TRAIN/DEV. The two independently root-enabled decoys leading to 7 and 12 also receive new heldout premise pairs. All source topology signatures are disjoint, and every encoded physical graph SHA is disjoint across splits and historical AF29–AF32.

This tests a specific goal-rule-arity and root-diversion distribution shift in a small set of synthetic Horn grammar motifs. Source-case/seed repetitions are not independent grammar families.

## Frozen STOP comparison

- **BAL_STOP**: the same ten public feature summaries inherited from AF-32, trained on 576 TRAIN feature rows, with all three premise-swap positions balanced at 192 rows each. It chooses a fixed threshold from 192 separate DEV roots, with 96 reachable and 96 unreachable, requiring at least 80/96 reachable continuation.
- **J12_STOP**: a prospectively frozen position-exposure ablation, using 384 TRAIN feature rows from positions 1 and 2, and 128 DEV roots (64 reachable and 64 unreachable) from those positions. Identical ten features, 600-step optimizer, regularizer and DEV threshold grid; it requires at least 54/64 reachable DEV continuation. The smaller exposure/sample count is explicitly reported and is part of this control's scope.
- Both use the **identical trained eight-feature legal-action ranker per seed**. Their learned-only inference receives the full public graph/current physical proven bits and ranker feature summaries, never exact closure/BFS, true goal labels, pair metadata, logical atom numbers or scorer-only proof budgets.
- **RANK_CERT** remains a separate strict-budget proof safeguard. Its negative STOP is determined by an exact once-per-root **public symbolic Horn closure certificate**, equally available to PATH/ANCESTRY/UNTRAINED/FIXED/RANDOM controls, and fully charged in source rule examinations. Public closure/BFS16/BFS64 remain independent complete symbolic baselines.
- Original fixed **6,036-weight AF-23 checkpoints**, all eight original SHA-pinned source archives, are evaluated unchanged as Native Direct, never-STOP Direct and cert+Direct legal ranking on the identical AF-33 heldout graphs with genuine independent Torch-forward checks.

## Science endpoints

The **prospective primary POSITION_BALANCED_STOP_TRANSFER_PASS** requires, in at least six of eight original fixed seeds at both HELD depths under exact-shortest *strict* proof budget, BAL_STOP to finish at least 60/96 reachable positive proofs, correctly initially STOP at least 60/96 unreachable roots, make at most 10/96 incorrect positive immediate STOPs, and obtain at least **10/96 more correct negative STOPs** than J12_STOP. Every accepted Horn proof must be sound. The **independent BAL_STOP_ABSOLUTE_PASS** drops the comparative improvement criterion, retaining the absolute proof/stopping floors. The **STRICT_RANKER_SAFEGUARD_PASS** independently checks the earlier 60/96 complete positive proofs and +10/96 positive-proof advantage over the strongest equally certified control, together with >=80/96 certified negative STOP and zero false proofs.

All three gates require six of eight seeds passing simultaneously at both held depths. Historical \`d+1\` slack budget is an independently scored sensitivity cohort and cannot rescue a failed strict science gate.

The source scorer also predeclares root paired STOP-score AUC, Brier, log loss, ten-bin ECE, by-swap-position and by-cycle breakdown, DEV threshold and absolute positive/negative decisions. Every statistic comes from source-verified heldout traces **after** the TRAIN fit and DEV threshold are frozen. Neither paired ranking nor probability calibration substitutes for the registered absolute STOP/strict proof thresholds.

## CPU reproduction

Source-only Python 3.12 checks, no Torch required:

\`\`\`bash
python -m unittest discover -s experiments/activation_funnel_af33/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af33.gym > af33_source_manifest.json
\`\`\`

One source-fitted full policy and independent bitset audit:

\`\`\`bash
python -m experiments.activation_funnel_af33.evaluate --seed 17 --out out/af33-seed-17
python -m experiments.activation_funnel_af33.audit --seed 17 --out out/af33-seed-17
\`\`\`

The original immutable AF-23 Direct control requires the exact original original-seed ZIPs under \`originals/\` plus CPU Torch 2.5.1:

\`\`\`bash
python -m experiments.activation_funnel_af33.direct \
  --seed 17 --original-zip-dir originals --out out/af33-direct-seed-17
python -m experiments.activation_funnel_af33.direct_audit \
  --seed 17 --original-zip-dir originals --out out/af33-direct-seed-17
\`\`\`

After all eight original scored source-fit and Direct artifact directories are available, run:

\`\`\`bash
python -m experiments.activation_funnel_af33.aggregate \
  --root receipts --out af33_results.json
\`\`\`

The registered [source-locked workflow](../../.github/workflows/activation-funnel-af33.yml) pins the graph/protocol SHA-256 and implementation source, then independently scores/audits all eight TRAIN-only ranker/BAL/J12 models, the eight original frozen Direct checkpoint families and all strict/slack trajectories. Any engineering repair and any registered failure remain visible. AF-22 through AF-32 outcomes are unchanged.
