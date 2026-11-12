# RMP-GG-08: Depth-eight competency recovery with a visit-matched training control

**Registered November 12, 2026 before GG-08 continuation training or fresh-panel scoring.** Source \`main\` at \`97b3eba35a5e0c037fc4f5f65b962e68b07bbdbb\`. This follows the [GG-07 scientific-negative result](../reports/checkpoint_rmp_gg07_protected_tail_20261112.md), in which the original GG-01 four-cue-trained recurrent classifier missed the preregistered 90% capability threshold on eight-cue exact-oracle boundary/rare/late-reversal cases. The published GG-07 panels and their outcomes are already known.

## Question

Does additional exposure to longer recurrent evidence streams improve eight-cue competency, and does a more competent model exhibit **stable signs of physically measured wrong-label SGD effects** under the frozen GG-07 protected-tail scorer?

A paired follow-up distinguishes three states of the same native \`EvaluationBeliefLooper\`, with identical source architecture, initial model seed, and 400-step four-cue base training:

1. **Original:** trained GG-01 checkpoint at step 400, with no continuation.
2. **Four-only visit-matched control:** 300 extra AdamW updates on four-cue task examples, batch **192**.
3. **Mixed-depth continuation:** 300 extra AdamW updates, batch **128**, cycling depths **4,6,8** exactly 100 times each.

Both continuations use the **same initial base weights** (independent cloned model copies), freshly initialized AdamW with learning rate **0.001**, betas (0.9, 0.999), weight decay **0.001**, terminal CE plus **0.2** mean per-visit prefix CE, and global gradient clipping at **1.0**. Both use the fixed final update as endpoint, without validation-based selection or any early stopping. The number of optimizer updates and total training *sample-visits* match exactly:

\[
300\cdot192\cdot4
=100\cdot128\cdot(4+6+8)
=\boxed{230{,}400}.
\]

This matches total unrolled sample-visits, a compute proxy; optimizer operations and attention overhead are not claimed to have exactly identical FLOPs. The four-only arm processes more training examples, so any difference remains a matched-visit, **not iso-sample**, comparison.

## Frozen training data and sampling

- Source neural seeds: **11 and 17**, trained through the original unchanged GG-01 \`train_cell(seed,steps=400,batch=128)\`.
- Frozen pools (separate from all gold audit seeds): \`make_panel(8192,depth,78000+10*seed+depth)\` for depths 4,6,8. Exactly one pool per depth and seed; four-only and mixed both sample depth four from **the same pool**.
- Batch-draw generator per continuation: \`torch.Generator().manual_seed(81000+100*seed+armcode)\`, where armcode 0 for four-only and 1 for mixed. Each optimizer update samples with replacement from its depth pool, and every minibatch has its own exact Bayes prefix labels.
- Torch CPU, one thread, no external model, no checkpoint selection by scoring outcomes. Keep the training loss at steps 1, 75, 150, 225, 300, endpoint weights, source hashes and optimizer budget. Re-run training deterministically from recorded configuration.
- Record every arm's sample-visit count, class label balance, and unique sampled item count per depth (diagnostics only). The training streams are frozen before model evaluation.

## Primary independent gold panels

Apply **the unchanged GG-07 protected-tail score, lambda and TV ambiguity** to new independently sampled gold panels.

At depths 4,6,8, each of four scoring replicates contains natural 64, exact-oracle boundary 32, exact-mixture-rare 32, and late-evidence-reversal 32 examples. Challenge **membership criteria** remain precisely those of \`challenge_pools\` in GG-07: lowest quartile of absolute exact Bayes log margin; lowest quartile of exact mixture sequence likelihood; and half-prefix versus full-sequence decision reversal. The new panel seeds, fixed before GG-08 training, are:
- Natural \`make_panel(64,depth,77000+100*depth+r)\`;
- Challenge sampling \`random.Random(79000+1000*depth+10*r+offset)\`, with offset 1 for boundary, 2 for rare, 3 for late reversal.

The panel is fully fixed for all three arms in a seed, and is also identical across model seeds. All scoring examples and task labels are drawn from the exact registered Bayes grammar. **New sample combinations** are independent of the known GG-07 panels. The task family itself remains the same finite Bayes decision problem, so this is not a new task-family generalization claim.

## Frozen gold capability gate

For each model seed and arm, compute natural/boundary/rare/late exact-match action accuracy separately at each of depths 4/6/8 on the **new GG-08 scoring panels**. The original GG-07 capability requirement is unchanged: every one of **12** depth×cohort accuracy cells must be at least **90%**.

A **competency-recovery pass** requires *both* mixed-depth models to clear that original 12-cell requirement. Report the paired per-cell improvement against the 400-step original and the four-only continuation, including any regressions at depth four.

If either mixed model fails, preserve the result as \`CAPABILITY_UNRESOLVED\` and retain all robust sign measurements. The continuation has no extra retry budget and no test-based hyperparameter change.

## Registered protected-tail physical-update audit

Each endpoint arm is frozen before the report-gradient assay. Source report queries remain the previously inspected **48** GG-04B heldout four-cue inputs and exact Bayesian labels; their membership is unchanged. For each query/arm, compute two real, independent PyTorch SGD steps on cloned task-model weights at learning rate **0.05**, global full-parameter report-gradient clip **1.0**, starting from identical frozen endpoint weights. Score all 1,920 per-example gold inputs at depths 4/6/8 after truth and flipped report updates. The baseline and both continuation arms therefore require \(3\times2\times48\times2=\boxed{576}\) native physical SGD steps in total, and \(576\times1920=1{,}105{,}920\) per-example post-update gold CE measurements.

The physical harmedness is \(D=\ell(\theta_X)-\ell(\theta_C)\). The protected objective remains exactly GG-07's:
\[
B_{d,r}(\lambda)=(1-\lambda)\bar D_{N}
+\frac\lambda3(\bar D_{B}+\bar D_R+\bar D_L).
\]
Use the original full sensitivity grid \(\lambda\in\{0,1/4,1/2,3/4,1\}\), within-cohort empirical-support TV radii \(\rho\in\{0,1/64\}\), and constrained gold depth weights \(\alpha_d\ge1/6,\sum\alpha_d=1\). The primary cell is \((\lambda,\rho)=(1/2,1/64)\). Compute the same sharp three-depth product-TV minimum and maximum as GG-07, compare each fixed source query across the three neural arms, and retain every uncertainty bound and raw physical loss.

A query has a **robust harm sign** if all four replicate lower bounds are strictly positive beyond \(10^{-7}\); a **robust benefit sign** if all four upper bounds are strictly negative; otherwise it is unresolved. All definitions and numerical thresholds carry over unchanged from GG-07.

The **robust-harm-recovery gate** requires (a) both mixed-depth models to be competency-eligible; (b) at least **24 of 48** robust harmful-or-beneficial query signs under the primary protected objective for **each** mixed model; (c) a gain of at least **8 robustly signed queries** compared with that seed's matched-visit four-only control. Keep the two separate gate components in the receipt.

A **secondary loss-sign-stability** event requires the mixed model to have at least 36 of 48 queries with the same nonzero nominal protected-score sign across all four replicated scoring panels and no more than 8 queries with a natural-only-to-protected sign switch. This readout does not affect the primary gate.

## Native controls and statistical limits

- Baseline stage/continuation clonability: source weight SHA matches and both continuation copies equal the base before their first update; record final exact model-weight hashes and prove they differ from base. Report any base hash contamination.
- The two 300-step continuations must have exactly equal 230,400 sample-visits. Independent finite mock controls check pooled label generation and repeated sampling behavior.
- Compare GG-08's actual one-step SGD result with an independently implemented ordinary PyTorch physical-update routine on the same model, query and gold panel.
- Reuse GG-07's sharp finite LP/TV and constrained-depth-mixture functions unchanged. Test numerator/denominator and weight-witness identities.
- Each depth/cohort scoring sample is drawn independently of the model's observed errors, source report-gradient harm and previously known GG-07 gold samples. New-sample empirical evidence remains conditional on the same 9×2^depth finite grammar and **two model seeds**.
- Neither arm knows a hidden SGD assignment or trains a separate goal-guarding reporting controller. This experiment tests **model competence and actual report-update harm on a protected gold scoring distribution**, without demonstrating spontaneous training awareness or future-gradient self-modeling.
