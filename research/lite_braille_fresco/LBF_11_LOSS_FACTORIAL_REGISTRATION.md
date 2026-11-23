# LBF-11: Separate outcome supervision from set ranking

**Registered 2026-11-24 before fresh LBF-11 collection, fitting or calibration scoring.** Development experiment following the negative LBF-10 structural replication. The exposed LBF-09 and LBF-10 results motivate this study. They are not independent confirmations.

## Question

The existing two-pass supervised objective combines mean binary cross-entropy on the outcome head and negative log policy probability assigned to a positive tactic set. Switching from a menu-first positive to all shortest native tactics changes both losses simultaneously. LBF-11 separates those changes, including a balanced positive ranking alternative.

For known native shortest-action labels y, let S be the positive set and a the menu-first member. For outcome logits o and policy logits z:

B1 = mean BCE(o, indicator(a)); BS = mean BCE(o, indicator(S)).
R1 = logsumexp(z) - z[a].
RS = logsumexp(z) - logsumexp(z[S]).
RB = logsumexp(z) - mean(z[S]) - log(|S|).

RB equals RS plus KL(uniform(S) || softmax(z restricted to S)). It preserves a penalty for concentration within S, has the same gradient as uniform-positive cross entropy, and equals RS when S contains one action. When every candidate is positive, RS has zero policy-logit gradient, whereas RB may still encourage balanced probability. These identities are analytical, not yet evidence of learned mechanism.

## Frozen diagnostic

Audit the archived LBF-10 run 45269242000, artifact 14865476852. Verify the original teacher, visited-source and replay manifests, behavior checkpoint, and selected checkpoint bytes. Use only the 55 archived multi-positive training visits; no archived calibration outcomes enter analysis. Diagnose seed-107 and seed-109 visited_one and visited_set checkpoints on precisely the same packets, at depth 2. Measure per-term loss, shared-backbone gradient norm and cosine, policy/outcome-head logit gradient norms, positive mass and conditional positive entropy, valid-to-invalid score margins, and the fraction of all-positive menus. Keep zero-norm cosine undefined. Preserve per-frame, root-clustered summaries and frozen parameter fingerprints. This is a retrospective diagnostic, not a causal explanation. Separately perform the same pre-fit diagnostic using the fresh behavior checkpoint on the new train-only visited corpus. A missing archive is recorded separately and does not change the fresh experiment.

## Prospective source and matched training

Freeze new structural diamond suite seed **20261203**, 64 roots per split, structural_alt=true, 48 train-only native teacher roots, first 32 calibration roots. Retain LBF-10's source-exposure thresholds: >=25% multi-positive witnessed visits, >=12 independent multi-positive lineages, >=50 planned multi-positive updates per seed. Independently compile all teacher reference proofs and replay the frozen 360-step seed-107 behavior checkpoint's 48 native train episodes and forced recoveries. Do not select source roots by observed outcomes. Stop with retained evidence if source exposure fails.

Use the LBF-08 deterministic frame schedule, seeds 107 and 109, 720 FP32 AdamW steps, LR .001, one CPU thread, same initial weights per seed, width 32, 57,475 parameters, AB recurrence depth 2, 384-byte hybrid_rule_intact view and clipping at norm 1. All five visited arms get exactly the same packets and indices. Source frames in every arm use the original full-set objective. Source-only draws use native reference frames on the matched root schedule. On visited frames:

| Arm | Outcome target | Policy ranking |
|---|---|---|
| b1_r1 | singleton | singleton |
| bs_r1 | full set | singleton |
| b1_rs | singleton | marginal set |
| bs_rs | full set | marginal set |
| bs_balanced | full set | balanced positive |
| source_only | full set on references | marginal set on references |

Repeat seed-107 b1_r1 on the same worker and require bitwise named-weight and optimizer-draw equality. Prove singleton/full endpoints exactly match the legacy loss and gradients. Verify the balanced loss identity and all-positive gradient edge cases. No treatment changes the optimizer budget or candidate menu. Report actual visitation, multi-positive exposure, gradient statistics, and final weights.

## Native outcomes and interpretation

Score the twelve frozen learners on the same 32 new structural calibration roots, at eight checks and depth two, with native Lean kernel, independent standalone compilation and model-free replay. Compute exact uniform expectation via LBF-10 audited native public menus. No ordinary or OOD test outcomes, no query-root updates. Same seed initialization, roster hashes, source/replay hashes and full proof receipts are hard gates.

Let Y_BR be binary whole-proof closure for the factorial arms B,R in {1,S}, averaged over the two seeds *within each root*. Primary contrasts: outcome main effect ((Y_S1-Y_11)+(Y_SS-Y_1S))/2; ranking main effect ((Y_1S-Y_11)+(Y_SS-Y_S1))/2. Adverse development floor -0.05 for each. Interaction Y_SS-Y_S1-Y_1S+Y_11. Balanced rescue Y_balanced-Y_SS with >=+0.05 development threshold. Also report bs_rs minus b1_r1 (replication), all arms versus reference-only, and per-cell exact uniform advantage and numerical competence (>=40% closure and >=10pp above uniform). For every contrast retain 2,000 descriptive root bootstrap draws, seed 20261123; report root counts and intervals. These are exploratory multiple contrasts, not multiplicity-adjusted confirmatory findings. Neither main effect alone establishes a molecular mechanism. Independent FSI-01 power and matched shams remain necessary for promotion.

**Evidence:** complete source and visit audit, per-frame diagnostic records, twelve checkpoints, exact repeat, matched root-paired certified outcomes, rational uniform control and frozen fresco. Keep sealed cohorts closed. CPU-only.
