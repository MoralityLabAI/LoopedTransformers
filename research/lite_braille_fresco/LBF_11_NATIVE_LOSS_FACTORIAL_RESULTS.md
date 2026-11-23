# LBF-11: Outcome/ranking factorial and frozen gradient audit

**Date:** 2026-11-24. **Disposition:** ALL_REGISTERED_LOSS_EFFECT_SCREENS_MISSED. One individual numerical competence pass. Development only.

[Registration](LBF_11_LOSS_FACTORIAL_REGISTRATION.md) · [Native and archived diagnostic run 45276730811](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45276730811) · Tested commit bb30808a3e461fd5d4cffb6fb0615637685bb39e.

- [Native evidence bundle](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45276730811/artifacts/14868743447), SHA-256 22190e5315333ebec3dc2dcccc60a7dbd54db576730493fd7d55cbafcdc113dd.
- [Archived train-only diagnostic](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45276730811/artifacts/14868051383), SHA-256 22eee078c46fe240127015d966a1cdb95845ffd8d696c0a42a1491868a17ca92. Both expire 2026-12-07.

## Source and treatment custody

Fresh structural suite seed 20261203. Real Lean 4.19 certified 48 train-only reference proofs; 259 reference frames, 178 with multiple shortest actions. Frozen seed-107 360-step behavior collection produced 97 witnessed visited frames, 71 multi-optimal (73.20%) across 27 lineages. All 48 train episodes and recoveries passed independent replay. Visited arms shared exactly 196/181 visits and 135/131 multi-positive updates for seeds 107/109. Every model had 57,475 parameters, 720 AdamW steps, two recurrent passes, one CPU thread, and matched seed-specific initialization and draw schedules. The seed-107 B1/R1 repeat passed exact named-tensor and optimizer-log equality. All 19 engineering tests passed.

On reference frames all models use the original full-set BCE and marginal ranking. On the same visited frames the factorial changes outcome labels (B1 or BS) and ranking (R1 or RS). The balanced condition uses full BCE with marginal ranking plus conditional-positive KL to uniform. Endpoint losses and gradients equal the original objective on the known native labels.

## Frozen gradient audit

The prior LBF-10 archive was verified by source, replay and checkpoint hashes. Each of four frozen checkpoints was measured on the same 55 multi-positive train visits, across 32 roots. No optimizer updates or calibration outcomes were used. There were 32 all-positive menus among these 55 frames. The fresh behavior checkpoint's independent pre-fit audit measured 71 multi-positive visits across 27 roots, including 38 all-positive menus.

Root-weighted means:

| Frozen model | Positive mass | Normalized conditional-positive entropy | Shared grad norm R1 | Shared grad norm RS | Shared grad norm RB |
|---|---:|---:|---:|---:|---:|
| LBF10 107 one | .7384 | .9793 | 3.0245 | .7232 | 1.1610 |
| LBF10 107 set | .7876 | .8695 | 4.9593 | 1.3561 | 1.9575 |
| LBF10 109 one | .7345 | .9881 | 3.0203 | .7508 | .9126 |
| LBF10 109 set | .8377 | .7059 | 9.2910 | 2.2715 | 4.8230 |
| LBF11 fresh behavior | .7401 | .9956 | 1.5564 | .3944 | .4336 |

Marginal set ranking has lower measured gradient magnitude than singleton ranking in these frozen diagnostics. The archived set-trained models assign more aggregate mass to optimal tactics and concentrate it more strongly on individual optimal tactics. All-positive menus make the marginal rank gradient exactly zero. These observations establish properties of the objective and measured training-state gradients, not the causal reason for poorer held-out proof completion.

## Native calibration

All twelve checkpoints were scored on the same 32 fresh development roots, eight tactic checks and depth two, with native Lean kernel closure, standalone certificates and independent model-free replay. Exact uniform expectation = 39527717/120960000 = **32.678337%**, audited across 1,872 representative goals and 3,054 public actions. Public auto8 = 6/32; reference budget reachable = 27/32.

| Seed | Reference only | B1 R1 | BS R1 | B1 RS | BS RS | BS balanced |
|---|---:|---:|---:|---:|---:|---:|
| 107 | 8/32 | 14/32 | 9/32 | 12/32 | 11/32 | 10/32 |
| 109 | 13/32 | 10/32 | 12/32 | 11/32 | 10/32 | 12/32 |

Only seed-107 B1/R1 (43.75%, +11.07166 pp above uniform) meets the descriptive >=40% / +10pp competence threshold. Its bootstrap interval versus uniform crosses zero. No scientific promotion.

| Preregistered root-paired contrast | Mean (pp) | Descriptive 95% root bootstrap (pp) | Wins/losses/ties |
|---|---:|---:|---:|
| Outcome main | -3.90625 | [-15.625,+7.8125] | 8/11/13 |
| Ranking main | -0.78125 | [-10.9375,+9.375] | 9/8/15 |
| Interaction | +1.5625 | [-20.3125,+25] | 10/10/12 |
| Balanced rescue | +1.5625 | [-14.0625,+17.1875] | 8/9/15 |
| Full minus single | -4.6875 | [-20.3125,+10.9375] | 6/10/16 |

All registered adverse-main-effect (-5pp) and balanced-rescue (+5pp) screens missed. The factorial gives no convincing isolation of the LBF-10 adverse effect. The intervals are descriptive, with seeds averaged within 32 distinct theorem roots, no multiplicity adjustment. The empirical result does not justify further tuning of the balanced loss on this exposed cohort.

## Research decision

The public Horn graph is explicitly available, and the independently merged [LSA-02J study](../../reports/lsa02j_branch_utility_cpu_20261124.md) obtained 32/32 with a deterministic public Horn solver under its 16-work-unit allowance. LBF's eight-check model comparison uses a different budget. A stronger learned-reasoning claim requires a harder proof grammar, a matched symbolic baseline, and a task where public graph closure alone is insufficient. Continue toward equality and rewrite decisions under real Lean, with exact public search and verifier budgets. The historical Horn studies remain useful as engineering and diagnostic infrastructure.

No sealed ordinary/OOD scoring, architecture promotion or claim of general reasoning competence.
