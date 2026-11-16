# AF-28: prospective STOP calibration with frozen Direct Horn-rule ranking

AF-27 found that VI-2's native STOP gate failed to improve correct immediate termination on two new grammar families in the same eight original frozen checkpoints. AF-28 trains a **new seven-parameter logistic STOP gate** on **separate train grammars**, selects its decision threshold exclusively on **development grammars**, and tests on **new held-out rule structures**. The original 6,036-parameter recurrent model and its Direct rule-ranking head remain exactly frozen.

**Prospective protocol:** [PROTOCOL.json](PROTOCOL.json), initial registration commit `45c1544`, with dated pre-feature amendments for negative-root critical-fact guarding and the exact development threshold-selection procedure. [Source-only structural CI](../../.github/workflows/activation-funnel-af28-preflight.yml) independently checks every split, positive-Horn derivation and proof length before training. Structural manifest SHA-256: `34c30216b8407fcf9339e7c909512e5011e9375102ef0451941868574b39c36d`.

## Data and frozen checkpoint isolation

| Source split | New grammar families | Cases | Provable | Unprovable |
|---|---|---:|---:|---:|
| Train | `train_shared_gate`, `train_dual_or` | 192 | 80 | 112 |
| Dev | `dev_crossed_and`, `dev_short_or` | 192 | 80 | 112 |
| Held depth 5 | `held_forked_and`, `held_or_gate` | 192 | 80 | 112 |
| Held depth 6 | Same held structural families at distinct six-step topology | 192 | 80 | 112 |

All cases use **four initial facts**, two alternative goal-concluding rules, the original 15-atom/14-rule public encoding, and source-pinned physical atom/rule permutations. Negative cases replace required root atom 2 with a derived decoy, and both goal routes are guarded by the critical fact so the independent bitset BFS certifies unreachability. Every source logical-incidence signature is distinct across train, dev and held-out splits. Held-out physical layouts 630–643 are never used for fit or model selection.

The same original AF-23 frozen final checkpoint is used for one seed throughout. GitHub Actions downloads each original model ZIP by immutable artifact ID and verifies the original source/member SHA-256 and Torch/NumPy exact state-dict parity. The original model receives **zero optimizer updates** in AF-28.

## What is trained

The newly fitted component is a logistic STOP gate using six public/frozen-model features: Direct STOP-minus-best-legal margin, root reach logit, distance logit, VI-2 STOP-minus-best-legal margin, number of legal rules, and number of already-known facts. Inputs are clipped and standardized using **train only**. The gate has six slopes and one bias; its 600 full-batch updates minimize class-balanced BCE plus 0.01 L2 slope penalty.

The matched **THR comparator** fits a one-feature logistic on the Direct STOP-minus-best-legal margin using the same training set, update count and regularization. Both fits select their decision threshold independently on the same fixed `0.05..0.95` grid using **dev roots only**. Selection maximizes correct negative STOP, constrained to continue on at least 72/80 provable dev roots; ties select the largest threshold.

Training features are collected from each train case's root and up to two legal first-rule successor states chosen by a source-fixed physical slot rule, without selecting them from model outcomes or oracle proof paths. Development uses only the 192 root states. Labels come from train/dev host BFS and never enter the model's inference feature vector. Both calibrator weights, source training/development feature rows, and chosen thresholds are saved and independently refitted before held-out scoring.

## Held-out arms and scientific gate

| Arm | STOP decision | Legal-rule rank |
|---|---|---|
| **DD** | Original frozen Direct STOP gate | **Original frozen Direct** |
| VD | Original frozen VI-2 STOP gate | Frozen Direct |
| **CAL, registered target** | Train-fitted six-feature logistic, dev-frozen threshold | Frozen Direct |
| **THR, equal-budget control** | Train-fitted one-feature Direct-margin logistic, dev-frozen threshold | Frozen Direct |
| ND | Continue when any legal rule remains | Frozen Direct |
| CLOSURE | Public additive Horn closure, no model |
| BFS16 / BFS64 | Original public bounded BFS algorithms |
| SHORTEST | Independent exact host shortest-proof oracle; evaluation upper control only |

Every neural arm performs a **fresh closed-loop rollout** with newly computed model scores after each selected legal Horn rule; calibrated STOP is applied at every visited state. A reached goal terminates through the host. Every policy uses the same shortest-proof length + one legal firing budget on provable goals and two decisions on unprovable goals.

**Preregistered primary gate:** CAL must correctly immediately STOP on **at least seven more negative cases out of 112** than DD and complete **at least DD's positive proofs minus four out of 80**, at **both held depths in the same checkpoint seed**, with zero false accepted physical proofs. **At least six of eight seeds** must clear the entire gate for `CALIBRATED_STOP_TRANSFER_PASS`. Complete source-checked failure is `CALIBRATED_STOP_TRANSFER_FAIL`; missing or corrupt source evidence yields `INCONCLUSIVE`.

THR, VD and all public graph-aware controls are reported for scientific interpretation. If THR matches or exceeds CAL, a claimed benefit from *multivariate calibration* is unsupported, even if CAL meets its original DD-relative gate. Public closure can solve much of this tiny Horn task, limiting any general neural proof-search inference.

## Evidence and independent audits

- Immutable original model source and SHA-256 model ZIPs, original Torch/NumPy weight parity, full per-seed calibration and official source manifest.
- Train/dev-only fitted weights, train/dev feature rows, deterministic repeat of all logistic optimizer updates and dev threshold selection before accepting each heldout result.
- Every heldout physical Horn rule firing and STOP rechecked from model-visible tensor incidence with a separate BFS truth oracle, independent numeric min-plus score reconstruction, signed calibrated probability/threshold replay, sampled frozen Torch forward scores and feature replays.
- Per-seed proof completions, negative correct STOP, premature STOP, paired success/loss transitions, initial-root Brier score, log loss and ten-bin ECE for the two probabilistic gates.
- Actual neural node-forward counts and symbolic state expansion/rule examination counts; no claims of matched FLOPs or runtime from matched state/action counts.

All original AF-22 through AF-27 verdicts remain unchanged. Keep this branch draft and unmerged for independent review.

### Reproduce

Source-only:

```bash
python -m unittest discover -s experiments/activation_funnel_af28/tests -v
python -m experiments.activation_funnel_af28.gym > af28_structural_manifest.json
```

With pinned original CPU runtime and an original AF-23 source ZIP named `original/af23-branch-seed-17.zip`:

```bash
python -m experiments.activation_funnel_af28.calibrate \
  --seed 17 --original-zip-dir original --out /tmp/af28-calibration-17

python -m experiments.activation_funnel_af28.evaluate \
  --seed 17 --original-zip-dir original \
  --calibrator-dir /tmp/af28-calibration-17 --out /tmp/af28-score-17

python -m experiments.activation_funnel_af28.audit \
  --seed 17 --original-zip-dir original \
  --calibrator-dir /tmp/af28-calibration-17 --output-dir /tmp/af28-score-17
```
