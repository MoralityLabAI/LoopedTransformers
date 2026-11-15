# AF-07: recurrence-resolved locality of latent Horn proof state

AF-07 is a fresh preregistered follow-up to the [AF-06 negative result](../activation_funnel_af06/README.md). AF-06 demonstrated qualified train/development Q-state inference under status warm-up, but long-dependency subgoal interchange lost **untouched-Q** correctness and irrelevant-patch goal retention. This experiment asks whether supervising the *untouched Q token throughout downstream recurrence* mitigates the failure.

The original TypedLoop implementation, AF-05 graph encoding and Horn oracle, AF-06 corpus, and all earlier registrations/results remain frozen. The new code lives exclusively under `experiments/activation_funnel_af07` plus its own CI workflow.

## Mechanism

The model reads initial facts and a public two-subproof Horn DAG. Five-to-ten upstream native-cell calls depending on graph depth infer Q0/Q1. All upstream incidence is then hidden; three downstream calls propagate the carried subgoal codes to goal G. Each of 15 atom and 14 rule tokens carries 1 or 2 float32 coordinates between native calls; a decoded 16-coordinate working state is recreated per token at every call.

A matched donor changes the truth of selected subgoal Q_j while keeping the rest of the graph and other-subgoal leaf values fixed. AF-07 changes **only the actual carried code** of that Q_j atom before executing the recipient's downstream stage. An independent Horn least-fixed-point computation supplies the correct patched Q0,Q1,G targets.

We record `q_steps`, `goal_steps`, and `q_codes` after each downstream call. Let \(z_{t,k}\) be the carried code for Q_k and \(p_{t,k}\) its predicted distribution after downstream step t. If Q_j was patched, the other Q is protected; when an irrelevant Q was patched, Q_j is protected. The extra training loss is:

\[
\mathcal{L}_{local}=0.25\,\frac{1}{T}\sum_t
D_{KL}(p^{clean}_{t,k}\Vert p^{patch}_{t,k})
+0.12\,\frac{1}{T}\sum_t\|z^{patch}_{t,k}-\mathrm{sg}(z^{clean}_{t,k})\|_2^2/d,
\]

where k is the protected Q, \(d\) is its code width, and `sg` stops gradients. The original task/patch cross-entropies and per-step exact Q cross-entropies remain shared by the status, ramp, and locality arms.

## Preregistered comparison

| Quantity | Registered setting |
|---|---|
| Seeds | 17, 29, 43 |
| Code sizes | 29 and 58 floats total (1 or 2/token) |
| Warm-up | 480 status-supervised updates, shared checkpoint per seed/width |
| Continuations | status, ramp, locality; each 160 further updates |
| Every minibatch | 16 depth-2 and 16 depth-3 cases, matched donor RNG |
| Native forward calls | Identical across all arms: two upstream and two downstream trajectories |
| Status control | Patch semantically irrelevant Q with original target |
| Ramp control | 0→32 addressed-Q patches over 160 updates |
| Locality arm | Same addressed patch schedule, plus full-step protected Q KL and code penalty |
| Training data | Original AF-05 672 train +224 dev, fixed manifest |
| Held-out | 16 novel composition, 224 novel physical layout, 128 depth4, 128 depth5 |
| Additional evaluation | 5 downstream loops on depths4/5; tensor-only extension |

Forward-call counts are matched. The locality arm performs additional KL/MSE and backward arithmetic, so **full compute is not matched**. All sources, weights, predictions, trace rows, original patch controls and independent audit receipts are retained. Full method and gates are fixed in [PROTOCOL.json](PROTOCOL.json).

The primary causal-locality gain is measured against the **ramp** arm, which receives the same addressed donor interventions and their exact targets. Status remains a separate comparator. Both depth4 and depth5 must pass the per-seed conditions after the warm-up, ramp, status and locality models all reach at least 80% goal and joint Q exact accuracy on train and dev. A failure in a qualified seed fails the aggregate. Ineligibility is recorded as inconclusive.

## Reproduction

Run from the repository root with Python 3.12, CPU Torch 2.5.1, NumPy and SciPy:

```bash
python -m unittest discover -s experiments/activation_funnel_af07/tests -p 'test_*.py' -v
python -O -m unittest discover -s experiments/activation_funnel_af07/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af07.experiment manifest --out /tmp/af07-manifest.json
python -m experiments.activation_funnel_af07.experiment train --seed 17 --width 2 --out /tmp/af07-s17-w2-train
python -m experiments.activation_funnel_af07.experiment evaluate --seed 17 --width 2 --training /tmp/af07-s17-w2-train --out /tmp/af07-s17-w2-eval
python -m experiments.activation_funnel_af07.verify --seed 17 --width 2 --train /tmp/af07-s17-w2-train --evaluation /tmp/af07-s17-w2-eval --out /tmp/af07-s17-w2-eval/independent_audit.json
```

The independent verifier reconstructs every Horn target and staged Q predicate from saved logits, checks the full paired-case roster, verifies whole-code donor equality, and replays samples from the saved Torch weights. CI runs the complete three-seed/two-width matrix and separately aggregates registered scientific verdicts.

**Scope:** Lean certifies generated closed Boolean equations for the synthetic Horn oracle, not neural proof terms. Extra downstream recurrence steps share the same target semantics and earn no new original-host proof credit.
