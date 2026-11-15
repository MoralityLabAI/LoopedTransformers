# AF-17: Bellman rule-cost policy and explicit STOP calibration

AF-17 is stacked on AF-16 and retains the original host soundness gate. Both
source and model receive exactly the same physical graph facts and proved-atom
bits as in AF-16. The shortest-proof and action-cost values are labels only.

For any provable unsolved Horn proof state \(S\),

\[
d(S)=1+\min_{a\in A(S)}d(S_a),\quad Q(S,a)=1+d(S_a).
\]

STOP has a cost of 10 when the goal is provable and unfinished. The same STOP
has cost 0 when the goal is already proven or the state is unreachable.
Unreachable rule successors have capped cost 10. We supervise raw action
logits against \((10-Q)/3\) and separately fit \(d(S)/3\). Exact source
DP supplies the training labels; the independent evaluator reconstructs
shortest paths by BFS from public **physical** incidence tensors.

## Arms

- **imitation:** 240 updates of AF-16 shortest-optimal action CE.
- **stop:** 240 matched updates adding a reachability and STOP-ranking loss.
- **bellman:** 240 matched updates adding exact action and state cost regression
  to the complete STOP objective.
- **imitation_extra:** 360 updates, 50% more optimizer/forward calls than Bellman.

All four arms have exactly 5,268 parameters, use the same initial tensors and
first 240 samples per seed, and choose an action using the identical raw-logit
readout and host mask. Loss arithmetic/backpropagation FLOPs differ.

## Dataset

AF-16's 928 training and 352 development cases are reused unchanged. A
train-only deterministic detour-state augmentation exposes actions outside the
single lexicographic shortest reference proof. Heldout is **416** new
physical input cases on layouts 52–56, with **192** depth-4 and **192** depth-5
cases containing unseen `serial_shortcut`, `dual_fallback` and
`contingent_goal` logical templates. A separately registered manifest verifies
all input feature hashes are pairwise disjoint and absent from AF-16 scored
inputs. The full test is strictly frozen before training.

## Reproduction

Run from repository root with Python 3.12/Torch 2.5.1+cpu, NumPy and two CPU
threads; local Python 3.13/Torch 2.10 experiments must be labeled separately.

```bash
python -m unittest discover -s experiments/activation_funnel_af17/tests -p 'test_*.py' -v
python -O -m unittest discover -s experiments/activation_funnel_af17/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af17.experiment train --seed 17 --out artifacts/af17/s17/training
python -m experiments.activation_funnel_af17.experiment evaluate --seed 17 --training artifacts/af17/s17/training --out artifacts/af17/s17/evaluation
python -m experiments.activation_funnel_af17.verify --seed 17 --training artifacts/af17/s17/training --evaluation artifacts/af17/s17/evaluation --out artifacts/af17/s17/evaluation/independent_audit.json
python -m experiments.activation_funnel_af17.aggregate --root artifacts/af17 --out artifacts/af17/results.json
```

The aggregate requires all four seeds and distinguishes source/verification
success from the registered scientific criteria. Every evaluation retains raw
model logits and actual Horn proof edges. No model output is ever accepted as a
new axiom by the host. The Lean file proves an abstract ranking-gap inequality;
its availability is separate from neural proof-search results.
