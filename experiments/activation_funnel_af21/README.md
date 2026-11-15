# AF-21: actual greedy STOP decisions, backed by exact Horn action labels

AF-20 trained reachability/STOP classifiers using a **mass gap** between STOP
and the `logsumexp` of legal rule logits. The host executes a greedy `argmax`, so
an accurate relative ordering on pairs can coexist with premature STOP.
AF-21 adds the exact **STOP versus the strongest enabled rule** decision margin.

The 6,036-parameter model and host rule gate remain unchanged. Every source
state receives an exact shortest-proof action set and remaining Horn proof cost.
Four registered training arms compare existing BCE, direct greedy STOP loss,
STOP plus ranking of optimal rules, and 50% more training for the last arm.
All arms get the same full source forwards and first 240 sampled updates. Extra
loss/backward FLOPs differ as documented in PROTOCOL.json. No target label is a
neural feature, and no proof transition is accepted from the student directly.

The **eight seeds** are `(17,29,43,61,73,89,103,131)`. Train/dev cases and
training counterfactuals are byte-identical to AF-20. Prospective heldout is
416 cases on layouts 72–76 and three new cross-branch/shortcut compositions.
The independent verifier recomputes source labels from physical Horn incidence
and a separate breadth-first search and replays checkpoint predictions.

`GreedyStop.lean` proves that if every selected nonterminal action decreases
exact remaining proof cost by one, the policy reaches cost zero within the
initial cost. This mathematical theorem assumes the greedy action meets the
score separation and Bellman hypotheses, which trained neural weights must
satisfy empirically.

## Local preflight and run

```bash
python -m unittest discover -s experiments/activation_funnel_af21/tests -p 'test_*.py' -v
python -O -m unittest discover -s experiments/activation_funnel_af21/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af21.experiment manifest --out /tmp/af21-manifest.json
cmp /tmp/af21-manifest.json experiments/activation_funnel_af21/fresh_manifest.json
for seed in 17 29 43 61 73 89 103 131; do
  python -m experiments.activation_funnel_af21.experiment train --seed "$seed" --out "artifacts/af21/s${seed}/training"
  python -m experiments.activation_funnel_af21.experiment evaluate --seed "$seed" --training "artifacts/af21/s${seed}/training" --out "artifacts/af21/s${seed}/evaluation"
  python -m experiments.activation_funnel_af21.verify --seed "$seed" --training "artifacts/af21/s${seed}/training" --evaluation "artifacts/af21/s${seed}/evaluation" --out "artifacts/af21/s${seed}/independent_audit.json"
done
python -m experiments.activation_funnel_af21.aggregate --root artifacts/af21 --out artifacts/af21/results.json
```

Local Py3.13/Torch2.10 results are separate from pinned hosted Py3.12/Torch2.5.1.
The host-enforced legality result should never be described as the network
having independently proved Lean theorems. Keep AF-20 negative outcomes intact.
