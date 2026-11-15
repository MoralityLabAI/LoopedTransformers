# AF-20: Counterfactual STOP calibration with connected distractors

This separate follow-up to AF-19 uses the source-pinned TypedLoop and identical 6,036-parameter **ancestor** readout for every model. It compares paired fact-change training against a forward-call-matched reachability/STOP classification control. Host Horn gating remains the only authority for accepted proof steps. Scientific failure does not fail engineering CI.

## Exact pair contract

A pair changes one initial fact, preserving the entire logical/physical Horn rule program. Both examples must have at least one enabled rule, creating a nontrivial STOP decision. The changed fact must have an undirected rule-incidence path to the final goal. Two independent source labels exist:

- `flip`: the exact reachability of the goal switches between members. Training encourages reachable examples to have higher reach logits and lower STOP-vs-legal-rule logit gap.
- `connected_invariant`: both members remain reachable with **equal minimum rule-firing proof cost**, despite the fact being graph-connected. Training asks the reach and STOP probabilities to agree. This is a *conditional outcome invariance* test; arbitrary connected perturbations need not be irrelevant.

The primary loss is AF-19's original shortest-action CE, STOP/reach BCE and Bellman cost Huber. All arms also classify both members of the same pair using reachability and STOP BCE, under identical sampled pair rosters. The paired arm adds two logistic ranking margins on truth-changing pairs and squared-probability consistency on connected-invariant pairs. The BCE arm omits these additional relative terms while retaining the same source forwards, parameters and optimizer steps.

## Study and boundaries

- Original AF-17/19 train and dev cases are unchanged: 928 train, 352 dev.
- Fresh physical layouts 67-71 give 416 held-out cases at depths 3/4/5; case feature hashes are disjoint from AF-16 to AF-19 held-outs.
- Seed 17/29/43/61, two and four native recurrent calls, 240 fixed AdamW updates, matched `bce` and `pair` arms at each loop count.
- Developer-set classification margins, loss coefficients and scientific thresholds are fixed in `PROTOCOL.json` before primary training.
- Eval records all raw rule/STOP logits in host-checked episodes; paired evidence includes raw reach/STOP logits, shortest cost, graph distance and source identity.
- Independent checker reconstructs rules from public physical incidence, recomputes Horn fixed-point reachability and BFS proof costs, validates pair classification and all proof traces, checks checkpoint/NumPy weights and replays model forwards.
- The resulting formal Lean identities apply to source Boolean labels. They do not certify that a neural predictor has learned a logical theorem.

## Reproduce

```bash
python -m unittest discover -s experiments/activation_funnel_af20/tests -p 'test_*.py' -v
python -O -m unittest discover -s experiments/activation_funnel_af20/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af20.experiment manifest --out /tmp/af20_manifest.json
cmp /tmp/af20_manifest.json experiments/activation_funnel_af20/fresh_manifest.json
for seed in 17 29 43 61; do
  python -m experiments.activation_funnel_af20.experiment train --seed "$seed" --out "artifacts/af20/s${seed}/training"
  python -m experiments.activation_funnel_af20.experiment evaluate --seed "$seed" --training "artifacts/af20/s${seed}/training" --out "artifacts/af20/s${seed}/evaluation"
  python -m experiments.activation_funnel_af20.verify --seed "$seed" --training "artifacts/af20/s${seed}/training" --evaluation "artifacts/af20/s${seed}/evaluation" --out "artifacts/af20/s${seed}/independent_audit.json"
done
python -m experiments.activation_funnel_af20.aggregate --root artifacts/af20 --out artifacts/af20/results.json
```

Use fresh output directories, Python 3.13/Torch 2.10 CPU locally, or the pinned Python 3.12/Torch 2.5.1 CPU hosted environment where available. The trained models only propose legal actions; the host reconstructs every accepted rule firing. Historical AF-19 negative results remain immutable.
