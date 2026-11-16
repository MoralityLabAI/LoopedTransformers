# AF-22 | Source-gated recurrent Bellman value iteration

All three planning modes share a **single** 6,036-parameter native TypedLoop checkpoint for each of eight seeds. This is a controlled inference-time algorithm comparison. The original AF-21 trainer, value head and graph-oracle sources remain unchanged.

The host enumerates at most two legal Horn rule fires from the current proved-atom state and checks each transition against the original premise/head incidence. The neural model supplies only the distance and reachability heads at the observed successor states. It receives no proof-cost ground truth during evaluation.

For an unproved-goal state, define \(p=\sigma(r(s))\), \(d=\mathrm{clamp}(\mathrm{softplus}(\hat d(s)),0,16)\), and

\[
V_0(s)=pd+(1-p)16,\qquad C_{stop}(s)=16p.
\]

At already-proved-goal states, both are zero by host authority. The one-step policy chooses the lowest \(1+V_0(s')\), or the STOP boundary cost. The two-step policy first computes \(V_1(s')=\min\{C_{stop}(s'),1+\min_b V_0(s'')\}\) and then chooses \(1+V_1(s')\) versus STOP. The direct control chooses the original source-hash-pinned rule/STOP logits from the **same** checkpoint. Physical-slot tie breaking and all masks are frozen.

All scores and physical proof actions are independently audited, including min-plus arithmetic, terminal conditions, model weights and sampled full native forwards. Finite synthetic Horn proofs are certified by their explicit host transition traces. Lean's theorem proves the conditional implication of exact Bellman descent, not neural correctness or new Lean theorem discovery.

Reproduce locally from the repository root:

```bash
python -m unittest discover -s experiments/activation_funnel_af22/tests -v
python -m experiments.activation_funnel_af22.experiment manifest --out /tmp/af22_manifest.json
python -m experiments.activation_funnel_af22.experiment train --seed 17 --out /tmp/af22-s17/training
python -m experiments.activation_funnel_af22.experiment evaluate --seed 17 --training /tmp/af22-s17/training --out /tmp/af22-s17/evaluation
python -m experiments.activation_funnel_af22.verify --seed 17 --training /tmp/af22-s17/training --evaluation /tmp/af22-s17/evaluation --out /tmp/af22-s17/independent_audit.json
```

The frozen registration is [`PROTOCOL.json`](PROTOCOL.json). Preserve every failed result, source checksum, and nonqualified seed. The AF-21 results remain immutable.
