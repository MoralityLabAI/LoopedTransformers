# AF-18: recurrent causal-cone and Bellman-cap sweep

**Status:** preregistered source. Reuses the AF-17 exact training/dev cases and model parameters. A new public-feature manifest fixes 416 prospective heldout cases across previously unseen physical graph layouts 57–61. The AND/OR Horn templates and host-owned proof gate remain unchanged.

The prior AF-17 results found exact equality of goal-only STOP, reachability, and distance readouts for two-hop recurrence even when a distant initial fact changes the true goal from unreachable to reachable. This study trains the same architecture at **2, 4, 6, and 8** cell calls, each from identical seeded initial weights. In parallel, the L=2 **cap 10** control isolates the effect of clipping a valid 10- or 11-fire proof to the same target as an unreachable action. The main arms use cap 16, above the maximum number of distinct atoms in the proof state.

## Locality argument

The original TypedLoop rule update depends only on its previous rule state and directly incident atom states; the subsequent atom update receives only incident, updated rules. In the undirected atom–rule bipartite incidence graph, an atom readout after L iterations therefore depends only on source-atom features within radius `2L` edges. The argument follows by induction using the exact masks of the original cell. LayerNorm and MLP operations mix within each token and do not expand the graph support.

Our registered paired examples toggle the initial fact at logical atom 0, while keeping graph, other fact bits, and all host inputs unchanged. Its graph distance to the final goal atom is 8, 10, or 12 incidence edges, for logical proof depths 3, 4, and 5. Where `distance > 2L`, STOP, reachability, and distance-head logits must coincide to numerical tolerance for **every** parameter choice. Where `distance <= 2L`, sensitivity becomes structurally possible and is measured; its occurrence depends on the learned weights and is not guaranteed by the graph argument. Rule-action logits are separate readouts and can change at other graph locations.

## Target construction

An independent host oracle obtains the minimum number of rule fires remaining from a state. For a legal rule a, `Q(S,a)=1+d(S_after)` when the goal remains reachable. A dead-end rule, or prematurely choosing STOP while the goal is still reachable, receives penalty `cap`. A valid STOP after finishing or when unreachable has cost zero. AF-18 compares `cap=10` with `cap=16` at two loops, then holds `cap=16` while sweeping cell calls.

The general mathematical inequality `minimum proof fires <= 15` follows from the monotone host rule: each fire adds a previously absent atom to a universe of 15. Thus `cap=16` leaves a strict positive action-value gap between any shortest successful proof and an unreachable/early STOP label. This is a label-separation property; it does not imply that the network learns that separation or solves unseen proofs.

## Reproduce

From repo root, using Python 3.12 / Torch 2.5.1+cpu for hosted fidelity or Python 3.13 / Torch 2.10 CPU for local engineering:

```
python -m unittest discover -s experiments/activation_funnel_af18/tests -p 'test_*.py' -v
python -O -m unittest discover -s experiments/activation_funnel_af18/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af18.experiment manifest --out /tmp/af18_manifest.json
python -m experiments.activation_funnel_af18.experiment train --seed 17 --out artifacts/af18/s17/training
python -m experiments.activation_funnel_af18.experiment evaluate --seed 17 --training artifacts/af18/s17/training --out artifacts/af18/s17/evaluation
python -m experiments.activation_funnel_af18.verify --seed 17 --training artifacts/af18/s17/training --evaluation artifacts/af18/s17/evaluation --out artifacts/af18/s17/independent_audit.json
python -m experiments.activation_funnel_af18.aggregate --root artifacts/af18 --out artifacts/af18/results.json
```

Replace seed 17 with 29, 43, and 61 and complete all four independent seed directories before aggregation. The source is checked against `fresh_manifest.json`, and evaluation checks named NumPy weights, proof actions, logits, host budgets, BFS Bellman targets, and the sampled causal-cone replay.

## Guardrails

The host applies every rule only when the current symbolic proved set contains its complete premise, and rules add new facts monotonically. Neural logits cannot fabricate a valid proof. Fresh heldouts use new physical permutations of the AF-17 logical templates; the experiment measures recurrence-depth and cost-cap effects and has no claim about unseen logical operators. Scientific PASS/FAIL/INCONCLUSIVE and engineering integrity are recorded separately. The AF-01–AF-17 sources and results remain untouched.
