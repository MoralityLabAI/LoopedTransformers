# AF-17 mathematical follow-up: the two-loop STOP causal cone

*Post-hoc analysis of the frozen AF-17 models. The source, threshold, corpus split and scored results remain unchanged.*

## Local influence theorem

Let `A` be atom vertices and `R` rule vertices in the public **bipartite rule-incidence graph**. A TypedLoop cell update computes each new rule vector using its prior rule state and the prior atom states incident to its declared premise/head edges. The next atom vector uses that atom's prior state and the updated rule states incident to that atom. The attention implementation normalizes **only within incidence-masked neighborhoods**.

Define the distance `dist_B(u,v)` to be shortest undirected incidence-edge distance in this bipartite graph. At the start, each atom feature encodes its own initial facts and host-known bit. For any fixed weights, graph, rule features, readout location and state initialization, after `L` complete TypedLoop updates the representation of an atom `v` depends only on initial atom features at nodes whose bipartite graph distance from `v` is at most `2L`. This holds by induction: the new rule state reads adjacent old atoms (one incidence edge) and the new atom state reads only adjacent new rules (a second incidence edge). LayerNorm, nonlinearities and the head are token-local and preserve the dependency bound.

**Corollary.** For two public inputs with the same incidence graph and the same initial features inside the closed radius-`2L` neighborhood of the goal atom, the model's goal-atom vector and every goal-only readout are equal after `L` updates. AF-17 sets `L=2`. Its STOP, reachability and distance heads consequently have a graph-radius-4 causal cone in the bipartite incidence graph, equivalent to two ordinary Horn inference edges along a simple chain.

This argument concerns the **goal-only heads**. Individual rule-action logits read the updated rule states at other physical locations and can react to distant facts through their own local neighborhoods. The entire action argmax therefore need not be invariant.

## Certified semantic counterexample pair

For `dual_fallback` at depth 3, 4 or 5, hold graph structure and all initial features fixed except initial fact atom 0:

- Recipient initial facts: `(0,1,1,1)` at logical atoms `(0,1,7,8)`. The goal is **unreachable**.
- Donor initial facts: `(1,1,1,1)` at those same nodes. The goal is **reachable**, with exact minimum Horn rule-fires `7`, `9` or `11` for depths 3, 4 and 5.

Fact atom 0 reaches the goal only through relay and subgoal dependencies exceeding the two-call causal cone. The host owns the exact reachability label. Both public graphs retain the same physical permutation and rule incidence.

I evaluated all **four seeds × four trained arms × three depths = 48** frozen-checkpoint pairs. Every pair changed the true goal reachability. The differences in STOP logit, auxiliary reachability logit, and distance value were **exactly 0.0**. Other rule-action logits do differ on these pairs, establishing that the experiment targeted the local goal readout rather than accidentally comparing completely identical model inputs.

The exact aggregate and per-model logits are retained in `posthoc_receptive_field.json`. The independent local replay script is `af17_receptive_field_probe.py`.

## Capped Bellman-supervision collision

AF-17 source Bellman targets use `Q_cap(S,a)=min(10,1+d(S_a))` for rule fires and 10 for a premature STOP. When an optimal legal rule has `1+d(S_a) >= 10`, its cost label is also **10**. An equally legal dead-end action receives the same label, erasing the Bellman cost-ranking gap on that pair. The shortest-action CE remains informative where the supervised costs saturate.

Among the **74 provable** depth5 heldout inputs, **eight** have exact minimum proof depth 11, and **four** have depth 10. All of their initial optimal-action costs saturate the target 10. The depth4 distribution contains four distance-8 and eight distance-9 cases, where the regression distinguishes some but not all alternatives. This is a second reason the current capped cost loss cannot give a uniform action-priority certificate at arbitrary proof depth.

These are structural and numerical observations from the fixed AF-17 architecture. They motivate a separate experiment with loop counts scaled to the graph dependency radius and loss targets whose cap exceeds the largest heldout shortest path, subject to a new frozen registration. Absolute depth-generalization claims would still require held-out evaluation with independent host proof traces.
