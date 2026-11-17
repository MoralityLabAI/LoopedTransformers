# AF-18: finite causal support and attenuation of distant proof facts

The AF-18 experiment separates *possible graph influence* from *learned numerical sensitivity*. The following argument applies to the exact incidence-masked TypedLoop cell imported from the parent research repository, with parameter matrices shared across every atom/rule token. It is independent of the trained weights.

## Causal support theorem

Form the undirected bipartite incidence graph whose vertices are atoms and rules, and whose edges connect a rule to each of its premises and head. Assume initial atom embeddings depend on their own public features and host-owned known-fact bit, and initial rule embeddings depend on each rule's own features. The rule update at iteration `t+1` reads its previous rule vector and its directly incident atom vectors from iteration `t`. The atom update then reads its own previous state and updated vectors of directly incident rules. The original TypedLoop attention is masked to these edges, with token-local activations and LayerNorm.

By induction, after `L` complete TypedLoop iterations, an atom vector at vertex `v` is a function solely of original atom features within **bipartite incidence radius `2L`** of `v`, and initial rule features within the corresponding adjacent rule neighborhood. The induction step expands the possible atom-source set by at most two incidence edges, one during atom-to-rule collection and one during rule-to-atom collection.

Thus if two inputs have identical graph incidence, rule features and source-atom features in that radius, their final atom vectors at `v` coincide for any shared network weights. Every readout that reads only that atom also coincides. In AF-17 and AF-18, STOP, auxiliary reachability and distance heads all read only the goal atom.

The theorem is a **necessary reachability condition for influence**. If a changing fact is inside the radius, the network may still produce an almost identical output through attenuation, saturation or learned insensitivity.

## Finite code separation

Every host-accepted Horn action adds one atom not already in the proved set. With 15 atom slots, any successful shortest path has at most 15 rule fires, even when the rule graph is cyclic. For a finite successful action cost `q ≤ 15`, clipping at cap 16 preserves `q < 16`. Dead-end actions and premature STOP have penalty 16, yielding a strictly positive *training-label* margin. Cap 10 instead maps valid 10- or 11-fire proofs to the same penalty as unreachable actions. The label distinction does not certify the learned model's action ordering.

This arithmetic corollary is stated in [CausalCone.lean](../../experiments/activation_funnel_af18/CausalCone.lean). Its hosted Lean 4.19.0 compilation remains outstanding because GitHub Actions preflight received no runner. The full graph-network dependency proof above is a source-checked mathematical argument and an independently replayed numerical invariant, rather than a kernel-checked mechanized formalization of the TypedLoop implementation.

## AF-18 prospective numerical probes

Each pair fixes the physical graph and all initial fact bits except atom 0, which switches the final Horn goal from unreachable to reachable. The independently reconstructed incidence distances from that atom to the goal are 8, 10, 12 for depths 3, 4, 5. This gives earliest possible loop counts 4, 5, 6.

Every frozen AF-18 model was replayed against all three pairs: **20 models × 3 pairs = 60 probes**. All out-of-cone comparisons had goal-only STOP, distance and reachability differences at or below the 1e-6 tolerance, as required. Inside the radius, response magnitudes varied dramatically.

Seed 61, `cap=16`, STOP-logit absolute differences:

| Loop calls | Depth 3 (distance 8) | Depth 4 (distance 10) | Depth 5 (distance 12) |
|---:|---:|---:|---:|
| 2 | 0 | 0 | 0 |
| 4 | 0.0016503 | 0 | 0 |
| 6 | 1.0084963 | 0.0103712 | 0.0001435 |
| 8 | 0.9457798 | 0.0152960 | 0.0007334 |

This model becomes markedly responsive to the depth-3 fact after six calls, with progressively weaker response at depth 4 and a very small response at depth 5. The other seeds also exhibit tiny or zero differences inside the nominal receptive field, even though the structural information path exists.

The **full proof policy** can react differently because its individual rule logits read rule vectors at other locations. Exact equality of the STOP/readout heads is a deliberately narrow diagnostic, not an assertion that all actions must coincide.

## Experimental consequence

Increasing recurrence depth expands the causal support but imposes new optimization and stability demands. AF-18 seed 61 achieves 78.4% depth-4 and 67.6% depth-5 proof success at eight calls; seed 29 falls to 0% at both depths under the same loop count and cost cap. The registered cross-seed hypothesis therefore fails despite the deterministic support theorem being respected by every tested model.

A follow-up should distinguish a global goal readout from local message passing. For instance, allow STOP to attend over current rule states with an explicit and equal-parameter compute control. A test should compare whether the strengthened readout detects distant *relevant* facts without becoming sensitive to logically irrelevant features, and retain the host-only Horn proof validation.
