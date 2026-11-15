# AF-19: backward rule ancestry and STOP causal support

A Horn rule r has premise atoms Prem(r) and head h(r). Starting from goal g, let B(X) be the union of X with the premises of every rule whose head lies in X. Repeated application of B reaches a least fixed point A_g within at most |A| strict-growth iterations in a finite atom universe. The ancestor-rule mask selects every rule r with h(r) in A_g. An independent search over physical Horn edges verifies the mask used in each scored probe.

For atom-state h_g and rule states u_r after L native recurrent calls, both AF-19 arms compute the same learned projections q = W_Q h_g, k_r = W_K u_r, and v_r = W_V u_r. A masked softmax assigns a weight proportional to exp(q·k_r/4), with the resulting rule summary added to h_g. STOP, distance and reachability heads read that summary. The local arm masks to rules immediately concluding g. The ancestor arm masks to the full backward-ancestor rule set. Both policies have 6,036 trainable parameters and identical tensor arithmetic at a fixed L.

## Causal-cone distinction

Under the source-pinned TypedLoop incidence masks, each cell call can carry influence through at most two atom-rule incidence edges. The goal atom after L calls therefore cannot depend on a source atom farther than 2L edges away. An ancestor readout may directly access a rule state adjacent to that remote fact after the rule has been updated, permitting shorter causal access to STOP.

This is an architectural possibility, not a guarantee of correctly learned logical inference. For fact-0-to-goal distances 8, 10 and 12, all trained two-loop local STOP logits are identical in the paired reachable/unreachable cases. The ancestor STOP logits differ on every registered pair across all four seeds. An independent verifier confirms the original graph, target truth and masks.

Padded rule tokens have zero incidence and remain outside either readout mask. Physically disconnected atoms cannot influence the goal readout through message passing. The observed exact stability under these interventions is a structural result. The study does not claim conditional independence for logically irrelevant features connected by neural feedback paths.

The source AncestorReadout.lean contains elementary backward-mask lemmas. Native Lean 4.19.0 checking remains outstanding; independent neural theorem discovery is outside the experiment's scope.
