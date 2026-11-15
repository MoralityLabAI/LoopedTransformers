# AF-21: exact greedy policy conditions for shortest Horn proofs

## Finite Horn proof-state graph

Let S contain the already-proved atoms in a finite Horn program R. For an enabled rule a with head h, the host transition is T(S,a) = S union {h}, with each premise of a in S. The host refuses unenabled actions. STOP terminates without adding a fact.

For any state from which the target goal g is reachable, let d(S) be the minimal number of further legal rule firings needed to add g; use infinity for an unreachable goal. The finite Horn state space ensures an exact shortest-action oracle via breadth-first search. For a reachable, unsolved state, define the optimal action set A*(S) = {a legal: d(T(S,a)) + 1 = d(S)}. This set is nonempty.

## Exact score dominance condition

For a greedy proof-search policy with action scores s(S,a) and STOP score s(S,STOP), the sufficient local condition is

    max_{a in A*(S)} s(S,a) > s(S,STOP)

together with

    max_{a in A*(S)} s(S,a) > max_{b in A_legal(S) excluding A*(S)} s(S,b).

Strict inequalities avoid ambiguities from tie breaking. If both hold, the host chooses an optimal rule, and d decreases by exactly one.

The chosen optimal rule need not be unique. When every legal rule is optimal, the second comparison is vacuous. When the goal is already proven or unreachable, the host may legitimately STOP; the proof-completion claim concerns reachable unsolved states.

## Induction and Lean correspondence

If a policy satisfies the sufficient score condition at every visited reachable unsolved state, its selected host transition decreases d by one. With initial distance d(S0) = n, repeated application yields d(S_n) = 0 in n rule firings. The Lean theorem in [GreedyStop.lean](../../experiments/activation_funnel_af21/GreedyStop.lean) proves the purely general countdown implication:

    cost(step s) + 1 = cost(s), whenever cost(s) is nonzero

implies

    cost(runSteps step n s) = 0, when cost(s) = n.

Lean 4.19.0 compiled the source-pinned theorem in [AF-21 CI #43729435530](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43729435530). This theorem assumes exact local descent. The neural policies have no kernel-certified guarantee that they meet it.

## Difference between training surrogate and executed policy

AF-20 used the rule-mass STOP gap

    m_mass = s_STOP - log(sum_{a legal} exp(s_a)).

AF-21 uses the exact executed STOP decision gap

    m_greedy = s_STOP - max_{a legal} s_a.

For k legal rules,

    m_mass <= m_greedy <= m_mass + log(k).

The upper bound follows from max <= logsumexp <= max + log k. For example, two enabled rules with logits 0 and 0, and STOP logit 0.2, yield m_mass approximately -0.493 while m_greedy = +0.2: a negative mass gap can coexist with greedy STOP.

The AF-21 greedy-only objective trains the correct STOP-versus-rule max, but a state-independent margin target on training examples cannot force inequalities to hold on unseen Horn compositions. The extra ranking term directly compares best optimal and best suboptimal rules, yet both terms can conflict through shared feature and cell parameters. A low average loss need not establish either strict inequality at every state a rollout visits.

## Mechanistic interpretation of the scored failure

Under the frozen hosted run, the combined greedy plus optimal-rule model solves less than 12% of provable depth4 examples on average over eight seeds, compared with approximately 32% under the BCE baseline. Seed 29 chooses STOP immediately on all provable heldout cases at both depths despite satisfying train/dev eligibility.

The counterexample is an out-of-distribution policy decision, while source-bound host proof soundness remains intact. Zero false accepted Horn proofs follow from the independently enforced legality gate. The failed scientific gate measures solved goals, premature STOP and connected-distractor robustness.

The most discriminating next test is a recurrent shortest-cost value model whose update explicitly conditions on the host successor states T(S,a). The host should still be the sole authority for rule legality and transition, with model value estimates judged by Bellman residual, action rank and whole-episode proof completion. Keep the new test on a separate registration and preserve this AF-21 result.
