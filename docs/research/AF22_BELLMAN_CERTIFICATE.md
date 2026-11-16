# AF-22 mathematical follow-up: Bellman backup and margin certificates

This note derives two **conditional** statements about finite host-state graphs. The neural model's satisfaction of those conditions is empirical. The source Lean module proves exact-cost descent termination; the sup-norm and decision-gap arguments below are elementary mathematical consequences of min-plus backups, checked numerically on held-out traces only after the registered score was frozen.

## Host graph and receding horizon

Let `S` be a finite set of Horn proof states, with a public, host-certified transition `T(s,a)` for every currently premise-enabled rule `a`. STOP has its own cost boundary `c_STOP(s)`. A true goal-containing state has cost zero. All other leaf costs are estimated from a frozen model's `distance` and `reachability` heads:

\[
 p(s)=\sigma(r(s)),\qquad
 V_0(s)=p(s)\,\mathrm{clamp}(\mathrm{softplus}(\hat d(s)),0,16)+(1-p(s))\,16,\qquad
 c_{\mathrm{STOP}}(s)=16p(s).
\]

At a reached goal, both outputs are host-clamped to zero. Receding horizon value iteration is

\[
(\mathcal TV)(s)=\min\!\Big(c_{\mathrm{STOP}}(s),\min_{a\in A_{\mathrm{rule}}(s)}\big[1+V(T(s,a))\big]\Big).
\]

`VI-1` compares `1+V_0` on one-step successors; `VI-2` compares `1+TV_0` on those successors. The scores are calculated on exactly the same trained weights, which makes this an inference-algorithm experiment. For states where STOP should be terminal, the explicit STOP branch is part of the min-plus operator.

## Sup-norm nonexpansiveness theorem

For any two finite value assignments `V,W` over the same legal-transition graph and the same stop-cost boundary, set `epsilon = max_s |V(s)-W(s)|`. Then

\[
 \|\mathcal TV-\mathcal TW\|_\infty\le\|V-W\|_\infty=\epsilon.
\]

Indeed for every legal action `a`, `1+V(T(s,a))` and `1+W(T(s,a))` differ by at most epsilon; the fixed STOP candidate agrees exactly. Since taking a minimum is 1-Lipschitz in the maximum norm, both directional inequalities hold. Induction yields the same bound for every finite number of synchronous backups. If STOP boundaries differ by at most `epsilon`, their inclusion in the same minimum gives the same bound with `epsilon` chosen as the maximum of the boundary and leaf errors.

**Interpretation:** extending the planner's horizon cannot *amplify* a uniformly bounded numerical leaf error through min-plus operations. It also cannot repair an inaccurate neural leaf cost when the source information needed for the correct decision lies beyond the explored horizon. Additional recurrence changes which candidates compete, and the model may have large or poorly calibrated errors on those candidates.

## Strict action-gap certificate

Let the exact optimal rule/STOP action cost be `Q*(a*)`, and let every strictly nonoptimal action have exact cost at least `Q*(a*) + Delta`, where `Delta > 0`. If the learned backed-up action values satisfy

\[
 \max_{a\in A_{\mathrm{legal}}\cup\{STOP\}} |\widehat Q_2(a)-Q^*(a)|\le\epsilon,
 \quad 2\epsilon<\Delta,
\]

then the greedy VI-2 action is optimal. For the best optimal action its estimated cost is at most `Q*(a*) + epsilon`. Every nonoptimal candidate has estimated cost at least `Q*(a*) + Delta - epsilon`, which is strictly larger under the stated condition. Ties among optimal actions are harmless. The proof is exact arithmetic and requires a *uniform* bound across the decision's relevant candidates.

In a finite graph with a true proof distance `d(s)`, a policy that always selects a successor with `d(T(s,a)) = d(s)-1` completes the goal in exactly `d(s)` rule firings. AF-22's Lean theorem checks this conditional descent consequence. The theorem makes no claim that the neural model learns `d` or that the two-step lookahead reaches the optimal solution under a restricted horizon.

## Retrospective numerical diagnostics on original hosted checkpoints

The [posthoc value diagnostic](AF22_POSTHOC_VALUE_DIAGNOSTIC.json) reconstructs the exact host BFS cost and true STOP target from public rule incidence at every stored node in each *initial VI-2 decision* on provable heldout depth-4/5 inputs. It computes a conservative `epsilon` across all stored neural leaf and STOP-boundary values, then checks the strict action-gap certificate against the exact local Bellman action costs. Every observed certified decision selects an optimal action. These truth values are analysis-only and were never supplied to the frozen planner.

In depth-5 cases, the empirical certificate rates range from zero to 15% across seeds. The learned STOP costs are often low even on provable goals, causing the min-plus planner to select STOP from a mathematically correct backup of incorrect predictions. Seed 89 gives the strongest planning advantage in the registered hosted study, with a 15% depth-5 certificate rate, 40% initial correct actions, and 40% complete proofs. This does not establish an extrapolation guarantee.

## Next discriminating experiment

An AF-23 study could hold the host transition graph, training states and true output budgets constant while calibrating the value head specifically on *out-of-distribution successor states* and measuring whether the empirical Bellman error bound falls below half the competing-action gap. This is a concrete test of value-error propagation. A separate larger policy or search-width ablation would isolate capacity and planning cost. Retain all present negative and ineligible results and register any additional experiment independently.
