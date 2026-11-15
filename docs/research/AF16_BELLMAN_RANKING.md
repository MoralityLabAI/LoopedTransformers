# AF-16 mathematical follow-up: shortest-proof Bellman ranking and STOP calibration

This is a post-registration analysis of the completed AF-16 source-bound experiment. It changes no AF-16 loss, threshold, seed, or heldout set.

For a finite Horn state \(S\), each host-legal rule action \(a\) produces \(S_a=S\cup\{h_a\}\). Let \(d(S)\) denote the minimum legal rule fires needed to reach the goal, taking \(+\infty\) for an unprovable state. On a provable unsolved state,

\[
 d(S)=1+\min_{a\in A(S)}d(S_a).
\]

The exact action-cost-to-go is \(Q(S,a)=1+d(S_a)\). Any minimal-cost action completes one step on a shortest derivation. In contrast, STOP has effectively infinite cost on such a state because it terminates without proving the reachable goal. This gives a direct mechanism for the AF-16 early-stop failures.

## Ranking-gap certificate

Suppose a learned score \(\widehat Q(S,a)\) approximates the finite action cost \(Q(S,a)\) uniformly by at most \(\varepsilon\), and every nonoptimal legal action has exact cost at least \(\Delta>0\) above the optimum. Then

\[
\widehat Q(S,a_{\rm sub})-\widehat Q(S,a_{\rm opt})
\ge\Delta-2\varepsilon.
\]

Consequently, if \(2\varepsilon<\Delta\), every approximate minimizer of \(\widehat Q\) is an exact optimal action. Since the finite legal action costs are integer-valued, strict nonzero gaps satisfy \(\Delta\ge1\), and uniform error below \(1/2\) suffices for exact ranking where such a gap exists.

**Proof.** Write the estimated costs as true cost plus signed error bounded in absolute value by \(\varepsilon\). For any optimal \(a_*\) and nonoptimal \(b\),

\[
\widehat Q(S,b)-\widehat Q(S,a_*)
\ge[Q(S,b)-\varepsilon]-[Q(S,a_*)+\varepsilon]
\ge\Delta-2\varepsilon.
\]

A positive bound excludes nonoptimal minimizers. The result applies to states with a finite shortest proof and cannot be used to equate a stopped-but-unprovable case with an unsupported proof.

## Experimental implication

AF-16's frontier loss ranks optimal versus nonoptimal **enabled** rule logits. It does not provide a state-dependent numerical estimate of the cost of stopping. The heldout inspection found immediate STOP on all provable depth-4/5 examples for seeds 17 and 29 under 240-step frontier training, despite independent host soundness. A future AF-17 should train a calibrated finite Bellman cost for each enabled action and a distinct goal-reachability/STOP head, then evaluate on unseen rule compositions with a stronger extra-compute imitation baseline. Scoring should still be under the original host gate, with full proof traces and independent finite-Horn shortest distance.

This certificate is elementary arithmetic and has not yet been implemented or kernel-checked in Lean. The AF-16 Lean file contains separate Boolean soundness lemmas.
