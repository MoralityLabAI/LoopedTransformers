# AF-14 mathematical postmortem: why bounded deadlines need semantic soundness

Let S∞ be the finite least Horn closure, d(v) a minimal proof height, and
m(v,t) the neural logit margin for class true. AF-14's additional objective
penalizes negative margins for v∈S∞ at recurrent times t≥2d(v).

**Degeneracy proposition.** For any fixed corpus containing at least one
unreachable atom, there exists an unsound classification sequence whose
bounded-positive-deadline loss approaches zero. Set every node's true-class
logit margin to M at every time. For each reachable node, every softplus term
`softplus(τ−M)` tends to zero as M→∞. Every unreachable atom is simultaneously
misclassified as true. Thus the bounded-positive-deadline loss by itself is
insufficient to ensure Horn soundness. AF-14 also retained AF-13's soft
soundness penalties, but the observed model remained far from a legal
intermediate Horn state, with unsupported truths increasing on both deeper
held-out families.

**Structural soundness theorem.** Let S₀ be initial facts. For each discrete
step choose any subset F_t of rules enabled by S_t, and set

`S_{t+1} = S_t ∪ {head(r) : r∈F_t and prem(r)⊆S_t}`.

Then for all t, `S₀⊆S_t⊆S∞` and `S_t⊆S_{t+1}`. Proof: the base case follows
from the least-closure definition. If S_t⊆S∞, every fired head with premises
in S_t belongs to S∞ since S∞ is rule-closed; union preserves the inclusion.
Monotonicity follows immediately from the union definition. If every rule
continuously enabled at t is fired by t+b, induction on minimal proof height
further gives `S_{bD}=S∞`, where D=max d(v).

The soundness theorem holds for arbitrary valid asynchronous firing policies;
the bound assumes bounded fairness and monotone rule availability. A neural
model using unrestricted continuous logits need not satisfy these structural
hypotheses. A future model could use a differentiable score to rank rule
firings while committing discrete semantic states through the Horn rule
operator and exposing the resulting certificates to an independent verifier.

AF-14's evidence shows a distinct failure: on eligible seed 17, stable
post-deadline correct classifications improved while final-Q, all-atom closure,
and counterfactual goal behavior deteriorated. A valid forward update needs
semantic admissibility throughout recurrence as well as eventual progress.

The AF-14 Lean file proves an abstract bounded-delay chain, not the complete
finite Horn update induction. The stronger structural theorem above is a
mathematical argument to formalize in a separate Lean corpus before claiming
kernel certification.
