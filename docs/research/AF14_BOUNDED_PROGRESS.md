# AF-14: a bounded-fairness certificate for neural Horn-progress training

A finite definite-Horn program has initial facts `S₀` and a monotone closure
operator `Φ(S) = S ∪ {v : ∃(P → v) ∈ R, P ⊆ S}`. Let `S∞` be its least fixed
point. Define the *minimal proof height* `d(v)` as zero for facts and the least
value `1 + max_{p in P} d(p)` among a node's finite valid Horn derivations.

Assume the execution is inflationary and **b-bounded fair**: whenever all
premises of a rule become true at time `t`, its head appears by `t+b` and stays
true. Then induction on any chosen minimal-height proof gives

`T(v) ≤ b d(v)` for every v in S∞.

The base facts are present at time zero. The chosen minimal proof of height d
has premises of heights at most d-1; by the induction hypothesis every premise
is present by b(d-1), and bounded fairness fires the witness rule by bd. If D
is the maximum minimal derivation height, the execution has reached S∞ by bD.
The proof allows arbitrary early derivations and nondeterministic fair rule
orders. Without the explicit firing-delay assumption there is no finite bound.

The committed Lean theorem `af14_bounded_chain` verifies the abstract
recurrence `t(n+1) ≤ t(n)+b ⇒ t(n) ≤ n*b` under `t(0)=0`. The general
minimal-Horn-witness derivation above is a paper argument; its complete graph
semantics remain outside the current Lean theorem's statement.

## A differentiable surrogate, with explicit limitations

For each actual carried recurrent state, the source model's atom readout gives
a logit margin `m_{v,t} = ℓ_{v,t}(true) - ℓ_{v,t}(false)`. AF-14 adds

`L_bd = E_case mean_{v∈S∞} mean_{t≥max(1,bd(v))} softplus(τ - m_{v,t})`.

Here τ=0.25 and the weight is 0.35. This enforces neither actual rule firing
nor a formal implication from neural logits to Horn semantics. It is a
classification surrogate trained on exact Horn oracle labels. Its practical
question is whether bounded upper deadlines provide a better inductive bias
than supervising the exact synchronous closure wave (AF-12) or only the final
fixed point plus unclocked admissibility (AF-13).

The independent evaluation records deadline correctness and persistent
correctness separately. For a case with no initial facts and no provable atoms,
the positive-deadline term is zero; the unchanged negative/fact soundness terms
still apply. This prevents empty-Horn cases from receiving fabricated positive
targets. A code that never identifies reachable atoms fails the deadline test.

## Interpretation

This is a synthetic proof-state experiment using the original TypedLoop. Lean
checks Boolean oracle facts but does not certify that the neural model
performs formal theorem proving. Existing AF-01 through AF-13 failures remain
immutable. AF-14's preregistered scoring decides whether the new loss actually
improves unseen depth-4/5 cases; an engineering-green run alone is insufficient.
