# AF-12 post-hoc mechanism diagnostic: neural time versus Horn waves

This diagnostic was calculated **after** the frozen local evaluation. It changes neither the preregistered AF-12 protocol nor its scoring; it is a separate hypothesis-generating analysis of locally verified Python 3.13.5 / Torch 2.10.0 evidence. The hosted pinned runtime has not executed because GitHub Actions could not assign any runner. See [AF12_LOCAL_RESULTS.md](AF12_LOCAL_RESULTS.md).

## Observation

For each retained clean held-out case, the independent AF-12 oracle reconstructs the finite least Horn closure `S[t+1] = S[t] ∪ {head(r): premises(r)⊆S[t]}`. We compare the `subgoal_head`'s argmax on **every physical atom after every upstream native-cell visit** with its wave-indexed Boolean membership target, and count decreases in the predicted true-class probability.

| Seed | Heldout | Arm | All-stage atom accuracy | Probability decreases over atom transitions | Last-step joint Q |
|---:|---|---|---:|---:|---:|
| 17 | Depth 4 | Final relay | 0.5271 | 0.5167 | 1.0000 |
| 17 | Depth 4 | Stepwise closure | **0.6458** | **0.4667** | 0.5625 |
| 17 | Depth 5 | Final relay | 0.5042 | 0.5000 | **1.0000** |
| 17 | Depth 5 | Stepwise closure | **0.6233** | **0.4500** | 0.2500 |
| 29 | Depth 4 | Final relay | 0.4917 | 0.5952 | 0.0625 |
| 29 | Depth 4 | Stepwise closure | **0.4958** | **0.5143** | **0.2500** |
| 29 | Depth 5 | Final relay | 0.4983 | 0.5778 | 0.0625 |
| 29 | Depth 5 | Stepwise closure | **0.5200** | **0.4796** | **0.5625** |

'Probability decreases' means the fraction of physical atom/time pairs satisfying `p_true(t) > p_true(t+1)+1e-4`; it is a descriptive neural-monotonicity measure. Whole-trajectory exact correctness is zero for all recorded arms/depths, even when the final Q prediction is perfect. The all-stage metric averages across atoms and stages, so it does not imply joint correctness.

**Mechanism:** the rollout objective can increase agreement with a rigid wave clock while damaging the final target. Its improvement in per-stage accuracy and reduction in nonmonotone fluctuations coexist with seed17's severe loss of terminal Q accuracy. This isolates a possible mismatch between internal neural computation time and Horn proof depth, rather than a failure of the Horn oracle or of the AF-10 causal cut.

## Mathematical follow-up: asynchronous Horn closure

For finite atoms `U` and positive Horn rules `R`, let

```
T_R(S) = S ∪ {h : some (P→h)∈R has P⊆S}
```

and `S* = ⋃_t T_R^t(S0)`. A **fair asynchronous update schedule** selects any subset of currently enabled new heads at each time:

```
S_(t+1) = S_t ∪ A_t,
A_t ⊆ T_R(S_t) \ S_t.
```

Assume every head that remains enabled is eventually selected. Then `S_t ⊆ S*` for every time by induction, since the exact closure is monotone. Since the atom set is finite and every strict change adds an atom, a fair run eventually stabilizes. If the limiting set omitted a currently derivable head, fairness would force another addition, contradicting stability. The limiting set is therefore an `R`-closed superset of `S0`; by minimality of the least fixed point, it equals `S*`.

This proof requires neither one Horn wave per neural call nor a fixed time at which any particular intermediate atom first appears. It motivates a new **AF-13 registration** that trains monotonic reachability, semantic soundness, and fixed-point correctness while allowing learned propagation schedules.

One possible differentiable surrogate, clearly distinct from the proven discrete statement, uses learned atom membership probabilities `p_h(t)` and soft rule support

```
support_h(t) = max_{r: head(r)=h} product_{a in premises(r)} p_a(t).
```

The surrogate would penalize `relu(p_h(t)-p_h(t+1))²`, unsupported new truth `relu(p_h(t+1)-max(p_h(t),support_h(t)))²`, and final disagreement with `1[h∈S*]`. This **requires new preregistration** and a fixed compute-matched final-relay comparator, with no AF-12 retuning.

The neural representations here are synthetic Horn graph states. Lean's Boolean-Horn certificates concern the oracle and do not certify neural theorem discovery.
