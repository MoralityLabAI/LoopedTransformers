/-! AF-18 arithmetic implications of the incidence-radius causal-cone argument.
This lemma captures the numerical separation between the finite Bellman
proof cost and the unreachable cap. The network's locality property requires
its source-checked graph-attention implementation and is tested independently.
-/

theorem bellman_finite_cost_separated (cap cost : Nat) (h : cost < cap) :
    min cost cap < cap := by
  simp [Nat.min_eq_left (Nat.le_of_lt h), h]

theorem influence_outside_radius (distance loops : Nat)
    (h : 2 * loops < distance) : ¬ distance ≤ 2 * loops := by
  exact Nat.not_le_of_gt h
