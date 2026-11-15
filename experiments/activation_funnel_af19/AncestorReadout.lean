/-! AF-19 finite backward-reachability mask lemmas.
The learning rule and geometry of TypedLoop are checked in Python separately.
-/

def backwardStep (upstream : Nat → Nat → Prop) (a : Nat → Prop) : Nat → Prop :=
  fun v => a v ∨ ∃ w, a w ∧ upstream v w

theorem backward_mask_monotone (upstream : Nat → Nat → Prop)
    (a : Nat → Prop) (v : Nat) (h : a v) : backwardStep upstream a v := by
  exact Or.inl h

theorem backward_mask_goal_invariant (upstream : Nat → Nat → Prop)
    (a : Nat → Prop) (goal : Nat) (hg : a goal) :
    backwardStep upstream a goal := by
  exact Or.inl hg

-- A direct edge into an active atom becomes included in one backward step.
theorem backward_mask_contains_direct (upstream : Nat → Nat → Prop)
    (a : Nat → Prop) (from to : Nat)
    (active : a to) (edge : upstream from to) :
    backwardStep upstream a from := by
  exact Or.inr ⟨to, active, edge⟩
