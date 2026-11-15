/-! AF-20: elementary paired reachability/STOP target identities.
    This formalizes the target contract, not neural reliability. -/

def desiredStop (reachable : Bool) : Bool := !reachable

theorem flip_changes_stop (a b : Bool) (h : a ≠ b) :
    desiredStop a ≠ desiredStop b := by
  cases a <;> cases b <;> simp_all [desiredStop]

theorem same_reach_same_stop (a b : Bool) (h : a = b) :
    desiredStop a = desiredStop b := by
  simpa [desiredStop] using congrArg Bool.not h

theorem goal_truth_preserved_by_equal_distance
    (left right : Nat) (h : left = right) :
    (decide (left < 16)) = (decide (right < 16)) := by
  simp [h]
