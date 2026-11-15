/-!
AF-11: exact finite Boolean influence of a single premise on a Horn goal.
This proves truth-table statements in Lean 4, independently of neural weights.
The latching theorem (AF-10 CausalCut.lean) is a separate structural result.
-/

def andGate (a b : Bool) : Bool := a && b

def orGate (a b : Bool) : Bool := a || b

-- Flipping the first argument of AND affects the result iff the second is true.
theorem and_influence (a b : Bool) :
    (andGate (!a) b != andGate a b) = b := by
  cases a <;> cases b <;> decide

-- Flipping the first argument of OR affects the result iff the second is false.
theorem or_influence (a b : Bool) :
    (orGate (!a) b != orGate a b) = !b := by
  cases a <;> cases b <;> decide

-- Symmetric statement for flipping the second premise.
theorem and_influence_second (a b : Bool) :
    (andGate a (!b) != andGate a b) = a := by
  cases a <;> cases b <;> decide

theorem or_influence_second (a b : Bool) :
    (orGate a (!b) != orGate a b) = !a := by
  cases a <;> cases b <;> decide

-- Every gate's output agrees with its own two-input Boolean closure.
theorem gate_and_cases (a b : Bool) :
    andGate a b = (a && b) := by rfl

theorem gate_or_cases (a b : Bool) :
    orGate a b = (a || b) := by rfl
