/-!
AF-14 bounded proof-progress lemma. If level zero is available immediately and
successive logical proof levels appear within b recurrent opportunities, every
level n appears by n*b. The Horn-graph induction using a minimum-depth proof
witness is given separately in docs/research/AF14_BOUNDED_PROGRESS.md.
This lemma alone does not certify a neural network or the Horn witness premise.
-/

theorem af14_bounded_chain (b : Nat) (arrival : Nat → Nat)
    (start : arrival 0 = 0)
    (fair : ∀ k, arrival (k + 1) ≤ arrival k + b)
    (n : Nat) : arrival n ≤ n * b := by
  induction n with
  | zero => simp [start]
  | succ k ih =>
      calc
        arrival (k + 1) ≤ arrival k + b := fair k
        _ ≤ k * b + b := Nat.add_le_add_right ih b
        _ = (k + 1) * b := by simp [Nat.succ_mul]

-- Pure Boolean inflationary closure laws used by the synthetic progress oracle.
theorem af14_horn_update_inflationary (old fire : Bool) :
    (old || (old && fire)) = old := by decide

theorem af14_eventual_truth_persists (old new : Bool)
    (h : old = true) : (old || new) = true := by
  simp [h]
