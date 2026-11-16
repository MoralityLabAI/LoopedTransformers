/-! AF-22: conditional completion for a policy that strictly descends an
    exact nonnegative Bellman cost on each nonterminal Horn state.
    This theorem only certifies the descent implication, not a trained network,
    planner accuracy, or the host graph's learned value estimates. -/

def execute {S : Type} (step : S → S) : Nat → S → S
  | 0, s => s
  | n + 1, s => execute step n (step s)

theorem exact_bellman_descent_terminates {S : Type}
    (cost : S → Nat) (step : S → S)
    (h : ∀ s, cost s ≠ 0 → cost (step s) + 1 = cost s) :
    ∀ n s, cost s = n → cost (execute step n s) = 0 := by
  intro n
  induction n with
  | zero =>
      intro s hs
      simpa [execute] using hs
  | succ n ih =>
      intro s hs
      have hnz : cost s ≠ 0 := by
        intro hz
        simp [hz] at hs
      have hs' := h s hnz
      rw [hs] at hs'
      have hnext : cost (step s) = n := Nat.succ.inj hs'
      simpa [execute] using ih (step s) hnext

/-! The action selector's strict score dominance is an explicit hypothesis. -/
theorem strict_bellman_choice (qOptimal qSuboptimal stopCost : Int)
    (hrule : qOptimal < qSuboptimal)
    (hstop : qOptimal < stopCost) :
    qOptimal < qSuboptimal ∧ qOptimal < stopCost := ⟨hrule, hstop⟩
