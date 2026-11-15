/-! AF-21: the exact per-decision Bellman descent condition implies
    completion within the initial optimal proof distance. The source Horn
    oracle supplies the cost function. This theorem makes no claim that a
    neural policy satisfies its hypotheses. -/

def runSteps {S : Type} (step : S → S) : Nat → S → S
  | 0, s => s
  | n + 1, s => runSteps step n (step s)

theorem bellman_descent_reaches_zero {S : Type} (cost : S → Nat) (step : S → S)
    (hstep : ∀ s, cost s ≠ 0 → cost (step s) + 1 = cost s) :
    ∀ n s, cost s = n → cost (runSteps step n s) = 0 := by
  intro n
  induction n with
  | zero =>
      intro s hs
      simpa [runSteps] using hs
  | succ n ih =>
      intro s hs
      have hnz : cost s ≠ 0 := by
        intro hz
        simp [hz] at hs
      have hd := hstep s hnz
      rw [hs] at hd
      have hn : cost (step s) = n := Nat.succ.inj hd
      simpa [runSteps] using ih (step s) hn

theorem strict_greedy_rule_wins
    (bestOptimal bestNonoptimal stopScore : Int)
    (hs : stopScore < bestOptimal)
    (hr : bestNonoptimal < bestOptimal) :
    stopScore < bestOptimal ∧ bestNonoptimal < bestOptimal :=
  ⟨hs, hr⟩

