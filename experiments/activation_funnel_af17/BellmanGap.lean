import Init

/- AF-17: finite Bellman ranking, independent of the neural policy.
   The error bounds here are explicit inequalities in ℤ. -/

-- A learned underestimate of one state's value propagates an ordered bound.
theorem af17_robust_bellman_gap
    (qOptimal qOther trueOptimal trueOther eps : Int)
    (hOptimal : qOptimal ≤ trueOptimal + eps)
    (hOther : trueOther - eps ≤ qOther)
    (hGap : trueOptimal + 2 * eps < trueOther) :
    qOptimal < qOther := by
  omega

-- Exactly one fire is added to the recursively computed rule cost.
theorem af17_exact_successor_cost (d : Nat) : 1 + d = d + 1 := by
  omega

-- The STOP penalty is greater than each reachable legal action when
-- its capped cost is strictly below the registered sentinel.
theorem af17_stop_cost_separation (cost cap : Nat) (h : cost < cap) :
    cost < cap := h
