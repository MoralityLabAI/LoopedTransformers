import Lean
import MathCan.FundedHooks

/-!
# CHTR-05: shared-ledger settlement budgets and potential certificates

These kernel-checkable theorems concern arbitrary state-dependent registered
transitions, including before/after callbacks and arbitrary account transfers.
They treat pool lower values as explicit hypotheses. The rational AM-GM
construction used to obtain such bounds appears in the companion paper and
exact oracle; that nonlinear construction has not yet been ported to Lean.
-/

namespace MathCan.Settlement

universe u v

/-- A complete marked ledger stays constant under any sequence of locally
    conserving state-dependent updates. -/
theorem marked_total_conserved
    {State : Type u} {Action : Type v}
    (step : Action → State → State) (totalValue : State → Int)
    (locallyConserved : ∀ a s, totalValue (step a s) = totalValue s)
    (s : State) (actions : List Action) :
    totalValue (HookState.endpoint step s actions) = totalValue s := by
  exact FundedHooks.conserved_along step totalValue locallyConserved s actions
set_option maxHeartbeats 100000

/-- Exact partition of terminal gains, including custodians, sponsors and pools.
    Every participant must be represented in exactly one of these accounts. -/
theorem exact_five_account_partition
    (trader₀ trader₁ cust₀ cust₁ sponsor₀ sponsor₁ pools₀ pools₁ : Int)
    (h : trader₁ + cust₁ + sponsor₁ + pools₁ =
         trader₀ + cust₀ + sponsor₀ + pools₀) :
    trader₁ - trader₀ = (cust₀ - cust₁) +
      (sponsor₀ - sponsor₁) + (pools₀ - pools₁) := by
  omega
#eval IO.println "RAM03_AXIOMS_BEGIN"
#print axioms exact_five_account_partition
#eval IO.println "RAM03_AXIOMS_END"
end MathCan.Settlement
