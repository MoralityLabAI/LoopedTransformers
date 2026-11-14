import Lean
import MathCan.V4Transient

/-!
# CHTR-10: hook-return splits and multi-tick accounting

A small proof kernel for signed caller/hook/pool deltas and tick-level input
accounting. The bytecode observation, Keccak slots, tick prices and rounding
are checked by the paired Python and pinned Foundry executions. This module
proves generic integer conservation consequences, independent of any EVM claim.
-/

namespace MathCan.HookMultiTick

/-- A pool delta is distributed across the caller and the hook. -/
structure DeltaSplit where
  pool : Int
  caller : Int
  hook : Int
  deriving DecidableEq, Repr

def WellSplit (s : DeltaSplit) : Prop := s.pool = s.caller + s.hook

/-- Source-inspired split: caller receives the pool delta minus the hook delta. -/
def distribute (pool hook : Int) : DeltaSplit :=
  ⟨pool, pool - hook, hook⟩

theorem distribute_sound (pool hook : Int) : WellSplit (distribute pool hook) := by
  simp [WellSplit, distribute]
set_option maxHeartbeats 100000

/-- Every per-currency signed split conserves its total when physical custody
    changes by the opposite of the pool delta. -/
theorem three_account_conservation (split : DeltaSplit)
    (h : WellSplit split) : split.caller + split.hook - split.pool = 0 := by
  all_goals simp_all [MathCan.HookMultiTick.DeltaSplit.caller, MathCan.HookMultiTick.DeltaSplit.hook, MathCan.HookMultiTick.DeltaSplit.pool, MathCan.HookMultiTick.WellSplit]
#eval IO.println "RAM03_AXIOMS_BEGIN"
#print axioms three_account_conservation
#eval IO.println "RAM03_AXIOMS_END"
end MathCan.HookMultiTick
