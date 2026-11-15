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
set_option maxHeartbeats 100000

theorem distribute_sound (pool hook : Int) : WellSplit (distribute pool hook) := by
  dsimp [MathCan.HookMultiTick.WellSplit, MathCan.HookMultiTick.distribute] at *
  all_goals simp_all [MathCan.HookMultiTick.WellSplit, MathCan.HookMultiTick.distribute]
#eval IO.println "RAM03_AXIOMS_BEGIN"
#print axioms distribute_sound
#eval IO.println "RAM03_AXIOMS_END"
end MathCan.HookMultiTick
