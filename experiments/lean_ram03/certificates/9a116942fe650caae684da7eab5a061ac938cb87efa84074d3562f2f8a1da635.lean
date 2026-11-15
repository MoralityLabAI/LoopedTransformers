import Lean
import MathCan.FlashAccounting

/-!
# CHTR-07: a finite-support ghost invariant for Uniswap v4 NonzeroDeltaCount

A declared transient delta map is represented by a finite list of signed
integers, with one position for each observed (caller, currency) pair. Updating
one entry changes the nonzero count by its indicator difference. This is an
accounting refinement obligation for PoolManager's `_accountDelta`; it does not
certify that an external EVM transcript enumerates every slot or call frame.
-/
namespace MathCan.V4Transient

/-- Count contribution of a signed transient delta (a ghost value, not an EVM slot). -/
def nz (x : Int) : Int := if x = 0 then 0 else 1

/-- Finite list model of NonzeroDeltaCount. -/
def count : List Int → Int
  | [] => 0
  | x :: xs => nz x + count xs

theorem nz_nonnegative (x : Int) : 0 ≤ nz x := by
  unfold nz
  split <;> omega

theorem nz_zero_iff (x : Int) : nz x = 0 ↔ x = 0 := by
  by_cases h : x = 0 <;> simp [nz, h]
set_option maxHeartbeats 100000

theorem count_nonnegative (xs : List Int) : 0 ≤ count xs := by
  induction xs
  constructor
  dsimp [MathCan.V4Transient.count] at *
  dsimp [MathCan.V4Transient.count, MathCan.V4Transient.nz] at *
  omega
#eval IO.println "RAM03_AXIOMS_BEGIN"
#print axioms count_nonnegative
#eval IO.println "RAM03_AXIOMS_END"
end MathCan.V4Transient
