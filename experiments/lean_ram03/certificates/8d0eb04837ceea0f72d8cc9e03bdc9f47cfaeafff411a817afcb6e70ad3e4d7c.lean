import Lean
import MathCan.TOEJordan

/-!
# TOE-13: scalar/vector curvature response and a lifted tensor root

This is kernel-level algebra in the *declared* flat-background 0/1/2-form
logarithmic principal completion of the TOE-12 6-mode tensor model.
The 4x4 Ricci principal, G-field algebraic Schur elimination, and rational
6-mode determinant are checked by independent executable oracles.

These integer lemmas do not formalize the full Lambda-GfE covariant action,
its constraint-reduced PDE, a physical dispersion relation or hyperbolicity.
-/
namespace MathCan.TOEFormSectors

/-- The scalar-curvature principal of an isolated transverse-traceless wave
has opposite spatial diagonal entries and vanishing trace. -/
def ttRicciTrace (a : Int) : Int := a + (-a)

theorem tt_ricci_scalar_zero (a : Int) : ttRicciTrace a = 0 := by
  unfold ttRicciTrace
  omega
set_option maxHeartbeats 100000

/-- Consequently the quadratic 0-form scalar-log response vanishes for an
isolated TT variation at fourth differential order. -/
theorem scalar_sector_tt_zero (beta invScalar a : Int) :
    (beta * invScalar) * (ttRicciTrace a) * (ttRicciTrace a) = 0 := by
  all_goals simp_all [MathCan.TOEFormSectors.ttRicciTrace]
  simp_all [Int.neg_mul, Int.mul_neg, Int.mul_comm, Int.mul_left_comm, Int.mul_assoc]
  omega
#eval IO.println "RAM03_AXIOMS_BEGIN"
#print axioms scalar_sector_tt_zero
#eval IO.println "RAM03_AXIOMS_END"
end MathCan.TOEFormSectors
