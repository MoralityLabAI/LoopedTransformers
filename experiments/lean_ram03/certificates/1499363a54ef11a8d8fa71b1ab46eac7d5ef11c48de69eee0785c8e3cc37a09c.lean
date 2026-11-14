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

/-- Consequently the quadratic 0-form scalar-log response vanishes for an
isolated TT variation at fourth differential order. -/
theorem scalar_sector_tt_zero (beta invScalar a : Int) :
    (beta * invScalar) * (ttRicciTrace a) * (ttRicciTrace a) = 0 := by
  rw [tt_ricci_scalar_zero]
  simp

/-- Both independent TT helicities receive the same one-form quartic kernel.
At formal wave covector (omega,0,0,p), the integer-normalized polynomial
is k*(omega²-p²)², with all normalization encoded in k. -/
def vectorTT (k omega2 p2 : Int) : Int :=
  k * ((omega2-p2)*(omega2-p2))
set_option maxHeartbeats 100000

theorem vector_tt_on_lightcone (k x : Int) : vectorTT k x x = 0 := by
  dsimp [MathCan.TOEFormSectors.vectorTT] at *
  all_goals simp_all [MathCan.TOEFormSectors.vectorTT]
#eval IO.println "RAM03_AXIOMS_BEGIN"
#print axioms vector_tt_on_lightcone
#eval IO.println "RAM03_AXIOMS_END"
end MathCan.TOEFormSectors
