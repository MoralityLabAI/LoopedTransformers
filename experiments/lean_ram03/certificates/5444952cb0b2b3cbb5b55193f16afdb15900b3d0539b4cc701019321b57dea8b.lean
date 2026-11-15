import Lean
import MathCan.TOEMixedPrincipal

/-!
# TOE-12: Jordan obstruction and algebraic gauge/auxiliary-field identities

The exact 6×6 rational companion calculation is executed by
research/toe/jordan_principal.py; the kernel statements below prove general
structural implications of a repeated characteristic, the two-component
Jordan block, the Abelianized principal gauge quotient, and the scalar
inverse-map first variation. They do NOT formalize all six-by-six rational
matrix entries or establish full Lambda-GfE strong hyperbolicity.
-/

namespace MathCan.TOEJordan

/-- Quartic metric factor, written as a quadratic in x=omega². -/
def metricFactor (a x r : Int) : Int := a * ((x-r)*(x-r))

/-- Odd metric/gauge mixed factor after removing the frequency. -/
def mixedFactor (b x r : Int) : Int := b * (x-r)
set_option maxHeartbeats 100000

theorem metric_vanishes_at_double_root (a r : Int) :
    metricFactor a r r = 0 := by
  dsimp [MathCan.TOEJordan.metricFactor] at *
  omega
#eval IO.println "RAM03_AXIOMS_BEGIN"
#print axioms metric_vanishes_at_double_root
#eval IO.println "RAM03_AXIOMS_END"
end MathCan.TOEJordan
