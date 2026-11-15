import Lean
import MathCan.TOETensorConstraints

/-!
# TOE-16: dynamical source curl-isotropy obstruction (conditional algebra)

The foundational two-form source M=Sum_i(u_i u_i^T+lambda v_i v_i^T)
has a matter-derivative Hessian involving U=Sum_i u_i u_i^T, even if
M is isotropic. These integer lemmas formalize the trace differences,
conditional rank gate and linearized mixed-trace controls.

Exact rational 3x3 source Hessians and full determinant second
derivatives are verified independently by research/toe/multiplet_kinetic.py.
We do not formalize the full matrix-rank theorem here, nor claim a
source background solution or physical hyperbolicity.
-/
namespace MathCan.TOEMultipletKinetics

/-- t = alpha W_B, u a unit background curl aligned with magnetic z.
These are the pure one-multiplet Hessian eigenvalues in two directions. -/
def hPerpendicular (t : Int) : Int := 2*(t*t)-2*t

def hParallel (t : Int) : Int := 4*(t*t)-2*t
set_option maxHeartbeats 100000

/-- The exact parallel-perpendicular splitting comes from the
logarithmic response term. The conventional quadratic norm cancels. -/
theorem hessian_parallel_perpendicular_gap (t : Int) :
    hParallel t-hPerpendicular t = 2*(t*t) := by
  simp_all [MathCan.TOEMultipletKinetics.hParallel, MathCan.TOEMultipletKinetics.hPerpendicular]
  omega
#eval IO.println "RAM03_AXIOMS_BEGIN"
#print axioms hessian_parallel_perpendicular_gap
#eval IO.println "RAM03_AXIOMS_END"
end MathCan.TOEMultipletKinetics
