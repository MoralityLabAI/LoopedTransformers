# NRF-01: Numerical-range-certified periodic loopers

Date: November 15, 2026. Status: exact finite-dimensional toy certificate. No neural checkpoint was evaluated.

OpenAI paper inputs:

- Family 325, A direct proof of the complete Crouzeix inequality (September 26, 2026): https://github.com/openai/math/tree/main/preprints/A-direct-proof-of-the-complete-Crouzeix-inequality-September-26-2026
- Family 328, Fixed Points of Nonexpansive Maps in Reflexive Banach Spaces (September 24, 2026): https://github.com/openai/math/tree/main/preprints/Fixed-Points-of-Nonexpansive-Maps-in-Reflexive-Banach-Spaces-September-24-2026

The finite-dimensional stabilization theorem and its proof are recorded in MathCan at research/compactification/CROUZEIX_FIXED_POINT_SWITCHING.md. Paper 328 motivates nonexpansive recurrence; the present fixed-point proof uses Banach contraction on a closed finite-dimensional domain. The complete Crouzeix estimate is treated as a cited paper-level result. The Jordan specialization has an independent direct norm bound.

A phase acts on the box C=[-1,1]^2 in Euclidean norm:

K_i(x)=(1-eta_i) F_i(x)+eta_i Pi_C(p_i(A_i)x+b_i).

F_i is an orthogonal box-preserving gate, Pi_C is coordinatewise Euclidean metric projection, and A_i=c_i I+u_i E12. Express p_i(z)=sum_k a_ik(z-c_i)^k. The numerical range W(A_i) is a disk of radius |u_i|/2; q_i=sum_k |a_ik|(|u_i|/2)^k bounds max |p_i(z)| on W(A_i). Thus r_i=1-eta_i+2eta_i q_i is a certified phase Lipschitz upper bound. The cycle gain Gamma=product_i r_i may be below 1 even when one or more r_i exceed 1.

When Gamma<1, the cycle admits a unique attracting periodic orbit, two seeds converge at most as Gamma^t, and any approximate fixed point y has error at most ||Q(y)-y||/(1-Gamma). For per-step disturbance <=delta, D_0=0, D_{i+1}=r_i D_i+delta is a one-cycle error budget; the t-cycle bound is D_m(1-Gamma^t)/(1-Gamma). An unregularized phase residual at an orbit point is <=eta_i diameter(C). Compactness supplies subsequential unregularized periodic orbits if every eta_i tends to zero along a family of contractive cycles.

Run the dependency-free exact-rational tests:

    python -m unittest tests.test_numerical_range_certificate -v

Nine tests cover analytical disk arithmetic, two-seed contraction, post-hoc error-radius certification, cyclewise expansive-phase compensation, residuals, 25 disturbance cycles, input guards, and the false confidence of a spectral-only certificate on a nonnormal nilpotent matrix.

The result certifies only the stated algebraic toy family. Native model experiments require rigorous spectral-set enclosures, gate nonexpansiveness checks, and explicit phase schedules. It does not establish safety, capability gain, or stability of trained Qwen/TRM/HRM weights.
