# Post-LS-SAE04 spectral boundary: what sampled recurrence contraction proves

**Retrospective mathematical analysis, November 16, 2026.** LS-SAE04's new native results are archived in [the audited research report](LS_SAE04_RESULTS.md), with the original scientific gate failing on both 12-puzzle families. The prospective LS-SAE05 protocol was committed independently before constructing its test hidden states. The statements here are classical finite-dimensional linear algebra specialized to a single-token-to-all-tokens LoopedTRM Jacobian. They are mathematical results and do not imply new trained-network measurements.

## 1. Average isotropic tangent gain measures Frobenius mass

Let J:R^d→R^m be the Jacobian mapping one injected source token into the full output state after a fixed number of tied recurrent steps. For a random input direction U uniform on the unit sphere, E[UUᵀ]=I_d/d. Therefore

\[
 \mathbb{E}\|JU\|_2^2=\frac{\operatorname{tr}(J^\top J)}{d}
 =\frac1d\sum_{i=1}^d \sigma_i(J)^2.
\]

The estimator targets the **average squared singular gain**, with an upper bound ||J||op² given by its largest singular value. A small average can coexist with a large maximum when gain concentrates in a narrow input subspace.

Exact rational 512D witness: take J=diag(4,1/20,...,1/20). Its maximum singular gain is exactly 4, while

\[
 \mathbb{E}\|JU\|^2=\frac{16+511/400}{512}
 =\frac{6911}{204800}\approx0.0337451.
\]

The root-mean-squared isotropic gain is about 0.1837. Thus observing low random-probe gain can be compatible with a hidden high-gain singular direction. LS-SAE04 additionally tested a learned SAE decoder direction and task-only head direction on 24 native puzzles; those measured gains constrain those particular directions and finite input states.

## 2. Matrix-free JᵀJ power iteration is monotone in exact arithmetic

Let A=JᵀJ, a positive-semidefinite d×d matrix, and let u₀ have a nonzero projection on a positive singular eigenspace. Define the normalized power iteration

\[
 u_{t+1}=\frac{Au_t}{\|Au_t\|},\qquad
 G_t=\|Ju_t\|.
\]

Write u₀ in an orthonormal eigenbasis of A, with squared coefficient weights c_i² and nonnegative eigenvalues λ_i. For m_k=∑ c_i²λ_i^k, the exact squared gain after t updates is

\[
 G_t^2=\frac{m_{2t+1}}{m_{2t}}.
\]

Cauchy–Schwarz gives the moment log-convexity inequality m_{k+1}²≤m_k m_{k+2}. Consequently the ratios m_{k+1}/m_k are nondecreasing wherever defined, and **G_(t+1)≥G_t**. Every observed G_t lower-bounds the operator norm ||J||op. Numerical failures of the monotone trend beyond a stated tolerance warrant inspecting the native differentiation path and floating-point precision.

When the top eigenvalue λ₁>λ₂ and u₀ has a nonzero component along the top right-singular vector, the relative norm of the other components is bounded by

\[
 \frac{\|P_{\perp}u_t\|}{\|P_1u_t\|}
 \le \left(\frac{\lambda_2}{\lambda_1}\right)^t
       \frac{\|P_{\perp}u_0\|}{\|P_1u_0\|}.
\]

This is a finite-budget convergence guide. LS-SAE05 freezes t=4 updates and one seeded start per puzzle; a failure to discover amplification does not certify that all of J's singular values are small.

## 3. Full-state loss forces a small task-margin derivative

Let P be the orthogonal projection from the full output state onto the selected 512D goal token, and c the fixed task-head true-digit-minus-clean-rival readout covector at that horizon. The scalar task-margin derivative along a unit source vector u is S(u)=cᵀ PJu. Cauchy–Schwarz yields

\[
 |S(u)|\le\|c\|\,\|PJu\|
 \le\|c\|\,\|Ju\|.
\]

When the global tangent ||Ju|| contracts strongly, task-margin sensitivity is forced down unless the head norm grows. The **target-token energy fraction** ||PJu||²/||Ju||² separately measures output redistribution between the edited token and other tokens. Both mechanisms can act together. LS-SAE04 measured low global gains and typically less than 1% of the surviving tangent energy at the goal token after two ACT steps.

## 4. Finite-horizon singular gain and recurrent eigenmode are distinct

For a nonstationary or nonlinear tied recurrence, J is the derivative of the full two-step continuation at one frozen clean trajectory, a product of local step derivatives. The leading singular vector of J measures maximal *finite-horizon* norm transmission. It need not be an eigenvector of a single loop map, and can reveal transient amplification even when each asymptotic mode is stable. Power iteration of JᵀJ is the appropriate matrix-free direction search for this finite objective.

The exact JVP/VJP transcript in LS-SAE05 therefore tests one well-defined question: among all 512-dimensional physical edits at one selected source token, can the model preserve or amplify a direction that LS-SAE04's source SAE, clean task-head and isotropic probes missed? Its prospective event threshold, random-baseline comparison, new puzzle seeds and 2000-cluster bootstrap were fixed separately before test outcomes.

**Proof scope:** these statements use real finite-dimensional linear algebra and textbook moment inequalities. They are written proofs and a rational witness, not Lean kernel certificates or trained-model results.
