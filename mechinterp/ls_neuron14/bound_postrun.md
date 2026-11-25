# LS-NEURON-14 interpretation and proposed LS15 conditional LP

## Observed result

Read `ls_neuron14/audit_results.json`, `certificates.json`, and `search14.py`. A read-only comparison of all 96 stored final certificate verdicts against the numerical bound threshold found zero disagreements. For single moves, the point arm certified seven seeds and the bound arm certified four. Seeds 447, 452, and 459 succeeded only under point selection. For paired moves both arms certified nine seeds; point-only successes were 452 and 462, while bound-only successes were 459 and 463.

The proposed selection change failed to increase certificate acquisition in this pilot. The endpoint comparison supports threshold agreement on the observed final models. It leaves the search landscape and continuous-fitting mechanism unresolved. The shared least-squares fitter could steer candidates away from a minimax solution, but LS14 did not vary that fitter and cannot establish this explanation. These results remain PILOT_ONLY, with initialization seed as the independent unit.

## Conditional minimax problem

Fix one codebook, the up-weight scale, and the up-bias scale. Freeze the resulting nonlinear weights and biases. For endpoint `e` with state/input vector `z_e=(h_e,x_e)` and target center `c_e`, define

\[
A_e=C_{\mathrm{skip}}\cdot z_e,\qquad
D_e=\frac15\sum_i C_{\mathrm{down},i}\,g(U_i\cdot z_e+b_i),\qquad
r_e=h_e-c_e.
\]

Let `a` be the positive skip scale, `q` the positive down scale, and `o` the combined offset. The ideal continuous-family error is

\[
e_e(a,q,o)=r_e+aA_e+qD_e+o.
\]

For segment `j` of width \(\Delta_j\), let \(m_{ij}\) be the maximum absolute GELU second derivative on that neuron's preactivation interval. It is obtained from the interval endpoints and contained critical points \(-2,0,2\). Define

\[
\kappa_j=\frac{\Delta_j^2}{40}\sum_i |C_{\mathrm{down},i}|\,U_{ih}^2m_{ij}.
\]

The continuous-region interpolation envelope for this segment is

\[
\max\{|e_{j,L}|,|e_{j,R}|\}+q\kappa_j.
\]

With the inherited box constraints, the four-variable linear program is

\[
\begin{aligned}
\min_{a,q,o,t}\quad &t\\
\text{subject to}\quad
&\sigma(r_e+aA_e+qD_e+o)+q\kappa_j\le t,
&&e\in\{j,L;j,R\},\ \sigma\in\{-1,+1\},\ \text{all cases and segments},\\
&10^{-6}\le a,q\le128,\\
&-8\le o\le8,\qquad t\ge0.
\end{aligned}
\]

In solver row form the coefficients for `(a,q,o,t)` are
`(sigma*A_e, sigma*D_e+kappa_j, sigma, -1)`, with right-hand side
`-sigma*r_e`. Both signs are required for every endpoint. Freezing a sign from the current residual would miss errors after they cross zero. The curvature coefficient contains the down *codes*; multiplying by the optimization variable supplies the down scale exactly once.

Each interior endpoint can use the maximum curvature coefficient of its adjacent segments, reducing duplicate constraints. This reduction is valid because the down scale is constrained positive. With 17 endpoints in each of 32 cases, the reduced program has at most 1088 envelope inequalities.

## Meaning of the LP result

For exact fixed coefficients, this LP globally minimizes the stated enclosure over its conditional continuous family. The up and bias scales remain fixed, and the curvature envelope can remain conservative through lost cancellation.

The deployed decoder first casts each scale to FP32 and then multiplies it by each integer code in FP32. Different code entries can incur different product rounding. The skip bias is also rounded after converting the combined offset. Consequently, the four-variable ideal family differs slightly from the family of stored weights. Floating Gaussian evaluations and a numerical LP solver add further numerical approximations. Treat the LP output as a candidate proposal, decode it, and recompute the chosen selection score before acceptance. The unchanged rational verifier remains the exact-real gate. An LP objective below `3/32` alone does not certify the stored model.

## Proposed LS15 contrast

Use paired moves and fresh matched initialization seeds. Compare **point-minimax LP**, which sets every curvature coefficient to zero, with **interval-minimax LP**, which uses the coefficient above. Hold the endpoint grid, proposal slots, nonlinear anchors, variable bounds, and solve budget fixed. This contrast isolates the added interval padding while both arms optimize a maximum error criterion.

Declare how nonlinear anchors arise before running the cohort. A finite bank made from initial scales and a common initial fit defines a constrained search. An alternative retains an identical nonlinear update in both arms and then applies the conditional LP as the last step on the skip/down scales and offset. A later least-squares fit would change those quantities again and should be omitted from that proposal path. Keeping the best decoded incumbent makes acceptance explicit.

A result from these two LP arms can support a claim about the interpolation term within that pipeline. Comparing it with LS14 supplies historical context. Establishing that the shared least-squares fitter caused LS14's failure would require a concurrent comparison that varies the conditional LP step while preserving the outer procedure.

Predeclare a primary certificate-count contrast and paired seed analysis. Retain exact initial/final checks, decoded rescoring, LP status and feasibility residuals, and solver-call accounting. No new experiments or source edits were performed for this review.
