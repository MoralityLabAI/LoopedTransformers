# LS-NEURON-14 independent bound and protocol review

## Mathematical enclosure

Write the selected scalar recurrence as

\[
F(h,x)=h+s\cdot(h,x)+b_s+\frac15\left[\sum_i v_i g(u_{ih}h+u_{ix}\cdot x+b_i)+b_d\right],
\qquad g(z)=z\Phi(z).
\]

For fixed input, the affine terms contribute zero second derivative. Direct differentiation gives

\[
g''(z)=(2-z^2)\phi(z),\qquad g'''(z)=z(z^2-4)\phi(z).
\]

The maximum of \(|g''|\) on a closed interval occurs at an endpoint or at a contained member of \(\{-2,0,2\}\). The additional corners at \(\pm\sqrt2\) have value zero. For a state segment \([l,r]\), let \(m_i\) bound \(|g''|\) over the affine preactivation range of neuron \(i\). Then

\[
M_{l,r}=\frac15\sum_i |v_i|u_{ih}^{2}m_i
\]

bounds \(|F''|\) throughout that segment. If \(e(h)=F(h,x)-c_{\mathrm{target}}\), the linear interpolation remainder gives

\[
\sup_{h\in[l,r]}|e(h)|
\le \max\{|e(l)|,|e(r)|\}+\frac{(r-l)^2}{8}M_{l,r}.
\]

Taking the maximum over every partition segment and all 32 state/input cases yields a continuous-region error enclosure in exact arithmetic. A value at most \(3/32\) proves the fixed-region inclusion with radius \(1/8\) and additive disturbance magnitude \(1/32\), provided every arithmetic enclosure is sound.

The bound discards cancellation when it sums neuron magnitudes. A network with canceling pairs can therefore receive a loose score. Refining the state partition decreases the interpolation penalty quadratically for fixed weights, with the global curvature bound giving a uniform convergence guarantee.

## Numerical score and exact gate

Computing the formula in float64 produces a numerical enclosure proxy. The analytic formula alone does not certify the evaluation of Gaussian functions or accumulated floating arithmetic. Gaussian density can underflow at large arguments. Calling a small float64 score a certificate would exceed the evidence.

Both scores should use weights decoded through the stored FP32 scale products and offsets. The exact verifier in the parent archive already treats those stored values as dyadic rationals and evaluates the recurrence with exact real beta \(1/5\). Retaining that verifier as the final gate preserves comparability. Its possible results are CERTIFIED, POINT_VIOLATION, and UNRESOLVED. A point violation concerns a state in the prescribed region; it does not establish reachability from the initial state.

An exact implementation of the local-curvature bound could use rational endpoint evaluations from `wf`. Given its established constant `CHI > 1/sqrt(2*pi)`, for rational \(z\) and any nonnegative integer \(N\),

\[
\phi(z)\le \frac{\mathrm{CHI}}{\sum_{k=0}^{N}(z^2/2)^k/k!}.
\]

This follows from the positive Taylor series for the exponential. Applying it at each endpoint and each contained critical point supplies an exact rational upper bound for local curvature. The final certificate can continue using the existing verifier, avoiding a gate change during this pilot.

## Experimental controls

The proposed four arms cross move class (single or paired) with selection score (point loss or local-curvature enclosure). Fresh seeds 445 through 468, the inherited 15% perturbation, 16 rounds, two anchors, and exactly five nonlinear fits per anchor define a bounded comparison. The initialization seed is the independent unit; inputs, time steps, and restarts within that seed do not add independent replication.

Keep the inherited point-loss nonlinear fitter in every arm. The changed procedure can then be described as candidate ranking, selection after fitting, and acceptance under a different score. Optimizing the enclosure during continuous fitting would add a second intervention. Include the no-op plus four non-no-op candidates at every anchor, using deterministic tie handling and a specified policy if the candidate set is small. Retain the stored incumbent as a possible winner. Report both fit calls and actual function evaluations because the shared iteration cap does not ensure equal realized work.

Use the same endpoint grid for each arm's selection score. The fitter can keep its inherited nine-point grid. Decode every profile candidate before ranking, and decode each nonlinear fit before deciding acceptance. If the initial fit is always accepted, exclude round zero from any monotonic-selection-score claim. A score-conditioned acceptance at round zero is also coherent if fixed in advance.

Within each move class, compare exact outcomes using matched seeds. Publish the discordant counts alongside the marginal certificate totals, and account for the two prespecified comparisons if making statistical claims. All results remain PILOT_ONLY at this sample size. Report exact initial and final outcomes, seed-level transitions, and any unresolved checks. A pooled collection of successful seeds across methods estimates no single method's success rate.

The selected objective changes both emphasis on the largest error and treatment of the gaps between points. This four-arm design measures the combined scoring intervention. A dense-grid maximum-error diagnostic can describe that mechanism; attributing the effect solely to curvature would require another controlled comparison.

## Review status

Reviewed parent `audit.py`, `rational.py`, `repair13.py`, and `restart13.py`. The existing certificate uses valid global second-derivative and affine-ReLU tail enclosures, subject to its stated exact-real semantics. The numerical local-curvature formula is valid as an analytic enclosure; arithmetic implementation and indexing still need a source review. No fresh seeds were run during this review.
