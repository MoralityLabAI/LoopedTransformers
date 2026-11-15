# PEF-12 | Mathematics of frozen local-linear noise forecasts

Let \`f(h)\` be a frozen scalar classifier margin obtained by replaying visits 9–12 from a recurrent h8 state, and let a unit perturbation vector be \`u\`. With label \`y\` in {0,1}, put \`t=2y-1\`, \`m=t f(h)\` and \`j=∇f(h)·u\`. For independent \`Z~N(0,1)\` and Gaussian perturbation \`sigma Z u\`, the signed first-order margin is \`m+sigma*t*j*Z\`. Its class-correct probability is **exactly**

\`P(correct linearized | h) = Φ(m/(sigma*abs(j)))\`, for \`j≠0\`.

The sign of \`j\` disappears by Gaussian symmetry. For \`j=0\`, the probability is 1, 0, or 1/2 according to the sign of \`m\`. Average this expression over the 768 probe examples to obtain a *probe-distribution forecast* for the mean correctness of previously frozen heldout predictions. The expression contains no fitted noise coefficient.

**Curvature bound.** Suppose \`f\` is twice directionally differentiable along \`h+s*u\` for \`|s|≤sigma R\` with absolute second directional derivative bounded by H. By Taylor's theorem, for \`|z|≤R\`:

\`|f(h+sigma*z*u)-f(h)-sigma*z*j| ≤ H*sigma²*z²/2 ≤ H*sigma²*R²/2\`.

The true nonlinear and linearized classifications can disagree only if the magnitude of the linearized margin is at most the Taylor remainder. Gaussian anti-concentration bounds the disagreement probability, for \`j≠0\`, by

\`min(1, H*sigma*R²/(abs(j)*sqrt(2π))) + 2*(1-Φ(R))\`.

This follows by bounding probability mass in an interval of length \`H*sigma²*R²\` under a normal density whose maximum is \`1/(sigma*abs(j)*sqrt(2π))\`, plus the two Gaussian tails beyond \`R\`. The bound is for **one fixed state**, conditional on the curvature assumption. It does not cover probe-to-heldout source differences, and it may be loose when the directional derivative is small.

**Quantization.** For a retained scalar coordinate \`q\`, the actual nearest-even quantization error is the deterministic \`e=delta*round(q/delta)-q\`. Approximating the phase by an *independent* \`Uniform[-delta/2,delta/2]\` error would give linearized correct-class probability \`clip(1/2+m/(delta*abs(j)),0,1)\`. Its accuracy relies on a uniform-phase assumption; the deterministic lattice has correlations with the source, explaining measurable departures from the approximation.

Both formulas describe *readout sensitivity* under frozen weights. This study uses a known four-focal-bit synthetic task, reused across earlier experiments, and cannot establish general-purpose or novel-operator transfer.