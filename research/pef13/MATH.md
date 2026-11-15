# PEF-13 mathematical note: deterministic phase, local curvature and context transfer

Fix a frozen network and downstream margin \(f:\mathbb R^d\to\mathbb R\). Let \(w\) be the normalized retained h8 direction, \(\mu\) the probe mean, and \(x\) the rank-one-ablated state. For positive step \(\Delta\), the deterministic nearest-even quantizer of coordinate \(q=\langle x-\mu,w\rangle\) changes the state by

\[
 d_\Delta(q)\,w, \qquad d_\Delta(q)=\Delta\operatorname{round}_{\rm even}(q/\Delta)-q.
\]

It is an ungated intervention; the same direction and lattice act on every focal-bit combination. Rounding guarantees \(|d_\Delta(q)|\leq\Delta/2\). If \(t=2y-1\) is the label sign, define signed clean margin \(m=t f(x)\) and signed directional slope \(j=t\nabla f(x)\cdot w\). The **exact-phase first-order classifier** predicts correctness from

\[
 A_\Delta(x)=\mathbf1\{m+j d_\Delta(q)>0\},
\]

where tie conventions follow the frozen two-class margin decoder. The observable probe-aggregate forecast is \(\widehat p_{\rm phase}=n^{-1}\sum_i A_\Delta(x_i)\). All input-specific phases, clean margins and Jacobians used here were computed without heldout samples.

## Conditional curvature theorem

Suppose \(f\) is twice continuously differentiable along \(\{x+s w:|s|\leq\Delta/2\}\) and \(|w^T\nabla^2 f(x+s w) w|\leq H\) throughout the segment. Taylor's theorem with integral remainder gives

\[
 |t f(x+d w)-(m+j d)|\leq H d^2/2\leq H\Delta^2/8.
\]

Consequently if \(|m+j d|>H\Delta^2/8\), the deterministic quantized nonlinear classifier and exact-phase first-order classifier agree on this input. For a finite dataset with such a uniform \(H\), the difference in their correct-class rates is bounded by the fraction of samples satisfying \(|m_i+j_i d_i|\leq H\Delta^2/8\). The same result holds for a sample-dependent curvature bound if its value is placed inside the indicator. The model's actual \(H\) was not certified, so the experiment treats this as a conditional theorem.

## Uniform-phase integration identity

If \(d\sim \mathrm{Uniform}[-\Delta/2,\Delta/2]\) **independently of** signed margin \(m\) and slope \(j\), the conditional probability that the *linearized* margin is positive is

\[
 \Pr[m+j d>0\mid m,j]
   =\operatorname{clip}\!\left(\frac12+\frac{m}{\Delta|j|},0,1\right)
\]

for \(j\neq0\), with the constant-sign limit at \(j=0\). Independence is an assumption about the joint distribution, not a property guaranteed by deterministic quantization. The PEF-12 uniform-phase predictor applies this conditional formula and then averages over probe margins and slopes. In PEF-13, using true probe phases does not consistently improve heldout-context forecasts because the empirical phase/margin/slope joint distribution can change between the two context-disjoint samples.

## Exact finite-data error decomposition

Let \(a_P\) be the phase-linearized correctness average on *probe* histories, \(b_P\) the actual nonlinear quantized correctness average on those same probe histories, \(a_H\) the phase-linearized correctness average on heldout histories (an oracle diagnostic that is **unavailable** to the prospective method), and \(b_H\) the actual heldout nonlinear score. From the triangle inequality:

\[
 |a_P-b_H|\leq|a_P-a_H|+|a_H-b_H|,
\quad
 |a_P-b_H|\leq|a_P-b_P|+|b_P-b_H|.
\]

The two inequalities independently decompose prediction error into within-cohort linearization and between-cohort accuracy transfer terms. They are deterministic identities/inequalities for finite sets. A small within-cohort term places no numerical bound on the between-cohort term. With continuous \((m,j,q)\), empirical total-variation distance between distinct samples is often trivially maximal; distributional assumptions or smoothed/coarse-grained features are needed to obtain useful generalization bounds from sample size alone.

The PEF-13 posthoc diagnostic found a mean within-heldout linearization difference of 0.000054 for coalition arms and mean probe-to-heldout linearization transfer difference of 0.015625. The former supports the observed accuracy of the local Taylor prediction on these finite cases. The latter motivates prospective conditional distribution forecasts.