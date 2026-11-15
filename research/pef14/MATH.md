# PEF-14 | A context-conditioned lattice phase U-statistic

Let the context index be a random variable `C`, taking values over source contexts, and the focal mask `K ∈ {0,...,15}` be uniform. A frozen model, intervention and clean ridge readout define a scalar margin `F(C,K)` and a directional Jacobian `J(C,K)` at a shared visit. For fixed quantization step `Δ>0`, write the retained coordinate `Q(C,K)` and nearest-even phase error

\[
E_\Delta(C,K) = \Delta\operatorname{round}(Q(C,K)/\Delta)-Q(C,K),
\quad |E_\Delta|\le \Delta/2.
\]

The conditional *linearized* risk is

\[
R_\Delta=\frac1{16}\sum_{k=0}^{15}
  \mathbb E_C\left[\mathbf 1\big\{\mathbf 1[F(C,k)+J(C,k)E_\Delta(C,k)\ge0]=Y(C,k)\big\}\right].
\]

The indicator is bounded by one. This supplies exact finite-distribution bounds without a continuity assumption on the classifier.

## 1. Dependence-error bound

Define `A=(F,J,Y)`, `E=E_Δ` and, conditionally on focal mask k, let `P_k` be their joint distribution. Let `Q_k = P_{A|k} ⊗ P_{E|k}` be the product of their marginal laws. Put `ψ(A,E)=1{1[F+JE≥0]=Y}`. Then

\[
\left|R_\Delta-R_\Delta^{\rm product}\right|
\le\frac1{16}\sum_{k=0}^{15}\mathrm{TV}(P_k,Q_k),
\]

where TV is defined as the supremum of `|P(B)-Q(B)|` over events. **Proof:** for each k, `ψ` is a [0,1]-valued measurable function; the absolute difference of its expectations is bounded by TV. Average the sixteen inequalities. The bound is sharp: for binary `A=E` equally likely zero or one, `ψ=1[A=E]`, the joint mean equals 1, the product mean equals 1/2 and TV equals 1/2.

As a consequence, separating phase from clean margin and directional slope is a statistical assumption. Sampling many phase errors from *different* contexts removes same-context dependence, which can help only when this dependence is small or noisy relative to finite sample error.

## 2. Off-diagonal estimator and finite-sample concentration

Given n complete context families `Z_1,...,Z_n`, define `A_{c,k}` and `E_{c,k}` for every focal mask. The phase-shuffled conditional U-statistic is

\[
\widehat R_{\rm cross}=\frac{1}{16n(n-1)}
\sum_{k=0}^{15}\sum_{c\ne c'}\psi(A_{c,k},E_{c',k}).
\]

Under the explicit **iid context-family assumption**, distinct contexts c and c' are independent. Therefore `E[\hat R_cross]=R_product`. The sixteen focal masks within each context can remain arbitrarily dependent. Replacing one context family affects at most `2(n-1)` ordered pairs per mask, each contributing at most 1, hence changes the statistic by at most `2/n`. McDiarmid's bounded-difference inequality gives

\[
P(|\widehat R_{\rm cross}-R_{\rm product}|\ge t)
\le 2\exp(-nt^2/2).
\]

This is a general, nonasymptotic but conservative bound. The actual experiment enumerates a *finite set of 256 contexts without replacement* using a fixed cryptographic permutation. The iid assumption is an idealization for this concentration theorem. The completed experiment's coverage audit is empirical and separate.

## 3. Frozen shrinkage and bias

The registered mixture uses n=48 context families and pseudocount λ=8:

\[
\widehat R_{\rm mix}
=\underbrace{\frac{n-1}{n-1+\lambda}}_{47/55}\widehat R_{\rm cross}
+\underbrace{\frac\lambda{n-1+\lambda}}_{8/55}\widehat R_{\rm unif}.
\]

Here the uniform model replaces phase E by an independent uniform `U[-Δ/2,Δ/2]`. At the population level, convexity yields

\[
|R-\mathbb E\widehat R_{\rm mix}|
\le\frac{47}{55}\cdot\frac1{16}\sum_k \mathrm{TV}(P_k,Q_k)
+\frac8{55}\cdot\frac1{16}\sum_k
\mathrm{TV}(P_k,P_{A|k}\otimes U_\Delta).
\]

Under iid context-family sampling, replacing one family alters `\hat R_cross` by at most `2/n` and the uniform sample average by at most `1/n`. Thus the mixture's bounded-difference constant is `[(2·47+8)/55]/n`, giving

\[
P(|\widehat R_{\rm mix}-\mathbb E\widehat R_{\rm mix}|\ge t)
\le 2\exp\left(-\frac{2nt^2}{(102/55)^2}\right).
\]

This bound controls sampling variation under the stated assumptions, while the preceding TV inequality controls structural bias. Neither implies a 90% predictive coverage statement for heldout finite-population contexts. The registered normal intervals, which covered 45 of 96 conditions, empirically show a major failure of their simple variance model.

## 4. Exact phase error and nonlinear remainder

For differentiable downstream logit f at the ablated h8 state x and retained unit vector w, set `d=Δ round(q/Δ)-q`. If the second directional derivative satisfies `|D_w² f(x+s w)|≤H` for all s between 0 and d, Taylor yields

\[
|f(x+dw)-[f(x)+d\,D_wf(x)]|\le H d^2/2\le H\Delta^2/8.
\]

A linearized classifier and the actual recurrent classifier can differ only when `|f(x)+d D_w f(x)|≤HΔ²/8`. Even without estimating H, the posthoc PEF-14 same-input replay measures this approximation's empirical impact: average absolute accuracy discrepancy 0.030 percentage points for coalition and 0.014 for threshold. The much larger probe-to-heldout calibration discrepancy therefore exposes distributional transfer as the dominant source of error in this finite experiment.

## Implication for future experiments

A future predictor should track *context-family* variance and dependence between (margin, directional slope, lattice phase), perhaps via a registered hierarchical or conformal calibration method that reserves a fresh calibration split. Any new interval procedure must be preregistered and evaluated on new contexts. The eight-pseudocount model is a proof-friendly baseline for that work, not a uniquely optimal estimator.
