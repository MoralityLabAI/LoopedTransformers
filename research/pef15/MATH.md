# PEF-15 | Group-level conformal coverage and a finite-population bound

PEF-14's 90%-labeled Gaussian intervals used the dispersion of fit-context *forecast contributions* to estimate uncertainty on the mean accuracy across new contexts. Their actual coverage was 20/48 coalition and 25/48 threshold conditions. PEF-15 changes the statistical target and reserves separate labeled calibration contexts before examining fresh heldout accuracy.

## A. Context is the atomic observation

For a frozen model, fit-only ablation direction, ridge readout, lattice step and source context `c`, let

\[
R(c)=\frac1{16}\sum_{k=0}^{15}\mathbf1\{\hat y(c,k)=y(c,k)\}\in\{0,1/16,\ldots,1\}.
\]

All sixteen focal-bit masks are counterfactuals from the **same** context. They are dependent and jointly define one group response `R(c)`. A quantization point predictor `p\in[0,1]` is fixed using only the 32 fit contexts; its construction can involve learned models, interactions and nonlinear feature selection, provided none accesses calibration or test responses.

## B. Split-conformal interval for one unseen context

Given `n=16` labeled *calibration context families* and fixed `p`, use absolute nonconformity scores

\[
S_i=|R(C_i)-p|,\qquad i=1,\ldots,n.
\]

For target error probability `\alpha=0.1`, the conformal order statistic rank is

\[
k=\lceil (n+1)(1-\alpha)\rceil=16.
\]

Put `q=S_{(16)}`, the largest calibration residual, and predict the next context's accuracy using

\[
I=[\max(0,p-q),\min(1,p+q)].
\]

**Finite-sample coverage theorem.** Assume `S_1,\ldots,S_{16},S_* ` are exchangeable conditional on all fit-only information. For distinct scores, the test score has a uniform rank among 17 positions. The interval misses only when the new score is larger than all calibration scores, an event of probability `1/17`. With ties and the interval's inclusive endpoints, the miss probability can only decrease. Therefore

\[
\Pr\{R(C_*)\in I\mid\mathrm{fit\ information}\}\ge16/17\approx94.12\%.
\]

This is a **marginal one-context** statement. It allows arbitrary dependence between the 16 focal masks *within each context*. It gives neither simultaneous coverage of 48 next contexts nor a conditional guarantee for a particular context identity. Sharing the same fit model across several conditions makes their coverage outcomes correlated and cannot be interpreted as 48 x 16 independent experimental replications.

If the original 256 contexts are treated as a fixed finite population rather than iid observations, a uniformly random role assignment of the 64 remaining contexts to 16 calibration and 48 evaluation groups also provides exchangeable ranks for a uniformly randomly chosen evaluation context. The specific fixed SHA256 salt is a reproducibility device, and formal uniform-random-assignment guarantees depend on a randomness/exchangeability assumption about the selection process. Cryptographic hashing alone is not a mathematical proof of the statistical premise.

## C. Bound for the **average** accuracy across 48 evaluation contexts

The target changes when the goal is the average

\[
\overline R_{\rm held}=\frac1{48}\sum_{c\in\rm held}R(c).
\]

Condition on a fixed finite population of `N=64` bounded context accuracies. Suppose `n=16` calibration contexts are selected uniformly without replacement and the other 48 form the evaluation set. Write their means `\overline R_{\rm cal}`, `\overline R_{\rm held}`, and the full-population mean `\mu`.

The accounting identity is

\[
\overline R_{\rm cal}-\overline R_{\rm held}
 =\frac{64}{48}(\overline R_{\rm cal}-\mu).
\]

Serfling's bounded without-replacement tail inequality gives, for `t>0`,

\[
\Pr\{|\overline R_{\rm cal}-\mu|>t\}
 \le 2\exp\left(-\frac{2nt^2}{1-(n-1)/N}\right).
\]

Thus a conservative `1-\alpha` interval for the 48-group **heldout average** is centered at the calibration average with radius

\[
r_{\rm agg}
=\frac{64}{48}
\sqrt{\frac{(1-15/64)\ln(2/\alpha)}{2\cdot16}}
\approx 0.3569631
\quad(\alpha=0.10).
\]

Intersect `[\overline R_{\rm cal}-r_{\rm agg},\overline R_{\rm cal}+r_{\rm agg}]` with `[0,1]`. This interval is deliberately conservative. It has different units and a different coverage target from the conformal **individual-context** interval above. Reporting both tests whether the earlier normal calculation was too narrow without confusing individual group uncertainty with heldout-mean uncertainty.

## D. Phase mixture remains a fixed point predictor

For each mask k and 32 fit contexts, preserve the full joint clean-margin/directional-Jacobian triple `A_{c,k}=(f,J,y)` and phase-error coordinate `E_{c,k}`. The cross-context phase estimator averages `\psi(A_{c,k},E_{c',k})` over `c'\ne c`; with pseudo-context count \(\lambda=8\), the frozen mixture is

\[
\hat p=\frac{31}{39}\hat p_{\rm cross}
       +\frac8{39}\hat p_{\rm uniform}.
\]

Since `p` is fully fit before the 16 calibration contexts are accessed, arbitrary bias in this point estimator does **not** invalidate split-conformal marginal coverage if the required context exchangeability holds. Its bias and precision still affect interval *width* and usefulness, which are reported separately. The auxiliary Serfling interval uses calibration accuracy directly and is valid regardless of the fit-only point estimator.

## E. What this experiment can and cannot establish

PEF-15 trains three fresh initialization seeds and changes the 256-context assignment. It tests repeatability within the existing two supervised Boolean tasks. A successful observed group coverage proportion on one model does not guarantee calibration for new Boolean operators or populations with nonexchangeable context structure. If the conformal intervals cover while their mean width approaches 1, the result is statistically defensible but operationally weak; the preregistered width cap makes that limitation explicit.