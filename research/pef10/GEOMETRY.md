# PEF-10 | Rank-one ablation angle and recoverability geometry

This mathematical note is derived independently of PEF-10 heldout scores. Its statements hold for Euclidean representations and do not require a transformer model.

## Exact rotation identity

Let `g,u` be orthonormal feature directions and `r` orthogonal to both. Write a centered hidden state `x = a g + b u + r`. For signed angle `theta` write `v = (cos theta) g + (sin theta) u` and `w = -(sin theta) g + (cos theta) u`. Then the ungated rank-one ablation is

\[
 P_{v^\perp}x = x - \langle x,v\rangle v
                  = (-a\sin\theta+b\cos\theta)w+r.
\]

The squared Euclidean norm in the retained 2D direction is

\[
 q_\theta^2 = a^2\sin^2\theta+b^2\cos^2\theta
                -2ab\sin\theta\cos\theta.
\]

For the **natural probe covariance** and the true first PCA eigenvector `g`, any vector `u` orthogonal to `g` satisfies `E[ab]=0` up to finite-precision eigensolver accuracy. Hence

\[
 E[q_\theta^2] = \lambda_1\sin^2\theta + \lambda_u\cos^2\theta.
\]

The original leading-PC coefficient re-enters a retained direction at **order theta in amplitude** and **order theta squared in energy**. A near-PC perturbation can therefore preserve linearly decodable information while removing nearly all original PC energy.

## A discontinuous recoverability example

Suppose `b=0` and the binary label is the sign of `a`. At zero angle the ablated state is `r`; if `r` is label-independent, the label information has been erased. For every nonzero `theta` with `sin(theta) != 0`, `-a sin(theta)` remains in the retained direction `w`; a linear decoder with weight proportional to `-1/sin(theta)` exactly recovers the original sign. The decoder norm diverges as the angle approaches zero.

Thus exact, noiseless linear recoverability can change discontinuously at a **single direction** even when the ablated representation varies continuously with angle. For a finite regularized ridge decoder, scaling and correlations among the remaining features govern the practical transition. PEF-07 ridge fitting standardizes individual original-coordinate features with a floor, so its measured transition includes that specific conditioning rule.

## A diagnostic signal-to-residual-noise ratio

When the residual tangent coefficient `b` contains variance `sigma_b²` uncorrelated with the label-bearing component `a`, the retained signal-to-residual-noise variance ratio is proportional to

\[
 {\lambda_1\sin^2\theta \over \sigma_b^2\cos^2\theta}
 = {\lambda_1\over\sigma_b^2}\tan^2\theta.
\]

This is a **two-component diagnostic**, not a general guarantee of classification accuracy. Tangents chosen from pairwise mixed activation features can carry greater residual variance, or label-conflicting structure, than a random orthogonal tangent. Under those conditions, the random tangent can recover a PCA-dominated label at a smaller angle. The PEF-10 controls assess that possibility directly.

## Model implications

A drop in frozen-head accuracy after projection only establishes that the original head no longer uses that representation successfully. A fresh probe-fitted ridge reading a slightly rotated representation can recover because the retained subspace has changed, and because the new head can rescale a weak direction. Recovery should be examined under finite precision, an explicit noise floor and cross-task shifts before treating small-angle effects as evidence for an identifiable causal feature.
