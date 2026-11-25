# ROC-06A journal note: gradient energy, signed displacement, and causal sufficiency

**Written after seeing the original six ROC-06A numerical intervention outputs, 25 November 2026. Retrospective interpretation only.** The experimental criterion was frozen first at `1c209f6`, and this note changes no measurement, model, rank, radius or comparator.

## Exact train-only optimization solved by the gradient observer

Let the native frozen recurrent four-step suffix and fixed query readout produce eight logits (z(h,q)). For train source rule (f_i) and query (q_i), define its legal swapped-rule answer (y'_i=f_i((q_i+4)mod8)) and old answer (y_i=f_i(q_i)). Let (g_i=
abla_h[z_{y'_i}(h_i,q_i)-z_{y_i}(h_i,q_i)]) at the *unchanged* training midpoint (h_i). Normalize each nonzero gradient (u_i=g_i/|g_i|_2), center (v_i=u_i-overline u), and form (C=(1/n)sum_i v_iv_i^{mathsf T}).

For any rank-k orthogonal projector (P=UU^{mathsf T}) with (U^{mathsf T}U=I_k), define the captured gradient-variation energy

[
E(P)=rac1nsum_{i=1}^n|Pv_i|_2^2
      =operatorname{tr}(PC)
      =operatorname{tr}(U^{mathsf T}CU).
]

By the finite-dimensional Rayleigh-Ritz/Ky Fan variational principle, the maximum over all rank-k orthogonal projectors is the sum of the top k eigenvalues of (C), attained by the associated eigenvectors. ROC-06A computes exactly this candidate through the top seven right singular vectors of the centered 1024×128 matrix of normalized training gradients. This derivation gives a precise *train-distribution objective* for the observer. It does not assert that its span is a closed quotient of the nonlinear recurrence.

## Why captured gradient energy is insufficient for causal source steering

For a valid early rule-table counterfactual let (delta_i=h_i^{mathrm{edited}}-h_i) denote its actual midpoint displacement. The induced rank-k hidden edit at radius (r) is (rPdelta_i/|Pdelta_i|_2), when that projection is nonzero. In the small-radius regime, the changed-target logit margin satisfies

[
G_i(P,r)
 approx r,rac{g_i^{mathsf T}Pdelta_i}{|Pdelta_i|_2}
 quad	ext{and}quad
 |G_i(P,r)|lesssim r,|Pg_i|_2.
]

The second expression follows by Cauchy-Schwarz. Gradient capture controls a magnitude bound and omits the **sign and orientation** of the actual source-induced displacement. The ROC-06A basis also optimizes centered variation, whereas the signed causal margin depends on the uncentered gradient, the distribution of (delta_i), nonlinear suffix curvature, and the radius-dependent interpolation path. Its training loss therefore does not maximize the registered signed source-edit effect.

A source-aware objective for future registration could instead measure train-only *gradient-displacement alignment* such as

[
J(P)=rac1nsum_i
 rac{g_i^{mathsf T}Pdelta_i}{sqrt{delta_i^{mathsf T}Pdelta_i+arepsilon}}
]

or a simpler symmetric bilinear surrogate
(operatorname{tr}(PM)) with
(M=rac1{2n}sum_i(g_idelta_i^{mathsf T}+delta_i g_i^{mathsf T})).
The latter admits a Rayleigh-Ritz top-eigenspace characterization for rank-k projectors, although negative eigenvalues and finite-radius normalization call for explicit regularization and train/dev separation. **These candidates are post-hoc hypotheses, not additional ROC-06A arms.**

## Provenance and inference

ROC-06A compares the registered gradient projector with a train-only supervised label-covariance projector and a continuous input edit at the same hidden L2 radius, on all 256 already exposed ROC-05A dev examples per frozen model. The softened input is off the discrete source-rule manifold and has no independent symbolic target. Full legal edited-rule task correctness is scored separately. The original ROC-05A source and checkpoint hashes remain fixed, and the ROC-05 384-rule prospective test and original ROC-02 family-6/7 test cases remain untouched.

The sign-flip analysis uses six original model seeds as the inference units. The registered primary is the gradient observer's per-seed mean gain minus the *larger* of the supervised and softened-input mean gains. The four-step VJP was separately checked against autograd and finite differences before experiment jobs started.
