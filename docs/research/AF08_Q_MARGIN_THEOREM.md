# AF-08: exact carried-code margin certificate for protected subgoals

This is a **post-registration mathematical analysis**, distinct from AF-08's frozen scientific loss and gates. It applies to AF-05/06/07/08's current hash-pinned GraphLooper. The model's atom-code decoder is affine; its subgoal-status head is a two-class affine map.

## Statement

Let the actual carried code of a chosen Q atom after recurrence step t be z in R^d (d = 2 in AF-08). Let

$$
D(z) = W_D z + b_D,\qquad R(h) = W_R h+b_R \in \mathbb R^2.
$$

For the true Boolean class y, define the clean *signed logit margin*

$$
m_y(z) = R(D(z))_y-R(D(z))_{1-y}.
$$

After an intervention on the other subgoal, let the protected Q's actual carried code be z' = z + delta. Set

$$
K = \left\|W_D^\top(W_R[y,:]-W_R[1-y,:])^\top\right\|_2.
$$

**Exact code-margin transport lemma.** For every state, donor, graph, recurrence depth and patch whose protected Q readout uses this same decoder/head,

$$
|m_y(z')-m_y(z)| \le K\|\delta\|_2.
$$

In particular, if the clean state predicts y with a strictly positive margin and

$$
m_y(z) > K\|\delta\|_2,
$$

then the protected Q is guaranteed to remain correctly classified after the intervention. When K = 0, the status logit difference is constant in this code; the same conclusion follows from a positive clean margin.

### Proof

Write v = W_R[y,:]-W_R[1-y,:]. Because D and R are affine,

$$
m_y(z+\delta)-m_y(z) = v W_D\delta.
$$

Cauchy-Schwarz gives |v W_D delta| <= ||W_D^T v^T||_2 ||delta||_2 = K||delta||_2. The sufficient condition leaves the post-intervention signed margin strictly positive, which forces the Boolean argmax to remain y. QED.

The statement is exact for the **actual final carried code**, even if every preceding TypedLoop step is nonlinear. It does not assume global reversibility or an invertible recurrent cell.

## Diagnostic consequences

AF-07 showed near-zero protected-code displacement alongside approximately chance protected-Q accuracy in one seed. This lemma separates two possible failure types.

A clean Q may have incorrect or near-zero signed semantic margin. In that case, small code drift says little about correctness, since a stable wrong answer remains wrong. When the clean Q is confidently correct, the product of the exact readout sensitivity K and the actual code displacement supplies a deterministic sufficiency test for preserved classification.

The certificate does **not** assert that the patched goal is correct. Goal validity still depends on the downstream Horn graph and the addressed Q transplant. It also says nothing about subgoal truth when the clean prediction is already wrong.

A per-case audit should retain, for each downstream step and selected Q: recipient truth y, clean true-class logit margin, protected code L2 displacement, exact head/decoder sensitivity K, whether the inequality certifies retained correctness, and the observed post-patch correctness. Any successful certificate with an incorrect protected prediction indicates an engineering, alignment or trace-replay error.

## Mathematical training follow-up

The theorem suggests a future *margin-qualified locality* objective that penalizes violation of the sufficient bound instead of raw code drift alone:

$$
L_{\rm margin} = \mathbb E\,\left[\max(0,\ \tau+K\|z'_{\bar j}-z_{\bar j}\|_2-m_{y_{\bar j}}(z_{\bar j}))^2\right],
$$

with tau > 0 chosen in a **new preregistration**. The clean margin term prevents the constant-code failure mode whenever the status head cannot classify the true class. A separate addressed-patch Horn-oracle loss would preserve the desired change in the selected Q and the downstream goal.

This is a candidate AF-09 hypothesis, not an amendment to AF-08. AF-08's original two numerical margins were already inactive on sampled qualified AF-06 train/dev states, making an exact status-logit margin a more targeted constraint.

## Provenance and limits

- [AF-07 negative-result report](AF07_RESULTS.md) and its original verified CI/release.
- [AF-08 preregistration](../../experiments/activation_funnel_af08/PROTOCOL.json).
- [AF-08 engineering repair](AF08_ENGINEERING.md): initial nonfinite Euclidean-norm gradients, followed by a numerical-stability repair.
- Only the synthetic Horn proof-state representation is covered. Lean's closed Boolean certificates check the independent reference oracle, not the neural model's theorem-proving ability.
