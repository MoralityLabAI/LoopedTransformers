# AF-07: a depth-locality criterion for recurrent causal abstractions

**Status:** Mathematical interpretation of frozen AF-07 experiment. This note does not change the preregistration, trained weights, or scientific gates. See [PROTOCOL.json](../../experiments/activation_funnel_af07/PROTOCOL.json).

## The causal object

A finite Horn DAG has initial Boolean facts \(f\), two derived subgoals
\(q=(q_0,q_1)\), and a downstream goal \(g=\Gamma(q_0,q_1)\), where \(\Gamma\) is a declared conjunction or disjunction. The host evaluates Horn least-fixed-point closure. The looper sees \(f\) and the graph structure but receives no derived truth values as encoder inputs.

For recipient and donor histories \(r,d\), choose a subgoal index \(j\) and preserve the recipient's other-subgoal leaves. The exact abstract counterfactual is

\[
q^{r\leftarrow_j d}
 =(q_j^d,q_{1-j}^r), \quad
g^{r\leftarrow_j d}=\Gamma(q_j^d,q_{1-j}^r).
\]

The independent Horn oracle checks these targets against the actual matched graph.

## Neural transport

After upstream iterations, let \(z^r=E(h^r)\) and \(z^d=E(h^d)\) be the *actual* compressed carried codes. \(\tau_j(z^r,z^d)\) copies the donor atom-code slice corresponding to \(Q_j\), leaving all other atom slices and all rule slices unchanged. The decoded state is resumed for \(t\) downstream calls:

\[
H_t^j=F_\theta^t\bigl(D(\tau_j(z^r,z^d))\bigr),
\qquad
H_t^r=F_\theta^t(D(z^r)).
\]

Here \(F_\theta\) includes the original graph-typed cell and the same encoder/decoder bottleneck after every call. The operator \(P_k\) extracts the carried code of \(Q_k\) after a downstream call. The desired *code-locality* diagram approximately commutes:

\[
P_{1-j} E(H_t^j) \simeq P_{1-j} E(H_t^r)
\quad\text{for every registered }t.
\]

A separate semantic diagram requires the output for \(Q_{1-j}\) to remain the recipient's truth, while the patched \(Q_j\) and final \(G\) agree with the Horn counterfactual. Code locality is intentionally stronger than semantic locality.

## Differentiable objective

AF-07 trains a shared status/goal loss in every arm and adds

\[
\mathcal L_{\mathrm{local}}
= 0.25\cdot \frac{1}{T}\sum_{t=1}^T
 D_{KL}\!\left(p^r_{t,1-j}\ \|\ p^j_{t,1-j}\right)
+0.12\cdot\frac{1}{Td}\sum_{t=1}^T
 \left\|z^j_{t,1-j}-\operatorname{sg}(z^r_{t,1-j})\right\|_2^2.
\]

The KL's clean distribution and the clean code are detached during the locality penalty. Exact subgoal CE supervision remains active on every intermediate step for **all** three continuation arms. The status arm patches the irrelevant atom; the ramp and locality arms have identical donor selections and addressed-patch schedules. Their native-cell *forward-call* counts match. The locality arm has additional loss computation.

This is a hypothesis about an identifiable recurrent circuit. Its failure is informative because a network may represent the same Boolean subgoal through a distributed, reparameterized or LayerNorm-gauge-dependent code. The code-MSE term measures a chosen coordinate intervention, not a coordinate-free theorem about causal abstraction.

## Operational measures

For each saved recipient/donor/selected-Q pair, store all Q logits, G logits, and the protected Q code displacement after downstream calls \(t=1,2,3\). Depth-4 and depth-5 heldouts additionally run \(t=4,5\), which are tensor-only recurrence diagnostics.

Two error functions remain separate:

\[
e^{\mathrm{sem}}_t
=\mathbf1[\widehat q_{t,1-j}\ne q_{1-j}^r],
\qquad
e^{\mathrm{code}}_t
=\|z^j_{t,1-j}-z^r_{t,1-j}\|_2.
\]

Large \(e^{\mathrm{code}}\) with small \(e^{\mathrm{sem}}\) can mean the model changes its code while retaining correct Q truth. Small \(e^{\mathrm{code}}\) with large \(e^{\mathrm{sem}}\) can expose a fragile subgoal readout. Both are retained.

The registered causal claim requires eligible seeds to improve *untouched Q semantic accuracy* at depths 4 and 5, with constraints on goal accuracy, informative counterfactual goal flips, irrelevant-patch retention and the extra recurrence horizon. A separate independent verifier recomputes every Horn target and recurrence score from stored logits; sampled Torch replays check saved numerical trajectory/codes.

## Interpretation boundary

AF-06 obtained perfect development inference of Q0/Q1 for two width-2 seeds but poor longer-depth intervention locality. AF-07 tests whether explicit recurrence-time locality supervision addresses that failure. If successful, it would establish a repeatable, structured-Horn causal abstraction at the tested sizes and graph families. Stronger applications to Lean tactics would require learning new proof states, demonstrating actual proof completion, and checking proposed proof terms in the Lean kernel.
