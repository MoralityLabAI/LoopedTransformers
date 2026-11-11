# SEM-ECHO-01: semantic recovery across recurrent computation

**Version:** 0.1, development note  
**Date:** 2026-11-11  
**Companion:** [experimental protocol](protocol.md)  
**Status:** mathematical derivations and experimental designs. Neural results require checkpoint receipts.

The central question is whether a recurrent model preserves the state needed to continue a computation while it compresses other information. Time reversal supplies useful mathematics for this question. The experiments need an explicit definition of semantics, a complete account of the recurrent state, and a declared observation or noise model.

The existing Q4 and Q5 outcomes for 24 toy checkpoints have already been examined. Work on those checkpoints belongs to retrospective development. Fresh graph instances test generalization conditional on those models. A prospective test of a new model-level prediction requires fresh training seeds and a frozen rule. The existing [LOOPSCOPE-01 registration](../../../docs/loopscope01_registration.md) retains its question numbering and decision rules.

Sai Karthik Reddy Koralla's referenced IPFS paper was unavailable when this note was written. Attribution to that paper awaits its text. The operational definitions below belong to this development protocol.

## 1. What transfers from time-reversal physics

For a closed quantum system with Hamiltonian $H$,

$$
U(t)=e^{-iHt/\hbar},\qquad U(t)^{-1}=U(-t).
$$

This equation states invertibility of unitary evolution. Time-reversal symmetry additionally concerns an antiunitary operator $\Theta$ and the transformed dynamics. Under the appropriate symmetry condition,

$$
\Theta U(t)\Theta^{-1}=U(-t).
$$

CPT concerns a combined transformation under its quantum-field-theory assumptions. Experiments in neutral B mesons have observed T violation. These are separate mathematical claims with different premises. [Greaves and Thomas](https://arxiv.org/abs/1204.4674) give a rigorous CPT treatment, while the [BABAR measurement](https://arxiv.org/abs/1207.5832) supplies a direct experimental source for T violation.

A recurrent model advances a computational index $t$. The proposed analogy concerns its state transition, the observables retained by that transition, and the conditions under which a preceding semantic state can be recovered. Assigning a particle identity to semantic gain would require additional physical claims.

A Loschmidt echo compares forward evolution with an attempted reversal:

$$
M(t)=
\left|
\langle\psi_0|
e^{+i(H+\delta H)t/\hbar}
e^{-iHt/\hbar}
|\psi_0\rangle
\right|^2.
$$

The perturbation $\delta H$ makes this a probe of reversibility under imperfect control. [Gorin et al.](https://arxiv.org/abs/quant-ph/0607050) develop the relevant fidelity-decay framework. For a neural recurrence $F$, a corresponding experiment can compare $G^kF^k(h)$ with $h$, where $G$ is a specified reversal procedure. Raw reconstruction error and semantic recovery receive separate scores. The quality and information access of $G$ belong in the experimental specification.

For a real neural Jacobian $J_t$, perturbations and gradients obey

$$
\delta h_{t+1}=J_t\delta h_t,
\qquad
g_t=J_t^{\mathsf T}g_{t+1}.
$$

An inverse tangent step uses $J_t^{-1}$ when that inverse exists. The adjoint equals the inverse for an orthogonal square Jacobian. TCA transports readout gradients through the adjoint, so its backward pass measures sensitivity. An echo requires a separately specified reversal. The toy input LayerNorm has exact shift-null directions, which make its full ambient Jacobian singular.

## 2. Full state, semantic state, and usable information

Write the neural transition as

$$
h_{t+1}=F_x(h_t).
$$

Here $x$ includes every fixed input read again during the loop. The toy model re-injects the prompt at each iteration. Its complete dynamic state is the tensor at every position after the core blocks. For a model with recurrent caches, those caches also belong to $h_t$. An externally supplied loop counter must be accounted for when defining the transition.

Let $q_x(h)$ describe the task state, such as the current vertex of a pointer chase. A proposed semantic model has a transition $T_x$ satisfying

$$
q_x(F_x(h))=T_x(q_x(h)).
\tag{1}
$$

Equation (1) is a claim about causal continuation. A probe's accuracy on observed states supplies one kind of evidence. Interventions test whether the decoded state has the proposed role in the subsequent computation. [Geiger et al.](https://arxiv.org/abs/2106.02997) formulate this approach through causal abstraction and interchange interventions.

The current toy architecture contains an exact example of disposable coordinates. For each position $p$, add a constant $c_p$ to every feature:

$$
h'_p=h_p+c_p\mathbf1.
$$

LayerNorm subtracts the feature mean and has the same variance after this shift. Consequently $\operatorname{inject}(h')=\operatorname{inject}(h)$ in exact arithmetic, so $F_x(h')=F_x(h)$. The cursor's LayerNorm also makes its current own-head readout invariant. This many-to-one map discards a known nuisance direction while preserving the declared readout and future recurrence. Finite-precision replay measures its numerical tolerance.

For a random semantic target $S$, deterministic processing of a complete state obeys the data-processing inequality. Apply that comparison at fixed input or to the augmented state $(x,h_t)$: re-reading $x$ can change the information in $h_t$ alone. Increased ease of prediction has a different definition. Fix a class $\mathcal V$ of predictors and its training budget:

$$
H_{\mathcal V}(S\mid H_t)
=
\inf_{g\in\mathcal V}
\mathbb E[-\log g(S\mid H_t)],
$$

$$
I_{\mathcal V}(H_t\to S)
=H_{\mathcal V}(S)-H_{\mathcal V}(S\mid H_t).
$$

Computing more loops can increase this usable information by producing a representation that the fixed observer can read more easily. [Xu et al.](https://arxiv.org/abs/2002.10689) provide the predictive $\mathcal V$-information framework. A valid comparison keeps the observer class, data split, and fitting budget fixed. SEM-ECHO-01 v0 starts with the model's own head; a learned probe would be a later observer with its own protocol.

## 3. An exact classical recovery identity

Consider a finite stochastic chain

$$
S\longrightarrow H\longrightarrow Z.
$$

$H$ is the retained state before a proposed compression or observation channel, and $Z$ is its output. Define the semantic information loss in nats by

$$
\Delta
=I(S;H)-I(S;Z)
=I(S;H\mid Z).
\tag{2}
$$

Use the Bayesian reverse channel

$$
R(h\mid z)=P(h\mid z)
$$

and form the reconstructed joint distribution

$$
\widetilde P(s,h,z)=P(s,z)P(h\mid z).
$$

Then

$$
D_{\mathrm{KL}}\!\left(P_{SHZ}\Vert\widetilde P_{SHZ}\right)
=\Delta.
\tag{3}
$$

**Proof.** The logarithm of the density ratio in (3) is
$\log[P(s,h\mid z)/(P(s\mid z)P(h\mid z))]$. Its expectation is conditional mutual information. The chain rule and the Markov premise give (2).

Marginalization contracts KL divergence, so the recovered joint over $(S,H)$ has KL error at most $\Delta$. Pinsker's inequality gives total-variation error at most $\sqrt{\Delta/2}$. Exact recovery of semantic correlations follows when $\Delta=0$.

These statements concern distributions. Several original states can remain compatible with one observed $z$. A reverse channel recovers their conditional distribution; a particular sample's original history requires sufficient side information. [Junge et al.](https://arxiv.org/abs/1509.07127) develop a related, stronger quantum recovery framework. Equations (2) and (3) are the classical identity used here.

For experiments, a finite simulator permits exact probabilities and a fully specified reverse channel. Estimating mutual information between high-dimensional continuous activations would introduce a separate statistical problem. Decoder failure can arise from limited decoder capacity, finite data, or the channel itself, so those causes need distinct controls.

## 4. The permutation task gives a precise semantic quotient

The fixed-depth toy dataset samples a permutation $\pi$ on the node set. Its oracle pointer is

$$
s_t=\pi^t(s_0).
$$

Fix the graph and the complete prompt. Let $D_x$ be a domain invariant under $F_x$, containing every state used by the claim. Suppose

$$
q_x(F_x(h))=\pi(q_x(h))
\qquad(h\in D_x).
\tag{4}
$$

The following are elementary consequences of (4).

### All-depth transport

$$
q_x(F_x^k(h))=\pi^k(q_x(h))
\qquad(k\ge0).
\tag{5}
$$

**Proof.** The case $k=0$ is immediate. Apply (4) to $F_x^k(h)$, which remains in $D_x$, and use the induction hypothesis.

This is an all-depth theorem conditional on the invariant-domain premise. Checking a finite sample of trajectories estimates violations of that premise on the sample.

### Semantic invertibility

Since $\pi$ is a bijection,

$$
q_x(h)=\pi^{-1}(q_x(F_x(h))),
$$

$$
q_x(F_x(h))=q_x(F_x(h'))
\quad\Longleftrightarrow\quad
q_x(h)=q_x(h').
\tag{6}
$$

**Proof.** Substitute (4) and cancel the injective map $\pi$.

Every exact collision $F_x(h)=F_x(h')$ therefore lies within one semantic fiber. Neural coordinates can disappear while the quotient remains invertible.

### A finite witness against semantic sufficiency

If $q_x(h)=q_x(h')$, (5) predicts

$$
q_x(F_x^k(h))=q_x(F_x^k(h'))
\quad\text{at every }k.
\tag{7}
$$

Two genuine visits to the same oracle vertex supply candidate donors. A future disagreement gives a finite counterexample to sufficiency of the tested decoder on those states, assuming the full state and the fixed inputs were accounted for. Agreement alone also permits both continuations to be wrong, so oracle agreement receives a companion metric.

### Period constraint

If $F_x^m(h)=h$, then $\pi^m(q_x(h))=q_x(h)$. Thus the semantic orbit length divides the latent period.

**Proof.** Apply $q_x$ to the periodicity equation and use (5).

A latent fixed point can carry a semantic fixed point. A task that advances around a longer semantic cycle requires the corresponding periodic structure somewhere in its complete state.

## 5. Koopman phase supplies a recurrence residual

For a semantic cycle $v_0,\ldots,v_{\ell-1}$, ordered so that
$\pi(v_j)=v_{j+1\bmod\ell}$, define

$$
z(v_j)=e^{2\pi i j/\ell},
\qquad
\omega=e^{2\pi i/\ell}.
$$

The Koopman operator acts on observables by $Uf=f\circ\pi$. Hence

$$
Uz=\omega z.
\tag{8}
$$

On a finite node set with the uniform measure, $U_\pi$ is unitary: permutation of the summation index preserves the inner product, and $U_\pi^{-1}=U_{\pi^{-1}}$. Define the pullback $q_x^*f=f\circ q_x$. Equation (4) gives

$$
U_{F_x}q_x^*=q_x^*U_\pi,
\qquad
U_{F_x}g=g\circ F_x.
$$

Both sides evaluated at $h$ equal $f(\pi(q_x(h)))$. Unitarity uses the uniform semantic inner product. The pullback identifies the corresponding neural observables while the neural transition can merge states within a semantic fiber.

Composing $z$ with $q_x$ turns (4) into the same equation on hidden states. This phase measures position around a finite symbolic orbit. Its magnitude and phase concern an observable of the state; they impose no Euclidean geometry on the complete activation tensor.

For successive own-head labels $\widehat s_t,\widehat s_{t+1}$ that lie on the oracle cycle, use

$$
r_t=
\frac{
|z(\widehat s_{t+1})-\omega z(\widehat s_t)|^2
}{
|1-\omega|^2
}.
\tag{9}
$$

For $\ell>1$, exact one-step motion has residual zero. A constant label on that cycle has residual one. This normalization controls the shrinking phase increment on long cycles. Cycles of length one are explicitly uninformative for this diagnostic. A prediction outside the oracle cycle makes that transition invalid for (9); its count and the oracle error remain visible.

A trajectory with a constant phase offset can satisfy (8) while predicting the wrong vertices. Report oracle accuracy beside the phase residual. Additional Fourier modes
$z_m(v_j)=e^{2\pi i m j/\ell}$ provide later checks, with constant and degenerate modes marked explicitly.

### A finite-horizon phase certificate

Fix one cycle with $\ell>1$, and allow a complex observation $z_t$ at each step. Set

$$
e_t=z_{t+1}-\omega z_t,
\qquad
R_T=\frac1T\sum_{t=0}^{T-1}
\frac{|e_t|^2}{|1-\omega|^2}.
$$

The exact identity

$$
z_T-\omega^Tz_0
=\sum_{t=0}^{T-1}\omega^{T-1-t}e_t
\tag{9a}
$$

gives the bound

$$
|z_T-\omega^Tz_0|
\le\sum_{t=0}^{T-1}|e_t|
\le |1-\omega|\,T\sqrt{R_T}.
\tag{9b}
$$

**Proof.** Substitute the definition of $e_t$ into the sum. Adjacent terms cancel. The triangle inequality and Cauchy-Schwarz give (9b), using $|\omega|=1$.

Distinct roots on this cycle are separated by at least $|1-\omega|$. If the initial observation equals the correct root and

$$
T\sqrt{R_T}<\tfrac12,
$$

nearest-root decoding at time $T$ is correct. For initial error
$\eta_0=|z_0-z_0^\star|$, the sufficient condition becomes

$$
\frac{\eta_0}{|1-\omega|}
+T\sqrt{R_T}<\tfrac12.
$$

This certificate applies to the observed finite trace and its known cycle. A phase observable needs a correct semantic anchor; a clock feature can advance regularly while being unrelated to the task. Causal continuation tests supply additional evidence for that relationship.

For a complete v0 trace whose hard own-head labels all lie on the chosen cycle, an incorrect one-step transition has $r_t\ge1$, since the minimum distance between distinct roots is $|1-\omega|$. Consequently $R_T<1/T$ already ensures that every observed transition follows the permutation relative to its initial label. A correct initial anchor then gives the correct whole trace. The more conservative endpoint bound in (9b) becomes useful for continuous phase observables whose values can fall between roots.

The certificate is sufficient. Residuals can cancel in (9a): with $\omega=e^{2\pi i/5}$, the trace $1,\omega^2,\omega^2$ has the correct final phase after two steps, although both one-step residuals are nonzero. The sum bound retains their magnitudes and can withhold a certificate for a correct endpoint.

## 6. Contraction and semantic margin

### A limit on global contraction

Suppose distinct semantic fibers stay at least $\delta>0$ apart:

$$
q_x(h)\ne q_x(h')\implies\|h-h'\|\ge\delta.
$$

If $F_x$ has a global Lipschitz bound $0<\rho<1$, then (4) and injectivity of $\pi$ imply

$$
\delta
\le\|F_x^t(h)-F_x^t(h')\|
\le\rho^t\|h-h'\|.
\tag{10}
$$

For a pair with different semantics, this forces

$$
t\le
\frac{\log(\|h-h'\|/\delta)}{\log(1/\rho)}.
\tag{11}
$$

**Proof.** Semantic labels remain distinct by (5). Apply the separation premise at time $t$, followed by the iterated Lipschitz bound.

The separation premise carries the robustness claim. For
$F(h)=-h/2$ and $q(h)=\operatorname{sign}(h)$, every nonzero state alternates sign forever in exact arithmetic. Its decoding margin tends to zero. Fixed noise or finite precision eventually affects the semantic readout.

The Q7 Jacobian statistic estimates sensitivity in a chosen subspace on observed trajectories. Applying (10) requires a uniform bound on the relevant complete domain and a positive cross-semantic margin.

### Invariant regions permit contraction within a semantic class

Choose code centers $c_s$ and disjoint closed balls $D_s=\overline B(c_s,r)$ in a complete normed space, with center separation greater than $2r$. Assume

$$
\|F_x(h)-c_{\pi(s)}\|
\le\rho\|h-c_s\|+\varepsilon
\quad(h\in D_s),
$$

$$
0\le\rho<1,\qquad \varepsilon\le(1-\rho)r.
\tag{12}
$$

Then $F_x(D_s)\subseteq D_{\pi(s)}$, and

$$
\|F_x^t(h)-c_{\pi^t(s)}\|
\le
\rho^t\|h-c_s\|
+\frac{\varepsilon(1-\rho^t)}{1-\rho}
\le r.
\tag{13}
$$

**Proof.** The radius condition proves one-step inclusion. Iterating the affine error recurrence yields the geometric series in (13).

Nearest-center semantics is therefore correct at every depth. If $F_x$ is also pairwise $\rho$-Lipschitz inside each ball, its $\ell$-fold iterate is a contraction on a closed orbit ball. Completeness supplies a fixed point of that iterate. Disjoint semantic balls ensure a latent cycle of length $\ell$.

An empirical fit of centers, radii, and Jacobians estimates the premises. Exhaustive checks on a finite model or validated bounds over the regions would certify them. The distinction belongs in every result receipt.

## 7. Finite-noise controls isolate conditioning

Initialize a real semantic coordinate at $u\in\{-1,+1\}$, and let the real coordinate $v$ carry nuisance information. Compare the maps on $\mathbb R^2$:

$$
F_{\mathrm{keep}}(u,v)=(u,\alpha v),
\qquad
F_{\mathrm{shrink}}(u,v)=(\alpha u,v),
\qquad0<\alpha<1.
\tag{14}
$$

Both maps have exact inverses. After $k$ forward steps, add independent terminal noise
$\eta\sim\mathcal N(0,\sigma^2I)$, then apply the exact inverse. The recovered semantic coordinate has mean-square error

$$
\operatorname{MSE}_{\mathrm{keep}}=\sigma^2,
\qquad
\operatorname{MSE}_{\mathrm{shrink}}=\sigma^2\alpha^{-2k}.
\tag{15}
$$

For a uniform semantic bit and sign decoding, the error probabilities are

$$
P_{\mathrm{err,keep}}=\Phi(-1/\sigma),
\qquad
P_{\mathrm{err,shrink}}=\Phi(-\alpha^k/\sigma).
\tag{16}
$$

**Proof.** The exact inverse multiplies the first noise coordinate by one in the first map and by $\alpha^{-k}$ in the second. The sign error is the corresponding Gaussian tail.

At $\sigma=0$, both semantic echoes are exact. Equations (15) and (16) measure inverse conditioning under a specified finite-noise channel. They provide analytic controls for interpreting an echo that degrades with depth.

Reversible residual networks offer an architectural control with a known inverse construction; see [Gomez et al.](https://arxiv.org/abs/1707.04585). Their reconstruction property supports a controlled comparison once numerical precision and the readout are fixed.

## 8. Proof-state semantics through bounded continuations

A proof state includes its local contexts, outstanding goals, metavariable assignments, and environment. Tactics construct proof terms that receive final checking. The [Lean reference](https://lean-lang.org/doc/reference/latest/Tactic-Proofs/) and its section on [proof states](https://lean-lang.org/doc/reference/latest/Tactic-Proofs/Reading-Proof-States/) specify this machinery.

Choose a finite grammar $A$ of primitive actions. Add absorbing success and error states so every action sequence has an outcome. For budget $B$, define

$$
\chi_B(s)=
\bigl(O(s,c)\bigr)_{c\in A^{\le B}},
$$

where $O$ can record open, failed, or accepted. Set
$s\equiv_B t$ when these signatures agree.

### Continuation congruence

$$
s\equiv_{B+1}t
\implies
T_a(s)\equiv_B T_a(t)
\qquad(a\in A).
\tag{17}
$$

**Proof.** For each continuation $c$ of length at most $B$, prefix it by $a$. The resulting word has length at most $B+1$, so its outcomes agree.

A compression $C$ preserves every $B$-step continuation outcome exactly when

$$
C(s)=C(t)\implies\chi_B(s)=\chi_B(t).
\tag{18}
$$

For an approximate comparison under a declared distribution $\mu_B$, use

$$
d_{B,\mu}(s,t)
=
\mathbb E_{c\sim\mu_B}
\mathbf1[O(s,c)\ne O(t,c)].
\tag{19}
$$

Sampling $m$ independent continuations for a fixed pair gives the Hoeffding radius
$\sqrt{\log(2/\beta)/(2m)}$ at confidence $1-\beta$. Theorem families and trained models remain separate units for generalization.

Adding a derived lemma can preserve logical consequences while shortening later proofs. Compare a recovered encoding with the same proof state, or a predicted successor with its expected symbolic successor. The continuation signature can change during valid progress because its budget is finite.

A saved proof log allows exact history reconstruction, as in [Bennett's reversible computation construction](https://www.cs.princeton.edu/courses/archive/fall06/cos576/papers/bennett73.html). The future proof outcomes may survive disposal of that history. Equation (18) defines the retention needed for the declared task.

## 9. A success-conditioned bridge on a finite proof graph

Let $K_t(s,s')$ be a transition kernel on a finite proof-state graph, with checked success absorbing. Fix terminal budget $B$ and event $A$ that the terminal state is successful. Define

$$
h_t(s)=P(A\mid S_t=s).
$$

Backward dynamic programming gives

$$
h_t(s)=\sum_{s'}K_t(s,s')h_{t+1}(s'),
\qquad
h_B(s)=\mathbf1_A(s).
$$

For $h_t(s)>0$, the Doob bridge is

$$
K_t^A(s,s')
=K_t(s,s')\frac{h_{t+1}(s')}{h_t(s)}.
\tag{20}
$$

**Proof.** The Bellman equation makes each row sum to one. For a fixed initial state $s_0$ with $h_0(s_0)>0$, multiplying (20) along a path telescopes the $h$ factors, producing the original path probability conditioned on success. For random initial distribution $\mu_0$, also use

$$
\mu_0^A(s)
=\frac{\mu_0(s)h_0(s)}{P(A)}.
$$

This initial tilt accounts for which starting states contribute to the successful population.

An ordinary Bayesian reverse kernel has a different definition:

$$
K_t^\leftarrow(s\mid s')
=\frac{p_t(s)K_t(s,s')}{p_{t+1}(s')},
\tag{21}
$$

on states of positive denominator. It depends on the forward population $p_t$. Equations (20) and (21) make both uses of future information explicit.

This bridge can generate paired successful forward and backward traces in a finite simulator. Exact success probabilities are privileged targets available during dataset construction. A learned predictor evaluated on held-out proof tasks receives only its declared input and budget. Receipts must account for every oracle query and the cost of producing the bridge.

## 10. Experimental consequences

The first implementation tests conditional continuation consistency with the own-head decoder. It can produce finite failures of (7) and characterize which loop depths carry them. Full-state replay checks that the implementation resumes the saved recurrence correctly.

Later experiments can fit a recovery channel and score (19), estimate accessible information under a frozen observer, or test the invariant regions in (12). Partial cursor and edge interventions require matched perturbations because their hybrid states can leave the observed state distribution.

The finite proof experiment should expose several budgets, with exact primitive action costs. The existing Q6 best-first b4 endpoint reached a ceiling across its 18 checkpoints, so it supplied little variation for the intended correlation. A Horn-logic environment with known shortest derivations can test whether compressed states retain the facts needed to finish within each budget.

A finite set of observed echoes supports an empirical statement over its declared distribution. The all-depth result is (5), conditional on a certified invariant domain and a commuting transition. Establishing those premises is a separate mathematical task.
