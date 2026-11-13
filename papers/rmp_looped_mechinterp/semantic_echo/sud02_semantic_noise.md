# SUD-02 | Reversible semantic information echo in LOOPSCOPE Q5

**Date:** 2026-11-13
**Evidence stage:** mathematical controls and executable checkpoint-gated assay. Q5's original frozen weight files remain external to GitHub.
**Related work:** [Semantic Echo](README.md); [SUD-01](https://github.com/MoralityLabAI/Compactification-Theory/pull/104); [LOOPSCOPE registration](../../../docs/loopscope01_registration.md).

## Source model and exact posterior

The Q5 fixed-depth task uses a permutation \(\pi\) of \(2^n\) nodes.
Let \(X\) be the uniformly sampled clean start and let \(Y=X\oplus N\),
where all n bits of \(N\) are independent Bernoulli(p) variables. The
clean semantic bit at depth t is

\[
S_t=f_b(\pi^t(X)),
\]

where \(f_b\) extracts binary bit b. The function \(f_b\circ\pi^t\)
is balanced because \(\pi^t\) is a bijection. At each fixed graph,
the exact posterior is

\[
q_t(y)=P(S_t=1\mid Y=y,\pi)
=\sum_x f_b(\pi^t(x)) p^{d_H(x,y)}(1-p)^{n-d_H(x,y)}.
\]

Uniform X implies uniform Y. Consequently

\[
I_t^\star=1-2^{-n}\sum_y h_2(q_t(y)),\qquad
e_t^\star=2^{-n}\sum_y\min(q_t(y),1-q_t(y)).
\]

OpenAI Math #119 (the sharp Courtade–Kumar inequality) implies

\[
\boxed{I_t^\star\le1-h_2(p),\qquad e_t^\star\ge p.}\tag{SUD2-1}
\]

The error inequality follows from binary Fano and the monotonicity of
binary entropy on [0,1/2].

## SUD2-2: Fourier spectrum predicts geometric information access

Set \(g_t(x)=(-1)^{f_b(\pi^t(x))}\). Let \(\widehat g_t(A)\)
be its uniform Boolean Walsh coefficient on subset A. Define degree
energy \(E_k(t)=\sum_{|A|=k}\widehat g_t(A)^2\) and
\(\rho=1-2p\). The balanced function has E0=0, and Parseval gives
the total energy 1. A BSC attenuates each Walsh mode by \(\rho^{|A|}\).

The posterior sign mean \(m_t(y)=E[g_t(X)\mid Y=y,\pi]\) therefore obeys

\[
V_t(p)=E_Y[m_t(Y)^2]
=\sum_{k=1}^n \rho^{2k}E_k(t). \tag{SUD2-2}
\]

For \(-1\le m\le1\), binary relative-entropy inequalities yield

\[
\frac{m^2}{2\ln2}
\le 1-h_2((1+m)/2)\le m^2.
\]

The lower inequality follows from Pinsker; the upper from
\(h_2((1+m)/2)\ge1-m^2\). Averaging establishes

\[
\boxed{\frac{V_t(p)}{2\ln2}\le I_t^\star\le V_t(p).}\tag{SUD2-3}
\]

The semantic Fourier degree distribution can change with t even though
the underlying computation is exactly reversible.

## SUD2-3: Exact reversible information echo

Consider the two-bit reversible map

\[
\pi(b_0,b_1)=(b_0\oplus b_1,b_0).
\]

It corresponds to integer permutation (0,3,1,2), of order three.
Reading bit b0 after powers of \(\pi\) gives, cyclically,
b0, b0 xor b1, b1, b0. Thus at p=.1:

| Depth mod 3 | Readout | Walsh degree | I(S;Y) | Optimal error |
|---|---|---:|---:|---:|
| 0 | b0 | 1 | 0.531004 | .10 |
| 1 | b0 xor b1 | 2 | 0.319923 | .18 |
| 2 | b1 | 1 | 0.531004 | .10 |

Because \(\pi^L=\mathrm{id}\) entails
\(g_{t+L}=g_t\), the exact source information and V curve are
L-periodic. The irreversible-looking information loss and recovery
arise from noisy observation of a reversible system.

## SUD2-4: Additional loss in the native Q5 head

The neural model receives the intact permutation edges and only the
corrupted start Y. The full computation restarts for each Y.
The original clean X enters only the separate scoring oracle.

Let \(M_t(y)\) be the declared one-bit projection of the model's
predicted cursor. Data processing gives

\[
I(S_t;M_t\mid\pi)\le I(S_t;Y\mid\pi)
\]

and therefore

\[
\boxed{\Delta_t(\pi)
=H(S_t\mid M_t,\pi)-H(S_t\mid Y,\pi)\ge0.}\tag{SUD2-4}
\]

A perfect pointer predictor of \(\pi^t(Y)\) can still have positive
additional information loss, since its one-bit output may discard
information about the original X present in Y. The hidden state and
full logits are wider observation channels and belong to later
experiments.

The reference dimension

\[
D_t^{\mathrm{iid}}=
\frac{H(S_t\mid Y,\pi)}{\log_2(1/r)},\quad0<r<1/2
\]

belongs to separated Cantor codes formed from **independent copies of
this depth-specific experiment**. Steps in a single native trajectory
share X and \(\pi\), so this reference is never asserted as the
Hausdorff dimension of that trajectory. The entropy/Lyapunov analogy
comes from OpenAI Math #148; its stationary self-similar theorem is
distinct from this conditional construction.

## Experimental settings and evidence separation

The development defaults are fixed before measuring Q5 on this branch:
32 independent random permutation draws, seed 940211, a separately
randomized edge presentation, all 16 possible Y values for each graph,
bit b=0, bit-flip rate p=.1, and depths 1,2,4,8,12,16. We sum the
exact probability weights of all 256 (X,Y) pairs at each graph and
depth. An independent graph draw is the averaging unit.

The primary descriptive quantities are exact source information,
Walsh degree-one energy, and native-head information loss Delta.
Companions include native clean-cursor accuracy, noisy-start
accuracy against the clean target, and Bayes-optimal accuracy.

Original Q5 checkpoints (12 models) have known training histories and
extrapolation labels. Any runs on them are retrospective development
data. They require matching the committed SHA-256 receipts. The runner
will refuse missing weights and hashes; permitting a partial
selection requires an explicit option and leaves visible coverage.

## Reproducibility

Exact control and tests, requiring only Python standard library:

    python -m unittest discover -s tests -p test_semantic_noise.py -v
    python -m research_gym.scripts.loopscope_semantic_noise --controls --out /tmp/sud02_exact

Native adapter smoke with PyTorch and NumPy:

    python -m unittest discover -s tests -p test_loopscope_semantic_noise.py -v
    python -m research_gym.scripts.loopscope_semantic_noise --smoke --graphs 2 --depths 1 2 4 --threads 1 --out /tmp/sud02_smoke

Evaluate the original 12 Q5 checkpoints only on the machine with
their frozen, original weights:

    python -m research_gym.scripts.loopscope_semantic_noise --q5-dir experiments/recurrence_monitorability_v0/loopscope_toy_q5/ckpt --device cuda --dtype float32 --out experiments/recurrence_monitorability_v0/sud02_q5

The checkpoint directory is an input path, and every Q5 filename is
checked against its training receipt. The output records source-code
hashes, graph-level measurements, grouped means, trace replay parity
and checkpoint provenance.

No results from trained Q5 weights are presently claimed here.
The exact two-bit echo is a mathematical control and PyTorch smoke
establishes native plumbing. A later comparison with F1/F8 and held-out
extrapolation is exploratory; new trained seeds and a precommitted
prediction rule would be required for a prospective claim.
