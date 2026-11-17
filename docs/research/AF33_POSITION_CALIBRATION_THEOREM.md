# AF-33: position-balanced STOP thresholds under Horn grammar shift

**Pre-scoring mathematical note.** The experimental protocol was first frozen in commit \`14893fb6\`. These statements concern distributions over the publicly derived STOP score, conditional on true Horn entailment and source grammar location. The results supply interpretation of source-locked AF-33 evidence rather than a learned-reachability theorem.

## Definition: STOP calibration under a location mixture

Let \(\mathcal G\) be a space of public positive-Horn graphs, each with four initial facts and a designated goal. For a graph \(G\), let \(Y(G)=1\) mean the goal is **unreachable** from the public root, so that immediate STOP is correct. Let \(J\in\{1,2,3\}\) be an *evaluation-only* latent location indexing which middle productive premise was swapped. The policy receives the public graph, not \(Y\) or \(J\).

For a STOP scorer \(s_\theta(G)\in[0,1]\) and fixed DEV-only threshold \(t\), define

\[
T_j(t)=\Pr[s_\theta(G)\ge t\mid Y=1,J=j],\qquad
F_j(t)=\Pr[s_\theta(G)\ge t\mid Y=0,J=j].
\]

Here \(T_j\) is correct negative immediate STOP sensitivity at location \(j\), and \(F_j\) is erroneous STOP on a reachable root. With \(J\) uniform within each truth label, the balanced performance is

\[
T_{\mathrm{bal}}(t)=\frac{T_1(t)+T_2(t)+T_3(t)}3,\qquad
F_{\mathrm{bal}}(t)=\frac{F_1(t)+F_2(t)+F_3(t)}3.
\]

The AF-33 position-ablated control instead selects its threshold from \(J\in\{1,2\}\), so its DEV objective measures \((T_1+T_2)/2\) and \((F_1+F_2)/2\). The control has fewer unique TRAIN/DEV examples. Its smaller data support must be reported and precludes attributing every potential advantage solely to the abstract idea of position balancing.

## Proposition: exact transfer bound at any frozen threshold

Suppose heldout distribution \(H\) and DEV distribution \(D\) have the same balanced \(J\) mixture conditional on each truth label. Write \(T_j^H,T_j^D\) for the conditional STOP probabilities at a fixed threshold, and similarly \(F_j^H,F_j^D\).

If for a particular frozen threshold \(t\),

\[
|T_j^H(t)-T_j^D(t)|\le\delta_j^{(1)},\quad
|F_j^H(t)-F_j^D(t)|\le\delta_j^{(0)}
\quad (j=1,2,3),
\]

then

\[
|T_{\mathrm{bal}}^H(t)-T_{\mathrm{bal}}^D(t)|
\le \frac13\sum_j\delta_j^{(1)},\qquad
|F_{\mathrm{bal}}^H(t)-F_{\mathrm{bal}}^D(t)|
\le \frac13\sum_j\delta_j^{(0)}.
\]

**Proof.** Substitute the mixture identities and apply the triangle inequality to the average of the three conditional differences. \(\square\)

A stronger statement follows if \(\delta_j^{(y)}\) bounds the Kolmogorov distance between DEV and held conditional score CDFs for each \(j\) and \(y\): the same inequalities then hold **uniformly over every threshold**. Therefore DEV-only selection of \(t\) transfers quantitatively when position-conditional score distributions are stable between DEV and held motifs.

## Failure of position support to guarantee semantic transfer

Uniform location support alone cannot bound those \(\delta_j\). In a new goal-clause motif, the model may assign the same score to a reachable/unreachable pair or shift all scores for a held subset across the frozen threshold. All three locations can be equally represented in DEV, yet held STOP sensitivity can be arbitrarily poor if the conditional feature/label relationship changes. AF-33 introduces a novel held goal-concluding rule arity and new root-diversion incidence to exercise this limitation.

Even exact matched-pair ranking \(s(G_{\mathrm{negative}})>s(G_{\mathrm{positive}})\) on every held pair does not imply that a **single global threshold** separates all positives from negatives: one pair can have scores \((0.91,0.90)\), another \((0.11,0.10)\), making a common threshold impossible despite perfect pairwise ordering. This is a finite, constructive counterexample to interpreting paired AUC alone as a viable STOP controller.

## Exact local-observer null

Each matched AF-33 pair retains identical seven-anchor physical-ID-preserving directed radius-two incidence, node degrees, initial-known and role markers, physical permutation and legal root action mask. Therefore any deterministic score that depends only on that registered local signature assigns equal scores to the pair. Under 96 balanced matched pairs, the within-pair AUC is exactly \(1/2\), counting every tie half. The complete public Horn graph contains the information needed by exact least-fixed-point closure and the learned policies, so this is a **restricted-observer impossibility**, not a bound on a full-graph model.

## Prospective interpretation

AF-33 preregisters an **absolute STOP test** and a **BAL-vs-J12 correct negative STOP improvement** at the same strict proof budget. Every success must preserve adequate completed positive proofs and a bounded false positive immediate STOP rate. Root matched AUC, Brier, NLL and ECE and by-position/cycle reports are separately registered diagnostic measurements, not alternative pass criteria.

If BAL outperforms J12 on heldout unseen goal motifs, the source supports improved generalization of this *particular* ten-feature classifier under enhanced position support. A successful registered gain would not establish a general learned Horn theorem prover, because complete public closure remains available and has a different symbolic compute cost. Any search across heldout thresholds, new feature ablations, or new grammar families must be scored as a separate prospective experiment with a new source lock.
