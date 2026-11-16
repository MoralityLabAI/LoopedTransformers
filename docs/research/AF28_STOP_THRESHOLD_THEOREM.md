# AF-28 mathematical note: when does a single STOP threshold separate provable and unprovable roots?

**Proof scope:** finite, source-registered binary STOP decisions on publicly represented positive-Horn states. The statements are elementary finite-order-statistic theorems with complete proofs here. They are not claims of Lean-kernel verification, learned neural generalization, or globally optimal proof search.

## A threshold feasibility theorem

Let \(P\) be a finite set of \(n_P\) *provable* nonterminal roots and \(N\) a finite set of \(n_N\) *unprovable* nonterminal roots. Let \(r(s)\in[0,1]\) be any score intended to estimate the probability that the host should immediately STOP. For a fixed scalar threshold \(t\), define

\[
g_t(s)=\mathbf 1\{r(s)\ge t\}.
\]

Assume each source root still has an enabled physical rule, so the gate genuinely chooses between STOP and continuing. A false initial STOP on \(P\) is a root-level false positive. A correct initial STOP on \(N\) is a root-level true positive.

Write the provable STOP scores in descending order \(p_1\ge p_2\ge\cdots\ge p_{n_P}\), and likewise \(u_1\ge\cdots\ge u_{n_N}\) for unprovable STOP scores. Fix integers \(0\le k<n_P\) and \(1\le m\le n_N\).

**Theorem (single-threshold root feasibility).** There exists a real threshold \(t\) for which at most \(k\) provable roots are falsely stopped and at least \(m\) unprovable roots are correctly stopped **if and only if**

\[
\boxed{p_{k+1}<u_m.}
\]

**Proof.** At most \(k\) provable roots satisfy \(r(s)\ge t\) exactly when \(p_{k+1}<t\), taking the strict inequality because any tied \((k+1)\)st root is stopped at equality. At least \(m\) unprovable roots are stopped exactly when \(t\le u_m\). Both conditions hold for some \(t\) precisely when the interval \((p_{k+1},u_m]\) is nonempty, which is equivalent to \(p_{k+1}<u_m\). \(\square\)

The theorem applies to **any** scalar gate, including a logistic model with frozen coefficients and a dev-selected threshold. It makes clear that changing only the threshold cannot repair a defective order of positive and negative roots.

### Fixed threshold grid

AF-28 selects thresholds from a predeclared finite grid \(\mathcal G=\{0.05,0.10,\dots,0.95\}\). The stronger discrete criterion is

\[
\mathcal G\cap(p_{k+1},u_m]\ne\varnothing.
\]

The grid can miss a feasible real interval, so the developer must report both the grid and the actual dev confusion counts. The experiment does exactly this in its signed \`dev_grid_receipt\`.

## Decision dependence on source features

Let \(x(s)\in\mathbb R^6\) comprise the six frozen AF-23/public Horn features used in the registered AF-28 logistic gate, and let the model coefficients \(w,b\) be learned exclusively from the TRAIN grammars. Its score is

\[
r_{\rm CAL}(s)=\sigma\!\left(b+w^\top x(s)\right).
\]

Since sigmoid is strictly increasing, for any two states \(s,s'\),

\[
r_{\rm CAL}(s)>r_{\rm CAL}(s')
\quad\Longleftrightarrow\quad
w^\top x(s)>w^\top x(s').
\]

The DEV-selected threshold changes the operating point but does not change this ordering. The one-feature THR control has its own train-fitted \(w_0,b_0\) and therefore supplies a lower-capacity alternative ordering over the Direct STOP margin alone.

**Interpretive rule.** If CAL and THR both achieve the registered end-to-end gate, and THR matches or exceeds CAL's held-out correct-negative STOP at similar provable proof completion, the result supports at most a benefit from prospective threshold/score calibration. An advantage due specifically to the extra reach, distance, VI-2 margin and public legal-count features would require a paired outcome superiority to THR.

## Closed-loop proof completion is more demanding

The finite-order-statistic theorem concerns **initial root STOP classification**. AF-28's primary gate also requires whole-episode proof completion under a fixed shortest-plus-one action budget.

Let \(\pi_D\) choose legal rules by frozen Direct logits, and let \(\pi_{g}\) use the same legal-rule ranking but a potentially different STOP gate \(g\) at every reached state. For any initial case \(c\), if the two policies produce the same STOP decision at every state along their common trajectory, then their selected legal actions and subsequent host states agree at every step. By induction on the number of executed Horn firings, they have the same complete-proof outcome.

The converse fails. A gate may classify the root correctly and nevertheless STOP prematurely after one or more successful legal Horn firings. This matters on positive Horn graphs because the **true** eventual reachability label is invariant under legal positive-rule firing, while a recurrent learned reach score can vary at the successor state.

A gate's root-level true-negative/true-positive performance must therefore be reported beside its complete proof completion and *later* premature-STOP counts. A perfect root ROC curve would not certify complete-proof success.

## How AF-28 tests the statements

AF-28 freezes 600 full-batch logistic updates and the standardization parameters on TRAIN grammars. DEV supplies only the threshold grid and its root classification labels. The HELD families are structurally distinct and withheld from fitting and threshold selection.

Both CAL and the one-feature THR comparator use exactly the same train roots/successor samples, loss, update count, class balancing and selection grid. The frozen Direct rule ranking controls the executed legal action after either STOP gate chooses to continue. This isolates one measurable intervention at the **policy** level without claiming that the trained coefficients are causally localized within the original recurrent model.

The original 6,036 recurrent parameters remain untouched. Any improvement is a property of the added light STOP classifier and its source-bound grammar-disjoint data, not newly trained internal representations in the original network.

**Scientific status of this note:** exact proved threshold feasibility and closed-loop identity, with the empirical AF-28 verdict determined separately by the eight-seed source-pinned held-out CI results.
