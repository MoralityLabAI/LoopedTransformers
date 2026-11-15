# AF-20: STOP mass-gap versus greedy decision-gap theorem

This mathematical observation was derived **after the AF-20 scientific protocol was frozen**. It is explanatory and motivates AF-21; it has zero effect on AF-20 training, score gates, or checkpoint selection.

## Exact statement

Let \(s\) be the STOP logit and \(r_1,\ldots,r_k\) the \(k>=1\) source-legal Horn rule logits. Define

\[
 m=\max_i r_i,\qquad L=\log\left(\sum_i e^{r_i}\right),
 \qquad g_{\mathrm{mass}}=s-L,\quad g_{\mathrm{greedy}}=s-m.
\]

The log-sum-exp inequality gives

\[
 m\le L\le m+\log k
 \quad\Longrightarrow\quad
 g_{\mathrm{mass}}\le g_{\mathrm{greedy}}\le g_{\mathrm{mass}}+\log k.
\]

**Theorem.** The actual host-gated greedy policy selects STOP if \(g_{\mathrm{greedy}}>0\). A negative \(g_{\mathrm{mass}}\) is therefore insufficient to guarantee a legal rule is selected. A sufficient condition for choosing a rule, with strict margin \(\tau>0\), is

\[
 g_{\mathrm{mass}}<-(\log k+\tau).
\]

**Proof.** Write \(a_i=r_i-m\le0\). Because at least one \(a_i=0\), \(1\le\sum_i e^{a_i}\le k\), whence \(0\le L-m\le\log k\). Substitute into both logit-gap definitions. The sufficient condition yields \(s-m< -\tau\), which ensures the maximum legal-rule logit strictly exceeds STOP. QED.

The AF-20 BCE and counterfactual losses use \(g_{\mathrm{mass}}\) while **rollout uses argmax**, so this logical mismatch is present in both controls. The paired ranking objective imposes \(g^-_{\mathrm{mass}}-g^+_{\mathrm{mass}}>0\) for reachable/unreachable examples, but both gaps can still be positive. Thus a correct relative ordering also fails to ensure an absolute continue decision on reachable inputs.

## Descriptive first-action audit on the successful hosted checkpoints

Each of the 16 model configurations was replayed on the originally scored depth-4/5 provable histories. The table counts cases where the source already had a legal enabled rule, using the **saved first-step raw logits**.

| Seed | Arm | Reachable initial cases | Chose STOP immediately | Mass gap negative | Mass gap negative while greedy chose STOP |
|---:|---|---:|---:|---:|---:|
| 17 | bce_l2 | 148 | 30 | 118 | 0 |
| 17 | bce_l4 | 148 | 138 | 24 | 14 |
| 17 | pair_l2 | 148 | 30 | 118 | 0 |
| 17 | pair_l4 | 148 | 148 | 0 | 0 |
| 29 | bce_l2 | 148 | 20 | 128 | 0 |
| 29 | bce_l4 | 148 | 112 | 42 | 6 |
| 29 | pair_l2 | 148 | 126 | 32 | 10 |
| 29 | pair_l4 | 148 | 130 | 28 | 10 |
| 43 | bce_l2 | 148 | 124 | 34 | 10 |
| 43 | bce_l4 | 148 | 142 | 32 | 26 |
| 43 | pair_l2 | 148 | 140 | 26 | 18 |
| 43 | pair_l4 | 148 | 148 | 0 | 0 |
| 61 | bce_l2 | 148 | 124 | 36 | 12 |
| 61 | bce_l4 | 148 | 112 | 44 | 8 |
| 61 | pair_l2 | 148 | 140 | 12 | 4 |
| 61 | pair_l4 | 148 | 72 | 84 | 8 |

The proxy mismatch accounts for some individual errors, while models with positive mass gaps can STOP immediately even without that discrepancy. For example, seed 17 paired four-loop stops on all 148 examined reachable cases at their first action. This experiment therefore indicates a broader failure to learn absolute continue-action margins, alongside the particular log-sum-exp/argmax calibration gap.

A future intervention should train the actual masked \(s-\max_i r_i\) margin and separately preserve correct next-rule ranking under the same host-owned eligibility. The new objective and hypotheses must receive a **new preregistration**. AF-20's failed scientific result remains frozen.
