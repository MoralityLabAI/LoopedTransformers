# AF-15: structural Horn soundness and learned proof-rule scheduling

Let `S0` denote the initial facts, `R` a finite definite-Horn rule system, and `S∞` the least fixed point of the inflationary Horn closure `Φ`. Define the host's legal action set

\[ E(S)=\{r\in R:\operatorname{prem}(r)\subseteq S,\ \operatorname{head}(r)\notin S\}. \]

The host changes the proved state only through

\[ S_{t+1}=S_t\cup\{\operatorname{head}(r_t)\}, \quad r_t\in E(S_t). \]

For every run (including neural models that choose badly), `S0⊆S_t⊆S∞` and `S_t⊆S_(t+1)`. The induction is immediate: `S∞` contains initial facts and is closed under every rule in `R`, so an enabled rule with premises in `S_t⊆S∞` has its head in `S∞`. STOP leaves the state unchanged and earns no proof claim. Every reached goal therefore has a valid source-bound Horn derivation. This is a guarantee of **semantic soundness**, conditioned on the trusted host interpreter and rule inventory. It does not imply that a neural policy finds the derivation efficiently.

The model sees the current host proof state, public atoms and incidence, and selects a rule slot. The host masks disabled rules. Training uses an independent finite shortest-path oracle `V(S)` for the smallest number of valid one-rule steps necessary to reach the goal, or `∞` when unreachable. On training states with `0<V(S)<∞`, an optimal enabled action satisfies

\[ V(S\cup\{\operatorname{head}(r)\})=V(S)-1. \]

AF-15 compares a probability-mass objective on the optimal action set with an additional positive-versus-negative rule-ranking loss. Evaluation gives one additional action beyond the exact shortest proof length. Exact shortest-path, uniform legal choice and fixed physical rule order serve as calibration baselines under the same budget.

## Failure modes and scientific scope

* **Premature STOP:** the model may terminate before proving a reachable goal. The host records abstention and awards no proof credit.
* **Unproductive but legal rules:** they keep states sound while consuming the action budget, lowering solve rate.
* **Scorer dependence:** physical graph permutations should preserve abstract rule scores up to floating-point tolerance; a failure indicates unintended positional shortcuts.
* **No neural proof-discovery claim:** the externally enforced Horn gate and shortest-path label oracle supply formal semantics. This experiment studies learned rule *scheduling* in a neuro-symbolic system, not unrestricted Lean tactic search.

The pure [Lean Boolean lemmas](../../experiments/activation_funnel_af15/RuleSoundness.lean) express the pointwise inflationarity and soundness principle. Full finite-graph semantics and all original-theorem checks remain outside these lemmas; the independent evaluator reconstructs every accepted premise and head from the public incidence tensors.
