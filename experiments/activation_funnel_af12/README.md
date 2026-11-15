# AF-12: depth-transport via synchronous Horn closure supervision

AF-11's independently scored failure concentrated at its **upstream Q boundary**: the eligible seed-17 influence model had only 14.06% exact joint-Q accuracy on new depth-5 instances. The downstream causal cut guarantees code transport, and the Boolean-influence objective trains final goal response, but the model's source Q representation often encodes the wrong truth. AF-12 targets the recurrence *before the cut*.

This is a separately preregistered experiment on top of AF-11. Prior scored weights, checkpoints, mathematical claims, and evidence remain immutable. It uses the original source-hash-verified `TypedLoop`, original AF-05 Horn oracle and AF-06 **qualified** seed-17/29, width-2 warm-ups. No re-fitting is allowed from the earlier AF-11 results or from old held-outs. The newly frozen prospective physical permutation IDs are 14–19, with exact feature arrays disjoint from AF-01 through AF-11.

## Mathematical object

Let `U` be the finite set of atoms and `R` the upstream Horn clauses. The closure operator is

\[
\Phi_R(S)=S\cup\{h:(P\to h)\in R,\ P\subseteq S\}.
\]

It is inflationary and monotone: `S ⊆ Φ(S)` and `S ⊆ T ⇒ Φ(S) ⊆ Φ(T)`. The sequence `S_t=Φ^t(S_0)` therefore stabilizes in at most `|U|` strict-growth rounds. In particular, downstream Q truth belongs to `S_depth`, and each intermediate relay is a verifiable derived variable. `HornClosure.lean` checks the Boolean AND/OR monotonicity primitives using Lean's decision procedure; the general finite-set induction is supplied here as a mathematical argument rather than claimed as a formally verified Lean proof.

The neural recurrence provides a probability \(p^t_v\) of each atom's membership in \(S_t\) after native-cell call `t`. The stage targets are recomputed independently from frozen Horn rules and **never supplied as input features**. The additional loss for the full process arm is

\[
L_{\mathrm{closure}}=0.35\,\mathbb E_{v,t}\,w_{vt}\,
\mathrm{CE}(p^t_v,\mathbf{1}[v\in S_t])
+0.15\,\mathbb E_{v,t}\bigl[\max(0,p^t_v-p^{t+1}_v)\bigr]^2.
\]

Positive labels receive weight two. `relay_final` supervises all intermediate atom statuses only after the final prefix call, using the same `0.35` CE coefficient. The `terminal` arm receives only the existing final Q and downstream goal losses. All three arms execute a separately traced prefix with identical source-cell calls, then the same clean, addressed-patch, and unrelated-patch causal-cut rollouts. Additional losses and backward arithmetic make total FLOPs unequal, even with identical optimizer updates and source-cell forward calls.

## Frozen experiment

- Fit: original AF-05 train depths 2/3 with layouts 0/1/2, 672 canonical inputs; dev depths 2/3 on layout 3, 224 inputs used for qualification.
- Prospective heldout: depth3/layout18 (128), OR/OR→AND/layout19 (16), depth4/layouts14–15 (256), depth5/layouts16–17 (256). They use existing logical templates and novel physical permutations.
- Two seeds, three arms, exactly 160 AdamW updates per arm, 32 examples per update sampled 16 from each training depth. Source initialization and minibatch/donor schedules are matched per seed. Each source carries 58 float32 values over 29 typed tokens, with 464 decoded working values.
- The preregistered comparison uses the final-step checkpoints and requires `relay_final` and `closure_rollout` to reach ≥0.90 goal and exact Q accuracy separately on train **and** dev. For each eligible seed, process supervision must improve upstream exact joint-Q and final atom accuracy by ≥0.05 on both depth4 and depth5, with clean-goal accuracy within 0.03, informative goal-patch accuracy within 0.03, no new false-positive goals, and irrelevant-patch goal retention ≥0.90. An eligible failure makes the aggregate FAIL. Ineligible results remain INCONCLUSIVE.
- Fresh cases are checked by exact input hashes and independently derived Horn targets. Numerical replays cover the original model's stage tensors and donor patches for every arm/family; all raw stage logits and counterfactual logits are retained.

### Reproduce

From the repository root (Python 3.12, Torch 2.5.1+cpu, NumPy ≥1.24):

```bash
python -m unittest discover -s experiments/activation_funnel_af12/tests -p 'test_*.py' -v
python -O -m unittest discover -s experiments/activation_funnel_af12/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af12.experiment train --seed 17 --source-zip qualified-af06-seed17.zip --out artifacts/af12/s17/training
python -m experiments.activation_funnel_af12.experiment evaluate --seed 17 --train artifacts/af12/s17/training --out artifacts/af12/s17/evaluation
python -m experiments.activation_funnel_af12.verify --seed 17 --training artifacts/af12/s17/training --evaluation artifacts/af12/s17/evaluation --out artifacts/af12/s17/evaluation/independent_audit.json
```

The workflow obtains exact GitHub Actions checkpoint artifacts and verifies their SHA-256 before invoking training. Lean checks the Boolean monotonicity primitives and the new graph's closed Horn oracle when their prerequisites are present. Every held-out family and failed hypothesis is preserved without selection or retuning.
