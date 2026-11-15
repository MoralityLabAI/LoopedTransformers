# AF-15: Host-gated Horn rule selection with a 5.2K TypedLoop

AF-14's bounded deadline surrogate improved partial milestone measurements while degrading unseen proof-state correctness. AF-15 tests a different allocation of responsibility: a small trainable TypedLoop scores candidate Horn inference rules, and an exact host verifier enforces that only a rule with proved premises can add its head to the state. Every committed intermediate state has a source-bound derivation trace.

Mathematical invariant: for the least Horn closure `S∞` and initial facts `S0`, each accepted transition `S' = S ∪ {h}` with premises `P⊆S` satisfies `S0⊆S⊆S'⊆S∞`. This guarantees soundness and monotonicity. It does not guarantee the model will select enough productive rules to prove a reachable goal within the action budget. The latter is the learned capability being evaluated.

The training label oracle computes the exact minimum number of single-rule firings needed to derive the target goal. The baseline `imitation` arm maximizes probability mass on any optimal eligible rule; the `frontier` arm adds a pairwise ranking margin to favor actions that reduce the remaining proof distance. Both models are initialized identically and trained for 240 updates with matched inference-cell calls. The additional frontier loss incurs extra backward arithmetic.

Train/development sets use the unchanged AF-05 logical task and graph layouts. The prospectively frozen heldout cases use six new layout permutations, including 512 depth-4/5 examples, while preserving the same logical AND/OR templates. An exact oracle, uniform legal-action policy, and fixed-order policy define calibrated comparison points. Only the host accepts proofs, with no claim of novel Lean theorem discovery by the model.

## Reproduce locally

Run from the repository root with Torch CPU, NumPy, Python and the source-pinned AF-02/AF-05 dependencies:

```bash
python -m unittest discover -s experiments/activation_funnel_af15/tests -p 'test_*.py' -v
python -O -m unittest discover -s experiments/activation_funnel_af15/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af15.experiment train --seed 17 --out artifacts/af15/s17/training
python -m experiments.activation_funnel_af15.experiment evaluate --seed 17 --training artifacts/af15/s17/training --out artifacts/af15/s17/evaluation
python -m experiments.activation_funnel_af15.verify --seed 17 --training artifacts/af15/s17/training --evaluation artifacts/af15/s17/evaluation --out artifacts/af15/s17/evaluation/independent_audit.json
```

Repeat seed 29, then run `python -m experiments.activation_funnel_af15.aggregate --root artifacts/af15 --out artifacts/af15/results.json`.

The protocol and data manifest are registered before scoring. Keep all negative outcomes. The exact host rule gate makes illegal proof firing impossible by construction, so improvements in *proving reachable goals under a fixed action budget* are the only permissible capability claims.
