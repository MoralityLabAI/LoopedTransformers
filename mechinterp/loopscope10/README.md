# LOOPSCOPE-10 / LGA-07: energy-tangent dark channels

The [prospective source-only protocol](PROTOCOL.md) was committed first at `4a3893606c20c1af47458a41279810aa26f4fb25`. Its source-only implementation tests whether local hidden directions can be simultaneously invisible to current logits and tangent to the frozen shared LGA-05 energy, yet have nonzero model-only future derivatives.

`energy_fiber.py` builds a float64 right-nullspace of the 10-class current goal head and whole-state energy gradient restricted to one 48D token. The physical orthogonal projector and delayed selected-logit derivative are source-blind. A candidate direction and eight equally null, equal-physical-dose shams can be frozen before calculating true target labels. `native_finite_edit` then advances **the actual original tied model blocks** through the remaining horizon. Energy and current-logit preservation are exact only as derivative constraints; finite edit effects are measured separately.

Run source-only tests on newly initialized **untrained** tiny TRM and an artificial quadratic observer:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s mechinterp/loopscope10/tests -p 'test_energy_fiber.py' -v
```

The smoke uses only the excluded generator seed **61059**, with neither source teacher weights nor model labels involved. **Reserved development seed 31059 remains unopened**; the original source-frozen LGA-02/05 teacher/potential bundles are explicitly excluded from automatic PR CI. Full development scoring requires a separate pretest source lock and an externally audited first Stage-B workflow, followed by a new-root EvalMath preflight before any scientific confirmation.

## Research questions

1. Does a token energy-tangent current-head-nullspace have nonzero derivative influence on delayed model outputs?
2. Do its model-score-selected physical directions alter independently verified correct-target outcomes relative to eight same-K equal-L2 random controls?
3. Does finite-step potential curvature and later-loop overshoot explain where energy-tangent interventions lose task utility?

The source-only instrument answers numerical algebra and native-continuation correctness questions, with no result asserted for those three causal hypotheses.
