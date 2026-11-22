# IPB-01 validation receipt, 2026-11-22

Scope: **implementation and synthetic unit-test verification only**. The full released Inkling 2 Mini checkpoint was not available in the executing environment; no real model positional-bias curves, attention probabilities, antecedent preferences, or performance changes were measured. The tests use explicitly synthetic arrays for all numerical assertions and a tiny synthetic safetensors file to validate native checkpoint key lookup and selective embedding-row extraction.

Local command:

```
PYTHONPATH=. python -m unittest discover \
  -s research_gym/inkling_position_bias -p 'test_*.py' -v
```

Local result: **8 tests passed; 0 failed; 0 skipped** on Python 3.13 with NumPy 2.3.5, `torch` and `safetensors` available. Checks cover layer-0 dual RMSNorm, projection and head separation, L2 clipping and logarithmic scaling, distance masks and the zero-bias tail, sink-aware frozen-logit ablation, four-tap static weight summaries, native checkpoint key lookup, and dimension rejection.

CI runs standard-library unittest and compileall on Python 3.10 and 3.12. Because PyTorch is optional in the lightweight CI environment, the safetensors-reading test reports skipped there if `torch` is absent; the seven other numerical tests still run.

The next data-bearing action is to run the documented extractor against a pinned model revision of `thinkingmachines/Inkling-2-Mini`. Publish the per-token/head bias arrays and measured attention decomposition only once that checkpoint and any captured runtime traces are present. Keep LoopVQ/LoopKV compression outside IPB-01's evidence claims.
