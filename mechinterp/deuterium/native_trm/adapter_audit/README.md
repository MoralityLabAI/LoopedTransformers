# DEUTERIUM-ADAPTER-01: recurrent weight-tie cache and mask semantics

**Post-result source-only analysis, 19 November 2026.** The first registered real four-pass recurrent-weight VPD study is closed. [First test Actions #44465490737](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44465490737) scored seed 731057 once, and the [independently audited saved-result recovery](../stagec/recovery/README.md) gives `DEVELOPMENT_NO_REGISTERED_ADVANTAGE`: mean matched random-minus-VPD goal NLL −0.00002608 nats across old teachers 103/211/307, 95% Student-t interval [−0.00028738,+0.00023521]. These numbers are frozen.

This diagnostic examines whether the *stock VPD causal-importance adapter* appropriately handles recurrent parameter sharing. It does not retrain or score original VPD, access new goal-target data, or reinterpret the original scientific verdict.

## Source hypothesis

Pinned `MoralityLabAI/param-decomp` commit `2df0092d5be4d8e4748b3c66ef8325dedef17549` stores \`ComponentModel.forward(...,cache_type="input")\` inputs under a dictionary indexed by **module path**: inside \`_components_and_cache_hook\`, \`cache[module_name] = x\`. When the *same* \`block.chan_mix.2\` module is invoked four times during the original \`TinyRecursiveModel(48,4)\` forward pass, this overwrites earlier input tensors, retaining the **last** physical pre-weight activation only.

The built-in per-module \`calc_causal_importances\` therefore uses that final activation to choose one mask for each 81-token source. Downstream masked model calls reapply this same per-module importance mask at all four native recurrent applications. This is a concrete mismatch between the original physical recurrence depth and the decomposition's **single-call site assumption**; it does not alone prove why the VPD steering result was negative. Every trained VPD component has the same actual shared weight matrix throughout inference, so this concern primarily bears on how VPD factors/importance masks were fitted.

## Controlled experiment

- [cache_ledger.py](cache_ledger.py) constructs a pinned stock \`ComponentModel\` and independently registers a native PyTorch forward-prehook to retain *all four* physical pre-weight inputs, while the unmodified stock \`ComponentModel\` exposes its single module-path cache.
- Recompute a diagnostic importance mask from each of the four actual input tensors using the unchanged network's own CI function. Report final-vs-depth input RMS differences, mean and maximum per-depth mask disagreement, and exact identity of the stock cached final pass.
- Register a hook on the actual \`LinearComponents.forward\` to verify the same final-pass \`component_mask\` tensor is passed into all four recurring calls, without changing stock VPD source.
- [tests/test_cache_ledger.py](tests/test_cache_ledger.py) uses a **newly initialized, untrained** 44K model and excluded source seed **61076**. The pretrained-teacher diagnostic downloads the original SHA-locked LGA-02 models **and the actual learned Stage B VPD component/importance-network checkpoints** for 103/211/307 from first run 44463652662. It strictly loads and verifies saved trained-model tensors, reusing eight original teacher-training strings from seed 31041. The reported per-depth importance-mask mismatches for these three teachers therefore come from **the originally fitted VPD CI networks**, while the excluded smoke model uses a new untrained diagnostic CI.
- [audit_receipt.py](audit_receipt.py) independently verifies the four native input calls, one stock cache, repeated mask, source hashes and physical RMS accounting without Torch.

The source-only workflow may run on pull requests repeatedly. It must not generate 731055/731057 or separately reserved LOOPSCOPE science seeds 31059,31064,31065. Any future *replacement* adapter that assigns a per-depth importance mask must be preregistered with its own fit, search-budget matching and fresh models/source test roster.

## Interpretation

If CI passes, \`ONE_PATH_CACHE_LAST_PASS_AND_REUSED_CI_MASK\` is an **engineering observation** about the frozen VPD implementation on a recurring module. It shows a plausible way factor-training objectives could misalign with physically distinct native passes. It is not causal evidence for the negative VPD steering outcome because the original experiment did not randomize adapter semantics, and its supervised task-gradient control has a different training objective. A future 2×2 test should separate optimizer mask provenance (last-pass vs pass-indexed) from intervention target (recurrent weight vs activation), with equal parameter/physical-dose/search budgets and new root-level confirmation.
