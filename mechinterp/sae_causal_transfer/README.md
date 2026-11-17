# SAE causal transfer for looped transformers (LS-SAE01)

This module brings the **SMC-13 through SMC-17** feature-transfer campaign, and the **OLM-14 through OLM-17** causal geometry, into LoopedTransformers/mechinterp. It contains executable physical-SAE intervention planning, a source-indexed negative-result ledger, and an exact-rational loop-depth witness. The current execution is a portable observer and mathematical control gym. **No new pretrained LoopedTransformers/LoopScope/LoopedChase checkpoint result is claimed by this module.**

The key empirical update is [SMC-17 PR #179](https://github.com/MoralityLabAI/Compactification-Theory/pull/179): an independent hosted full/pilot workflow and stdlib audit passed, while its **registered scientific gate passed for 1/4 independently fitted receiving SAE comparisons** after improving all-92-inactive-feature displacement matching. The immediately preceding [SMC-16 PR #177](https://github.com/MoralityLabAI/Compactification-Theory/pull/177) passed 3/4 against simpler four-active-feature-preserving tangent shams. Those decisions and the failed pair conditions are retained in [source_evidence.json](source_evidence.json).

## What can now be run in this repo

- [observer.py](observer.py): accepts a frozen hidden vector and a physical 96×d top-four SAE dictionary. Performs a minimum-L2 target-feature kill while preserving three other active preactivations, then proposes a 4-seed original tangent sham and selects the best inactive-preactivation-matched **same-dose** sham from the original valid baseline, signed unconstrained least-squares tangents and 128 seeded Gaussian tangents. Selection uses encoder geometry and clean feature activation only. Full 96-dimensional displacement traces, rank, residual floor, effect dose, refusal reasons and finite candidate receipts are saved. A separate callback scores the clean and edited task margins after selection is frozen.
- [adapter.py](adapter.py): converts a normalized SAE to physical hidden-state covectors and decoder directions; also provides a deterministic array digest to accompany the original checkpoint-file SHA256.
- [loop_depth.py](loop_depth.py): exact Fraction witness on two weight-tied six-dimensional linear recurrent teachers with identical clean SAE codes and correct task classifications. Their per-loop matched-feature causal effects are **+5k/8** and **−5k/8** at recurrence depth k, including k=0,1,2,4,8. The 92-row inactive measurement system has full column rank and a strictly positive unavoidable matching residual. This is a mathematical construction, not a measured neural feature.
- [tests/test_observer.py](tests/test_observer.py): checks full-rank inactive encoder leakage, normalized coordinate conversion, frozen-control repeatability, active coefficient preservation, source-gate fidelity, opposite-sign loop-gradient effects, and fail-closed cases.

Run at the repository root:

    python -m pip install numpy==2.1.3
    python -m unittest discover -s mechinterp/sae_causal_transfer/tests -p 'test_*.py' -v
    python -m mechinterp.sae_causal_transfer.loop_depth --out runs/ls_sae01_exact_toy.json

The dedicated CI workflow runs the same checks and uploads the exact-rational JSON output.

## Interfacing with actual LoopScope checkpoints

First freeze a checkpoint and its original file-byte SHA-256, model architecture, loop step, source/receiving SAEs, top-four feature correspondence, and calibration feature-assignment hash. Preserve complete calibration and test puzzle manifests. The original SMC teacher used 48-dimensional hidden states, while **LoopScope-02's real TRM** is a different model; its SAE must be independently fitted and source-locked, with physically matching hidden dimension and loop site.

Convert normalized SAE weights into physical hidden coordinates. If z(h) is computed on x=(h−mu)/scale, then the physical encoder has rows W/scale and bias b−(W/scale)mu; physical decoder columns are diag(scale)D. Use:

    from mechinterp.sae_causal_transfer.adapter import to_physical_dictionary
    from mechinterp.sae_causal_transfer.observer import plan_edit, score_margin

    physical_W, physical_bias, physical_decoder = to_physical_dictionary(
        sae.encoder.weight, sae.encoder.bias,
        sae.decoder.weight, train_mean, train_scale)

    edit = plan_edit(hidden_state, physical_W, physical_bias,
                     physical_decoder, matched_target_feature,
                     case_key="frozen-calibration|test-family|puzzle-seed|goal-position")

    if edit["eligible"]:
        effects = score_margin(
            edit, hidden_state,
            continuation=lambda changed_site: frozen_model_goal_margin(
                changed_site, fixed_clean_rival=True))

The caller's continuation must substitute the changed vector into **one predetermined token at one predetermined recurrence step**, hold all other token states unchanged at injection, then replay the frozen model for a preregistered loop budget. It must fix the correct digit and strongest clean non-target rival for evaluation *after* the candidate search. The callback in the code example is schematic and must be implemented for the chosen LoopScope checkpoint. The current portable planner uses NumPy float64; it makes no claim of bit-identical FP32/topk or Torch RNG behavior with the original SMC runner.

For the next LoopScope run, preregister a loop-depth grid and score the same frozen interventions at each unroll depth. Let J_{t→T} be the Jacobian of the frozen continuation and c the clean target-minus-rival readout covector. The first-order margin response of a matched kill delta versus sham s is

\[
D_T\simeq c^\top J_{t\to T}(\delta-s).
\]

More recurrent steps can amplify, cancel or reverse the direction. Compare task-verified margins, flip rates and output gradients across depth and source/receiving checkpoints, preserving puzzle-ID clustering. **The exact linear witness verifies the formula, not the behavior of any trained checkpoint.**

## Scientific and control boundary

The historical sequence must be read with its stronger baselines:

| Experiment | Frozen question | Registered pair verdict |
|---|---|---|
| [SMC-13](https://github.com/MoralityLabAI/Compactification-Theory/pull/171) | Can activation overlap occur across independently trained teacher weights after visible clue/position conditioning? | Observational; Gaussian autoencoding alone yielded 95/96 and 94/96 self-selection cones |
| [SMC-14](https://github.com/MoralityLabAI/Compactification-Theory/pull/174) | Does a frozen cross-checkpoint SAE match predict finite decoder-direction causal sensitivity? | 4/4 |
| [SMC-15](https://github.com/MoralityLabAI/Compactification-Theory/pull/176) | Does erasing a naturally active matched native decoder contribution matter more than an equal-dose other-active erasure? | 4/4 |
| [SMC-16](https://github.com/MoralityLabAI/Compactification-Theory/pull/177) | Does minimum-norm selective encoder feature kill beat a tangent sham preserving the other active features? | 3/4; B501 SAE89 failed coverage |
| [SMC-17](https://github.com/MoralityLabAI/Compactification-Theory/pull/179) | Does the effect survive a 131-candidate sham search matching shifts in all 92 inactive encoder rows? | **1/4**; only B701 SAE89 passed both fresh cohorts |

The [OLM-17 mathematical result](https://github.com/patrickdugan/MathCan/pull/100) shows why perfect inactive displacement matching is generally impossible when the 92×d inactive matrix has full column rank. SMC-17 found rank **48/48 on all 267 selected goals**. The least-squares floor was close to the finite candidate best residual, and more sham refinement alone is unlikely to remove the remaining rank obstruction.

The next native LoopedTransformers study should measure how much of the effect is explained by **the actual task-gradient projection** along the matched edit, and the residual nonlinear response after each recurrent step. Require a task-only matched-budget comparator, **at least 100 rank-matched random observer bases**, corresponding normalised random intervention controls, visible-token/position/digit-conditioned shuffles, fresh solution-family separation and no heldout-label-assisted feature or sham selection. The existing 128 Gaussian sham directions are geometry controls and do not substitute for the independent random-observer-basis baseline.

Successful geometric editing can coexist with a feature that is causally irrelevant to the task, as OLM-16's counterexample shows. SMC-17 receiving teachers had substantially lower Sudoku accuracy than the original pretrained teacher. All measurements are conditional on active receiving features and retained full-SAE reencoding validity. A source audit replay verifies recorded numerical observations rather than independently recomputing trained-model forward passes.

## Related work in this repository

- [LoopScope-02](../loopscope02/README.md): learned TRM hidden-state roles, fiber controls, Sudoku propagation strata and model-specific causal ablations.
- [LoopScope-03](../loopscope03/README.md): three-ledger recurrence, path-independence and cached-state interpolation across frozen looped checkpoints.
- [Phase-gauge07](../phase_gauge07/PROTOCOL.md): label/phase-conditioned controls and source-locked output telemetry.

Use the same heldout split firewall and current LoopScope native checkpoint harness when upgrading this portable observer to trained-network evidence.
