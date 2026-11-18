# LOOPSCOPE-08 / LGA-05: Native recurrent energy and curl

This gym directly tests whether one scalar physical-state potential can explain the learned recurrent updates of the original 44,001-parameter four-pass TinyRecursiveModel on new cyclic-copy tasks.

The prospective experiment trains an external **depth-independent E(h)** on the original TRM's 48D hidden-state update vectors, then tests source-disjoint sequences for alignment of −∇E with each F(h)−h, actual decrease in that SAME energy across all four passes, and genuine downstream effects of matched-L2 physical edits. Its conditioned-depth, untrained and train-only radial potentials provide fixed controls. The original frozen teacher/SAE weights, true native terminal gradients, trained value critics and earlier model checkpoints are never refitted.

**Original registered verdict: POSITIVE, 3/3 independent frozen teacher seeds, all 12 depth cells passing.** Across fresh source seed31057, shared negative-gradient/native-update cosines average +0.771,+0.567,+0.495,+0.287 over passes k=0..3, and the shared scalar E decreases across all four actual recurrence steps on 256/256,255/256,256/256 full sequences in the three original model seeds. The original field's FP64 two-plane Jacobian skew witness is 0.733 and 0.757 at intermediate depths, so the full native transition is NOT a pure Euclidean gradient field in the sampled neighborhoods. Native same-dose physical edits descending E have no nominal advantage over isotropic random edits in six matched seed×depth task-CE comparisons.

- [Original preregistration](PROTOCOL.md)
- [Complete scientific results, confidence intervals, native causal and curl evidence](RESULTS.md)
- [Calibration-only physical recurrent field and potential fitter](fit_potential.py)
- [Fresh true native recurrent/energy/causal/curl evaluator](eval_energy.py)
- [Source-independent Python standard-library audit](audit_receipt.py)
- [Eight source-independent CPU regression tests](tests/test_energy.py)

All source/result artifacts are attached to original [GitHub Actions run #44235767614](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44235767614). The 31055/31056 potential-fitting split is frozen and uploaded before new 31057 scientific testing in separate CI jobs. Repeating the experiment on a now-public 31057 seed is a reproducibility exercise rather than a second independent prospective test.

## Run locally

Use Python 3.11, CPU PyTorch 2.7.1. First retrieve original frozen LGA-02 first-science model+SAE bundle.pt/train.json from run 44222863870 into runs/lga05/original_teachers:

    python -m unittest discover -s mechinterp/loopscope08/tests -p test_energy.py -v
    python -m mechinterp.loopscope08.fit_potential --source-bundle runs/lga05/original_teachers/bundle.pt --source-meta runs/lga05/original_teachers/train.json --potential-bundle runs/lga05/calibration/potentials.pt --potential-meta runs/lga05/calibration/fit.json
    python -m mechinterp.loopscope08.eval_energy --source-bundle runs/lga05/original_teachers/bundle.pt --source-meta runs/lga05/original_teachers/train.json --potential-bundle runs/lga05/calibration/potentials.pt --potential-meta runs/lga05/calibration/fit.json --out runs/lga05/fresh/full.json
    python -m mechinterp.loopscope08.audit_receipt --source-bundle runs/lga05/original_teachers/bundle.pt --source-meta runs/lga05/original_teachers/train.json --potential-bundle runs/lga05/calibration/potentials.pt --potential-meta runs/lga05/calibration/fit.json --test-receipt runs/lga05/fresh/full.json --out runs/lga05/audit/audit.json

The original smoke unit test uses seed61057 and never opens the genuine test seed31057. Exact original first-science receipt SHA256s are pinned in RESULTS.md. A complete reproduced run after inspecting the first source outcomes is an engineering check, without independent heldout significance.

## Interpretability scope

A depth-independent external energy fitted to ORIGINAL native step directions can capture a monotonically descending conservative component along the task-generated state manifold. The original F(h)−h field contains nonzero curl in randomly sampled 81×48 tangent planes, and actual E-descending edits do not reliably improve final-task CE against random. Thus the source results support a learned trajectory-geometry model, with intrinsic internal energy computation and task-general convergence left open.
