# LOOPSCOPE-06: Pre-data engineering correction

18 November 2026. The original LGA-03 prospective protocol was committed at e01429a, before any value fitting or new test data. Initial [GitHub Actions run #44228179946](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44228179946) halted at Python unit-test module import, before the original frozen LGA-02 teacher weights were downloaded, and before training any value function.

The new independent bootstrap auditor exports a helper named seed; the unit-test import initially requested new_seed. The correction aliases that existing helper in the source-independent test. It changes no heldout source, calibration cohort, model/SAE checkpoint, potential architecture, value training rule, physical intervention, bootstrap statistic or original scientific success gate. The failed run remains in Actions provenance. A fresh complete workflow will train and freeze the value model before generating any new test input.
