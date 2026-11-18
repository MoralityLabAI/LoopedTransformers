# LOOPSCOPE-06: Horizon-conditioned recurrent value potentials

LGA-03 probes whether a separately trained, depth-conditioned scalar loss predictor can recover the **terminal four-pass loss gradient** of a frozen 44,001-parameter weight-tied TinyRecursiveModel. The original local-head gradient alignment was positive in LGA-01 (3/3 seeds), while direct terminal-gradient alignment failed its LGA-02 criterion (1/3). LGA-03 tests an external potential, its original random-initialization control and the true current-head gradient on a fresh source cohort.

- [Prospective preregistration](PROTOCOL.md), committed before calibration or fresh test.
- [Complete registered results, source hashes and limitations](RESULTS.md).
- [Calibration-only horizon value fitter](fit_value.py).
- [Fresh true terminal-gradient and matched native causal evaluator](eval_value.py).
- [Independent Python standard-library receipt auditor](audit_receipt.py).
- [Seven source-independent CPU tests](tests/test_horizon_value.py).
- [Pre-data import-only engineering amendment](ENGINEERING_AMENDMENT.md).

**Registered result: FAIL, 0/3 independent frozen teacher seeds.** The learned value gradients had a lower cosine with the true terminal gradient than the local current-head gradients at every seed and depth, and often worse scalar loss prediction than a constant old calibration mean. Equal-dose interventions along the negative learned-value gradient showed no nominal benefit against random, with four of six contrasts nominally worse.

Original full first-science [GitHub Actions #44228381515](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44228381515) passed all stages, source-locked fit and fresh-test artifacts, and independent stdlib receipt audit. The original value-fitting source split is 31048/31049; first unseen 256-string native test is seed 31050. No model or critic is fitted on seed 31050. The preceding original teacher and train-only SAE bundle from LGA-02 remains frozen.

Reproduce using Python 3.11+ and PyTorch 2.7.1 CPU, from repo root, with source bundle.pt and train.json downloaded from LGA-02 first-science run #44222863870:

    python -m unittest discover -s mechinterp/loopscope06/tests -p test_horizon_value.py -v
    python -m mechinterp.loopscope06.fit_value --source-bundle runs/lga03/old_source/bundle.pt --source-meta runs/lga03/old_source/train.json --value-bundle runs/lga03/value/value.pt --value-receipt runs/lga03/value/calibration.json
    python -m mechinterp.loopscope06.eval_value --source-bundle runs/lga03/old_source/bundle.pt --source-meta runs/lga03/old_source/train.json --value-bundle runs/lga03/value/value.pt --value-receipt runs/lga03/value/calibration.json --out runs/lga03/test/full.json
    python -m mechinterp.loopscope06.audit_receipt --source-bundle runs/lga03/old_source/bundle.pt --source-meta runs/lga03/old_source/train.json --value-bundle runs/lga03/value/value.pt --value-receipt runs/lga03/value/calibration.json --test-receipt runs/lga03/test/full.json --out runs/lga03/audit/audit.json

The independently archived first-science receipts can be checked without re-running any model. Repeating the scientific evaluation on the public 31050 seed provides engineering reproduction, not a second prospective heldout test.

A useful subsequent experiment would fit a gradient/Sobolev-matched value critic using *only* old calibration teacher derivatives, then test on newly registered source-disjoint sequences. The result would remain evidence about an EXTERNAL observer unless the original tied recurrence is separately shown to follow that potential.
