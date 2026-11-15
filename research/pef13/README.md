# PEF-13 | Phase-aware quantization forecast

A prospective continuation of PEF-12 with a **negative registered result**: 0/3 coalition seeds met the precommitted phase-aware calibration criterion. Probe-only exact rounding phases and frozen PEF-12 input Jacobians did not consistently outperform the existing uniform-phase predictor across four new lattice steps and 48 heldout context groups.

* [Preregistered protocol](PROTOCOL.md): committed separately at `9483da079c3707711b071efa9dbd04fa1a286b40`, before PEF-13 heldout scoring.
* [Local run and controls](LOCAL_RESULTS.md): all six trained task arms, 96 new quantization measurements, independent audit and explicitly posthoc geometry diagnostics.
* [Mathematics](MATH.md): conditional curvature theorem and exact probe/heldout error decomposition.

The full benchmark uses six original PEF-07 450-step checkpoints and the exact PEF-09/10/11/12 sealed probe ledgers. Those binary artifacts are kept in the downloadable evidence ZIP, not committed to git. No model weights or ridge heads are updated.

## Run unit tests

From repository root, Python 3.12+, PyTorch 2.10 CPU, pytest:

```sh
python -m pytest -q research/pef13/test_pef13.py
```

## Full replay

Unpack the previous PEF-12 evidence bundle, which includes the six original checkpoints and frozen ledgers. Run:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 DNNL_MAX_CPU_ISA=AVX2
python -m research.pef13.pef13 \
  --evidence /path/to/registered \
  --selection-ledger /path/to/pef09_results/frozen_selection_ledger.json \
  --angle-ledger /path/to/pef10_results/frozen_angle_and_ridge_ledger.json \
  --precision-ledger /path/to/pef11_results/frozen_precision_ledger.json \
  --jacobian-ledger /path/to/pef12_results/sealed_probe_forecasts.json \
  --out /tmp/pef13_results
```

The `sealed_phase_forecasts.json` file is written and SHA256-sealed **before** the experiment accesses heldout memories. Follow with an independent math-and-source auditor:

```sh
python -m research.pef13.audit_pef13 \
  --forecast /tmp/pef13_results/sealed_phase_forecasts.json \
  --summary /tmp/pef13_results/summary.json \
  --evidence /path/to/registered \
  --out /tmp/pef13_results/independent_audit.json
```

The **posthoc** diagnostic, run only after scoring, uses heldout Jacobians to separate linearization error from transfer error; it cannot change the registered result:

```sh
python -m research.pef13.diagnose_pef13 \
  --evidence /path/to/registered \
  --selection-ledger /path/to/pef09_results/frozen_selection_ledger.json \
  --angle-ledger /path/to/pef10_results/frozen_angle_and_ridge_ledger.json \
  --precision-ledger /path/to/pef11_results/frozen_precision_ledger.json \
  --jacobian-ledger /path/to/pef12_results/sealed_probe_forecasts.json \
  --forecast /tmp/pef13_results/sealed_phase_forecasts.json \
  --summary /tmp/pef13_results/summary.json \
  --out /tmp/pef13_results
```

GitHub Actions Python 3.12 source-only tests verify algebraic and data-independent invariants. They do **not** certify a full reproduction without the frozen archive, and passing unit tests would not change the negative registered outcome.