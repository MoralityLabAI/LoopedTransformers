# PEF-12 | Probe-only precision calibration

Building on PEF-11, use six frozen PEF-07 checkpoints and the immutable PEF-09/10/11 source-only feature/readout ledgers. The [registered protocol](PROTOCOL.md) fixes two previously selected tangents (random, pair01), angles 0.25° and 4°, three new Gaussian levels, 16 replicates, matched off-plane perturbations and three coordinate-quantization steps. The [results](LOCAL_RESULTS.md) and [mathematical derivation](MATH.md) document the prospective CDF forecast, numerical error, and limitations.

The full study uses:
- \`research/pef12/pef12.py\`, which verifies checkpoint and ledger hashes; computes probe-only h8 input-Jacobian forecasts; freezes their JSON receipt before any heldout states; then scores the fixed readout at all 2,376 perturbation conditions.
- \`research/pef12/audit_pef12.py\`, which independently evaluates the Gaussian CDF and uniform approximation with Python standard-library math, verifies original checkpoint bytes and per-replica score integrity, and reproduces the frozen decision.
- \`research/pef12/test_pef12.py\`, nine model/algebra/seed-schedule unit tests requiring no checkpoint archive.

The original PEF-07 checkpoints are retained outside GitHub. The local evidence ZIP includes checkpoint binaries, frozen ledgers, scripts, complete raw results, reports and SHA256 manifest.

\`\`\`bash
python -m pytest -q research/pef12/test_pef12.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DNNL_MAX_CPU_ISA=AVX2 \
 python -m research.pef12.pef12 --evidence /path/to/registered \
 --selection-ledger /path/to/pef09_results/frozen_selection_ledger.json \
 --angle-ledger /path/to/pef10_results/frozen_angle_and_ridge_ledger.json \
 --precision-ledger /path/to/pef11_results/frozen_precision_ledger.json \
 --out /tmp/pef12
python -m research.pef12.audit_pef12 \
 --evidence-root /path/to/extracted-evidence-root --out /tmp/pef12
\`\`\`

Local Python 3.13/PyTorch 2.10 CPU: 9/9 new unit tests, 35/35 combined tests, and independent receipt audit passed, with three coalition seeds meeting the registered calibration threshold. Hosted Python 3.12 verification is separate. No experiment changes the original trained weights.