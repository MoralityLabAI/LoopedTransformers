# LGA-10E: archived precision diagnosis and CI preparation

Protocol: [PROTOCOL.md](PROTOCOL.md). This directory currently contains
**phase R** only, an independently auditable retrospective analysis of the
original LGA-10D first-run seed-61067 source. Phase P implementation and
fresh-source generation are intentionally absent.

Run locally after extracting the original GitHub Actions artifact:

    python -m unittest discover -s mechinterp/loopscope13e/tests -p 'test_*.py' -v
    python -m mechinterp.loopscope13d.audit_receipt \
      --receipt runs/lga10d/feasibility.json \
      --raw runs/lga10d/raw.npz --out runs/lga10e/parent_audit.json
    python -m mechinterp.loopscope13e.diagnose \
      --archive-dir runs/lga10d \
      --json-out runs/lga10e/diagnosis.json \
      --csv-out runs/lga10e/edits.csv

The CI workflow downloads the original artifact from run 44363670991,
validates the two pinned SHA-256 digests, reruns the NumPy auditor and
produces both retrospective outputs. GitHub Actions artifact retention is
90 days, so the separately preserved original ZIP remains important.

The signed identity separates the left-hand contraction displacement,
the HVP arithmetic at an identical representable probe, and substituting
that probe for the actual physical edit. Individual components can cancel.
The stored output is a diagnostic; any one-component subtraction is only
an arithmetic counterfactual. There is no claim of an improved FP32 model.

The prospective seed 61068 is a **reserved identifier**, not an opened
source. Protected science seeds remain closed. A source-lock implementation,
algorithm-specific tests and future first-push guard require their own
subsequent protocol-review and code-freeze commit.
