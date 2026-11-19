# LGA-10G: native-state sham-level-set rescue

Prospective protocol: [PROTOCOL.md](PROTOCOL.md). Implementation:
[source61070.py](source61070.py), [sham_repair.py](sham_repair.py),
and independent NumPy [audit61070.py](audit61070.py).

The first source is **61070**, one untrained 44,001-parameter TRM,
eight independent input bases and eight correlated goals per base.
Source and model initialization occur **only on the first guarded
GitHub Actions CPU push**, after the committed protocol and full
preflight tests. Local GPUs are never used for this source experiment.

New sham correction attempts begin only after the frozen original
seven-slice search exhausts a sham, with a hard bound of 36 local
candidate checks per exhausted sham. Exact physically realized
FP32 additions, HVP scalar errors and original source constraints
remain unchanged. All numerical failures and incomplete plans
remain in the 64-case denominator.

The registered 61/64 engineering threshold is independent of CI
integrity. Provenance and full first-run source archives are preserved
without rerunning or altering post-source code.
