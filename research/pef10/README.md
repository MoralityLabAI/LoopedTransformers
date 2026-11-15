# PEF-10 | Frozen angle sweep after a dominant-PC ablation

[PROTOCOL.md](PROTOCOL.md) was committed alone at `de6666c71b65287df56b24cb0d57e0e89ff7553c` before PEF-10 scoring. This follow-up builds on PEF-09/09R without changing their experiments, the PEF-07 source split or any model parameters.

At recurrent visit 8, rotate the frozen PEF-09 global PC toward two independently selected pair-SVD residual directions, the probe PCA-2 tangent or one seeded random orthogonal tangent. Project each entire 16-mask context cube with the same rank-one affine projector. Replay natural steps through 12 and the fixed continuation (0,1) to 14. The original native head and original h14 ridge stay fixed; a new ridge is fitted **only from ablated probe states** for h12 and h14.

Per trained model: four tangents, 11 fixed angles (endpoints deduplicated; both signs otherwise) and two pair-specific original PEF-09 directions give 84 conditions. Six models give **504 scored conditions**. The code writes every direction and newly fitted ridge to `frozen_angle_and_ridge_ledger.json` before heldout scoring. The heldout set contains 48 new context groups, 768 histories.

The local [results report](LOCAL_RESULTS.md) records 3/3 coalition seeds reaching the preregistered geometric recovery criterion. The pair-specificity criterion fails 0/3 because the random orthogonal tangent can recover at 0.25° in every seed. This is evidence for a sharp, readout-dependent geometry rather than a unique pair-specific recovery mechanism. [GEOMETRY.md](GEOMETRY.md) derives the exact two-dimensional rotation identity.

Run the five math/control unit tests without archives:

```bash
python -m pytest -q research/pef10/test_pef10.py
```

For full replay, extract the already-published PEF-09 local evidence package and run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
DNNL_MAX_CPU_ISA=AVX2 python -m research.pef10.pef10 \
  --evidence /path/to/registered \
  --selection-ledger /path/to/pef09_results/frozen_selection_ledger.json \
  --out /tmp/pef10
```

The PEF-07 six checkpoints and PEF-09 selection ledger are external immutable research artifacts, not retrained dependencies. The independent GitHub Actions workflow verifies invariant tests on Python 3.12 CPU; full evidence replay additionally requires the checkpoint archive and, as of the local run, has been performed on Python 3.13.5 with PyTorch 2.10.0+cpu.
