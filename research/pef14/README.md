# PEF-14 | Fresh trained contexts and conditional phase-mixture calibration

[Registered protocol](PROTOCOL.md) committed before training at `e63ad7c` follows the PEF-13 negative result without changing it. [Local results](LOCAL_RESULTS.md) include all six independent seeds and task controls; [mathematical note](MATH.md) gives a total-variation bias bound and concentration statement under explicit iid-context assumptions.

The 12-bit context split uses **a new salt** with 160/48/48 complete 16-pattern source groups. Six recurrent attention trunks train from scratch at seeds 211,223,239 for the original coalition and threshold tasks; all six checkpoints freeze before probe fitting. The new estimator learns a per-focal-pattern empirical phase distribution by excluding the same probe context, then mixes it with a fixed uniform-phase prior. The complete probe forecast ledger freezes before any heldout inference.

The registered criterion passed in 2 of 3 fresh coalition seeds, while 90%-labeled interval coverage was poor. The independent audit verified all six frozen weights and recomputed 96 forecasts. [Posthoc diagnostic](diagnose_pef14.py) separates same-input linearization error from source-to-heldout calibration shift.

From the repository root, with Python 3.12+ and pinned CPU PyTorch 2.10:

```sh
python -m pytest -q research/pef14/test_pef14.py
DNNL_MAX_CPU_ISA=AVX2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONHASHSEED=0 python -m research.pef14.pef14 --out /tmp/pef14
python -m research.pef14.audit_pef14 --out /tmp/pef14
python -m research.pef14.diagnose_pef14 --out /tmp/pef14
```

**Training resumability:** if the experiment is interrupted after completing one or more models, the runner reuses an existing checkpoint only after verifying its seed/task/step metadata, final parameter SHA256 and all optimizer parameter step counts of exactly 450. Its unavailable historical training-loss scalars are marked null rather than guessed. For pristine bit-level reproduction, use an empty directory. All checkpoint hashes in the local receipt can be checked before reuse.

The GitHub workflow has an isolated Python 3.12 unit job (green on Actions run 43721689601) and a separate fresh six-model CPU replication job that reruns training, freezes predictions and uploads its own receipts. The independent hosted cross-host check passed on [Actions run 43722118391](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43722118391) under Python 3.12. All six training checkpoint byte hashes and all 96 heldout integer scores match the local Python 3.13 run. The complete hosted artifact and numeric comparison are documented in [HOSTED_REPLICATION.md](HOSTED_REPLICATION.md).
