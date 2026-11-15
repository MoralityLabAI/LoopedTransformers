# PEF-08 | Causal coalition localization after PEF-07

This is an independent follow-up branch based on PEF-07 PR #67, preserving its sealed 12-bit task and six 450-update checkpoints. The [PEF-08 protocol](PROTOCOL.md) was committed **before** the new intervention results; it is separate from PEF-07's original registration. The original PEF-07 outcomes were known when PEF-08 was designed.

## What the program measures

For all 15 memory states h0 through h14, inspect all six focal-bit pair four-corner hidden and native-logit differences. Select 8 of 32 channels using only the 48 PEF-07 probe contexts, apply deterministic pair-gated second-order erasure at visits 4, 8 or 12, and continue the frozen memory model on its natural suffix. Compare top-8 with bottom-8 and eight random size-matched masks for every intended pair, each seed and both task arms. The native h12 head and the h14 ridge head are fixed for all postintervention scores.

The [local results](LOCAL_RESULTS.md) reproduced each of the original six 768/768 native h12 and h14 scores and found both coalition pairs had greater targeted interaction attenuation than random-mask mean in **3/3 seeds** at visit 12. Each targeted h12 coalition edit gave 624/768 with exactly three distinct wrong focal patterns per heldout context. The fixed h14 ridge retained 768/768 in five of the six pair interventions. Threshold controls were affected too; these contrasts have meaningful limits as causal identification.

## Running

Needs Python 3.12+ with PyTorch 2.10 CPU and pytest. The five small unit tests do not require downloading a model archive:

```bash
python -m pytest -q research/pef08/test_pef08.py
```

For the full intervention analysis, extract the existing PEF-07 evidence ZIP containing `registered/seed*.frozen.pt`, `registered/sealed_training_ledger.json` and `registered/source_split.json`, and invoke:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  DNNL_MAX_CPU_ISA=AVX2 python -m research.pef08.pef08 \
  --evidence /path/to/registered --out /tmp/pef08
```

The output `summary.json` includes every per-loop interaction diagnostic, checkpoint hash, pair-channel mask, intervention response and randomized control. Checkpoint and source mismatches fail closed. The PEF-07 checkpoints were recovered from the original local evidence ZIP, so they are **not** checked into this PR.

The Python 3.12 GitHub Actions workflow verifies source/intervention invariants only. The full six-model PEF-08 result is local Python 3.13.5 evidence until a runner with the original checkpoint archive replays it.
