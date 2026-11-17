# ROC-TRM-01: Sudoku loop-convergence within 6–8 GiB

**Registered 2026-11-17.** This directory begins with a source-locked experiment protocol. It contains **no new trained checkpoint, generated Sudoku examples, Jacobian traces or causal result**.

- [PROTOCOL.md](PROTOCOL.md) defines the complete recurrent state, ROC-01 nulls, exact-rank closure constraints, BPD toy gate, causal interventions, stopping criteria and provenance.
- [CONFIG.json](CONFIG.json) freezes the new seeds, clocks, output heads, threshold family, difficulty strata, process limits and original model SHA256s.
- [test_registration.py](test_registration.py) checks invariant dimensions, seed counts, nonconflation of ROC and BPD nulls, real-checkpoint safeguards, and exact rotation coordinate witnesses.

Source-only checks from repository root:

```bash
python -m json.tool mechinterp/trm_roc01/CONFIG.json > /dev/null
python -m unittest discover -s mechinterp/trm_roc01 -p 'test_registration.py' -v
```

**Stages:** S0 frozen sources and tiny static tests; S1 miniature 6×6 Sudoku training and a complete symbolic baseline; S2 task/rank-matched observer cards, conditional closure and finite causal effects; S3 frozen pretrained 7M checkpoint pilot if the preregistered toy gates pass. The BPD routed toy activation-permutation and gate-precision tests must also pass before S3.

The actual full TRM carries 2×97×512 = **99,328** coordinates. Each 512D cell head has up to 9 native digit directions and 11 whole-vocabulary directions; the full Sudoku output has up to 81×9 = **729** direct dimensions. A rank-9/11 per-cell projection is a *local diagnostic*. The global observer, its 729-row rank ceiling, full recurrent clock and cross-buffer effects are documented in the protocol.

For historical context, see [LOOPSCOPE-02](../loopscope02/README.md), [LOOPSCOPE-03](../loopscope03/README.md) and the [ROC-02A retrospective note](../../experiments/observer_calculus_roc02a_v1/ROC02A_FROZEN_NRF_RANK_CAUSAL.md).

**Capacity rule:** one process; CPU by default with optional 4GB 3050 for miniature models; stop at 6 GiB RSS or 8 GiB process committed/private bytes, whichever comes first. Inspect host memory before launch. This registration does not resume, suspend or stop any Windows process.
