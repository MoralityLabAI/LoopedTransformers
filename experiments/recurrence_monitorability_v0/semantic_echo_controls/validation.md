# Semantic Echo validation

**Date:** 2026-11-11  
**Source baseline:** `642ee1868b3fdd826f8b10f9f105b30ab5147a0a`  
**Machine-readable receipt:** [validation.json](validation.json)

The receipt identifies the exact tested sources by SHA-256. The baseline commit identifies the existing LoopedChase implementation. This validation covers the new metrics and adapter before their introduction to the branch.

## Environment and tests

| Component | Version |
|---|---|
| Python | 3.12.14 |
| NumPy | 2.3.5 |
| PyTorch | 2.15.0+cpu |
| Native smoke arithmetic | CPU, float64, one thread |

All **35 tests passed**, including 24 metric/control tests and 11 adapter tests. The tests exercise endpoint-only selection, exact cycle indexing, undefined denominators, independent-item aggregation, native state and logit parity, scoped interventions, safe checkpoint roundtrips, and rejection of incompatible variable-depth checkpoints.

```bash
python -m unittest discover -s tests -p '*semantic_echo.py' -v
```

All **seven constructed controls passed**. Their full measurements and analytic targets appear in [controls.json](controls.json), with an accompanying [report](report.md). They carry `CONTROL_ONLY` status.

```bash
python -m research_gym.scripts.semantic_echo_controls \
  --out /tmp/semantic_echo_controls
```

## Native model smoke

The smoke used a randomly initialized model and carries `UNTRAINED_SMOKE` status. Reproduce it with:

```bash
python -m research_gym.scripts.loopscope_semantic_echo \
  --smoke --out /tmp/semantic_echo_validation \
  --items 8 --max-loops 12 --horizon 4 --max-pairs 4 \
  --seed 940117 --cpu-threads 1 --device cpu --dtype float64
```

| Check | Observed result |
|---|---:|
| Native state comparisons | 96 passed |
| Native logit comparisons | 96 passed |
| Maximum native state or logit deviation | 0 |
| Selected full-donor replays | 4 passed |
| Maximum full-donor state or logit deviation | 0 |
| Maximum relative state deviation after a LayerNorm shift | 1.615e-16 |
| Independently generated items | 8 |
| Unique permutation values | 6 |
| Structurally eligible visit pairs | 123 |
| Pairs passing endpoint correctness | 11 |
| Items with a defined primary score | 4 |

The primary disagreement mean was zero. All 11 gated pairs shared an error somewhere in their four-step futures, and zero pairs had two fully correct futures. Across individual future steps, oracle agreement was 34/88, or 38.64%. This supplies a concrete execution check for the oracle companion metric: repeated wrong predictions can pass a continuation-agreement test.

The machine-readable receipt retains the smoke configuration, source hashes, native parity results, and each selected replay check. Running the command regenerates the complete token trace and pair-level artifacts.

## Trained checkpoint evaluation

The 24 Q4/Q5 checkpoints were evaluated on 2026-11-11 with the README commands (CUDA, float32). Native parity passed on all 24 and full-donor replay on 384/384 capped pairs. Results and receipts: [SEM-ECHO-01 Q4/Q5 report](../semantic_echo_q4q5_summary/report.md).
