# LOOPSCOPE-01

Which loop-level statistics of a looped transformer predict whether it extrapolates past its training depth?
Every question was registered before it was computed; the registration and its dated amendments are in
[docs/registration.md](docs/registration.md), and the full write-up is [docs/interim_report.md](docs/interim_report.md).

## Setup

24 frozen LoopedChase toy loopers (d = 256, tied 2-layer core, h_{t+1} = core(LN(h_t) + e)) trained to answer a
k-hop pointer chase after k loops, k ≤ 8. Arms: per-loop supervision weight λ ∈ {0, 0.01, 0.1, 1} × seeds 1–3 (Q4)
plus λ ∈ {0, 0.01} × seeds 4–9 (Q5). An arm extrapolates if held-out depth-9..16 accuracy is ≥ 0.90: 17 do, 7 fail.
AUC below is for predicting failure.

## Results

| Q | statistic | AUC | registered verdict |
|---|---|---|---|
| Q4/Q5 | F1/F8, first-order fragility of loop 1 over loop 8 (TCA) | 0.950 | Q5 prospective rule weak, 9/12, no false alarms |
| Q7 | contraction γ restricted to the top-16 PCA subspace | 0.824 | weaker than F1/F8; every arm contracts inside the subspace while the full Jacobian expands |
| Q8 | drift out of the depth-8 subspace / contraction beyond depth 8 | 0.882 / 0.840 | weak; two failure modes: schedule desync (5 arms) and contraction loss (2 arms) |
| Q9 | state memory / loop-clock decodability | 0.908 / 0.958 | supported; F1/F8 tracks loop non-stationarity; ablating the clock subspace does not reproduce failure |
| Q11 | compactification error Ê: hop belief vs the exact permutation advance at loops 4–7 | **0.992** | weak by rule (mode match 5/7, needs 6); bound holds in 17/17 extrapolators; phase obstruction isolates all 5 desync arms; shuffled-permutation control 0.071 |

Q1–Q3 (Huginn-0125, Lean-CGRT) and Q6 (transfer to Lean-CGRT loopers) are in the report. Q10 (a depth-32 cohort,
edge-clock causality, semantic echo) is a separate line of work registered by another session under the same
registration file; its code and results stay on branch `rmp-v0-phase2`.

Numbering note: the Q11 block in the registration is headed "Amendment 10 (2026-11-13)" because it was appended
before the clash with the earlier amendment 10 was noticed; the note that follows it renumbers it to 11.

## Layout

- `scripts/`: `loopscope_toy.py` (model, data, training), `loopscope_toy_tca.py` (TCA), `loopscope_toy_q7.py`,
  `_q8.py`, `_q9.py`, `_q11.py` (one per question; each has `run` and `score` subcommands).
- `tests/`: unit tests for each script.
- `results/`: per-arm JSON and `qN_report.{md,json}` for Q7–Q11, the Q5 report, the toy training report, and the TCA report.

## Reproduce

Run from the repo root. The checkpoints (`experiments/recurrence_monitorability_v0/loopscope_toy*/ckpt/`) are not
in Git; train them with `python -m mechinterp.loopscope01.scripts.loopscope_toy train` or copy them in.

```bash
python -m pytest mechinterp/loopscope01/tests -q
python -m mechinterp.loopscope01.scripts.loopscope_toy_q11 run --device cuda
python -m mechinterp.loopscope01.scripts.loopscope_toy_q11 score
```

Provenance: snapshot of branch `rmp-v0-phase2` at 8092ba4, where development continues under `research_gym/scripts/`.
