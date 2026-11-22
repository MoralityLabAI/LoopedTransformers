# DOSE38 BF16 intervention feasibility by pass

On frozen COMP14 calibration captures, a two-sided search over the input
scale improved the realized 2% physical displacement for phase-2 IO18
writes. The same search left the low-dose phase-1 failures intact at the
focused pointer readout. Each sign must land within 3% of the requested
residual L2 displacement, with direction cosine at least 0.98.

## Focused pointer CAL, 8 roots per precision

The counts are qualified world/direction measurements out of 128,
with 16 worlds and eight fitted directions. Both signs must pass.
Complete roots require qualification for both worlds and all eight
directions.

| Precision | Pass | Requested dose | One-shot qualified | Calibrated qualified | Complete roots, one-shot → calibrated |
|---|---:|---:|---:|---:|---:|
| NF4 | 1 | 1% | 112/128 | 112/128 | 0/8 → 0/8 |
| NF4 | 1 | 2% | 112/128 | 112/128 | 0/8 → 0/8 |
| NF4 | 2 | 1% | 103/128 | 110/128 | 0/8 → 2/8 |
| NF4 | 2 | 2% | 118/128 | 126/128 | 2/8 → 6/8 |
| BF16 | 1 | 1% | 112/128 | 112/128 | 0/8 → 0/8 |
| BF16 | 1 | 2% | 112/128 | 112/128 | 0/8 → 0/8 |
| BF16 | 2 | 1% | 110/128 | 116/128 | 1/8 → 3/8 |
| BF16 | 2 | 2% | 120/128 | 127/128 | 2/8 → 7/8 |

The seven eligible phase-1 writes qualified in both worlds for every
focused CAL root. The first fitted write failed throughout those
conditions, including after scaling. Coordinate 2279 itself still
qualified in **0/16 focused phase-1 worlds in each precision at 1%**.
At 2% it qualified in **3/16 NF4** and **0/16 BF16**. The phase-2
coordinate qualified in **16/16 per precision at 2%**. The fixed
orthonormal control qualifies in all 128 focused directions at each
pass and dose.

The next low-dose CAL intervention can use the seven eligible
phase-1 writes and a separately calibrated phase-2 readout control.
A full eight-write comparison needs a separately registered change
to the allowable phase-1 dose or intervention geometry. A direction
with a discretized or distorted realized edit remains unqualified for
a claimed matched-dose comparison. Delayed exact correctness and
matched shams remain separate experiments.
[READPATH37 direct readout](NAN_READPATH37_RESULTS.md)

The audit exercised 192 CAL dose/state rows per precision, including
copy and erased arms. Peak sampled private RAM was **60.7 MiB** with
at least **7.23 GiB** system RAM available. Zero new forwards and zero
formal tests. The protected ARG-03 test remains sealed.
[Sanitized aggregate](nan_dose38_results.json) ·
[Frozen calibration protocol](../../experiments/nan_dose38/PROTOCOL.md)
