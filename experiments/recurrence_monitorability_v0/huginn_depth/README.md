# Huginn-0125 accuracy by hop count and recurrence (2026-11-03, exploratory)

`tomg-group-umd/huginn-0125` at `bb6621b6`: a 4-layer core block applied `r` times between a 2-layer prelude
and a 2-layer coda. bf16 weights on the RTX 5080, `bvg-v03` venv (transformers 4.46.3). Load check: greedy
"The capital of France is" continues "Paris." at r = 4 and 32. 96 items per cell; the same items at every `r`.
Script: `research_gym/scripts/rmp_huginn_depth_sweep.py`; plot: `report_huginn`.

## Pointer chase (Nanbeige format): fails on Huginn

- `depth_sweep_fused_arrow.json`: with `A->B`, Huginn's tokenizer makes `>B` one token, so edge targets never
  appear as the ` B` tokens the readout scores. Three-shot one hop is 4–9% at every `r` (chance 6.25%).
- `depth_sweep_chase_spaced.json`: with `A -> B`, three-shot one hop is 17–25% for r ≥ 8, and two hops is
  7–10%. Both runs were stopped early once this was clear.

## Variable-copy chains (`depth_sweep.json`, `depth_sweep.png`)

Each program has three interleaved chains (a digit constant, then `hops` copies), ends with `print(<last
variable of the target chain>)`, and is scored by forced choice over the 10 digit tokens (chance 0.10). Picking
one of the three digits in the program at random scores 0.33.

| three-shot | r = 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| 1 hop | 0.16 | 0.50 | **0.76** | 0.71 | 0.70 |
| 2 hops | 0.12 | 0.34 | **0.47** | 0.41 | 0.39 |
| 3 hops | 0.20 | 0.31 | 0.26 | 0.29 | 0.27 |
| 4–6 hops | 0.12–0.15 | 0.23–0.35 | 0.20–0.26 | 0.22–0.24 | 0.22–0.26 |

| zero-shot | r = 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| 1 hop | 0.35 | 0.55 | **0.69** | 0.61 | 0.64 |
| 2 hops | 0.23 | 0.23 | **0.34** | 0.33 | 0.31 |
| 3–6 hops | 0.23–0.26 | 0.11–0.23 | 0.09–0.19 | 0.08–0.14 | 0.08–0.14 |

- Recurrence buys accuracy up to r = 16 and nothing after it; r = 32 and 64 are slightly worse.
- Only one and two hops are above the pick-a-digit-in-the-program level. From three hops Huginn does not
  chase the chain at any `r`, so more iterations do not buy more hops here.
- Standard error per cell is about 0.05 (n = 96).
- Heat map cell chosen from this: two hops, three-shot, r = 16 (`../huginn_heatmap/`).
