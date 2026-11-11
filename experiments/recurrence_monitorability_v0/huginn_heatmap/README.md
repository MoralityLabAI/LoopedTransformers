# Huginn-0125 per-iteration component heat map (2026-11-03, exploratory)

Two-hop variable-copy chains, three-shot, r = 16 (the best cell of `../huginn_depth/`). 64 items the clean
model answers correctly (64/134 tried); clean mean log p(correct) −0.81 under the 10-digit forced choice.
Script `research_gym/scripts/rmp_huginn_heatmap.py --hops 2 --shots 3 --steps 16`, plot `report_huginn`.

- **Components:** no VPD fit exists for Huginn, so each core layer's `mlp.proj` (17920 → 5280) is split into
  its top 64 singular directions, `(x·Vh_c)·S_c·U_c`.
- **Usage:** mean `|x·Vh_c|·S_c` at the answer position, per core layer and iteration (left panels).
- **Ablation:** the 16 most-used components per layer, removed in iterations 1–4, 5–8, 9–12, 13–16 or all;
  mean change in log p(correct) (right panels). Control: 8 random rank-one directions per layer,
  norm-matched to a random one of the 64 components.
- **Accuracy caveat:** items were selected one at a time; the scored runs batch them, and the recurrent state
  starts from seeded noise per batch, so clean batched accuracy is about 0.9, not 1.0. Changes are measured
  against the clean run with the same batches and seed.

## Results

| most harmful ablation (nats) | it 1–4 | 5–8 | 9–12 | 13–16 | all | random min (all) |
|---|---|---|---|---|---|---|
| core 0 | −0.00 | −0.01 | −0.02 | −0.01 | −0.03 | −0.005 |
| core 1 | −0.01 | −0.00 | −0.04 | **−0.11** | −0.11 | 0.000 |
| core 2 | −0.00 | −0.00 | −0.03 | −0.03 | −0.03 | −0.004 |
| core 3 | −0.01 | −0.03 | **−0.47** | **−0.77** | −0.70 | −0.001 |

- **Only late iterations are load-bearing.** No component matters in iterations 1–8 in any layer. The same
  components removed in iterations 9–16 cost up to 0.77 nats. Core 3's top direction (c0) drops accuracy
  from 0.94 to 0.42 when removed in iterations 13–16 alone; c1, c4 and c49 cost 0.18–0.30 nats.
  One reading: early-iteration damage is repaired by later iterations, late damage is not.
- **The same weights, used differently by iteration.** Usage of the 64 components at iteration 1 and at
  iteration 16 correlates at r = 0.24, 0.16 and 0.07 in core layers 0–2, and 0.65 in core 3. Usage is highest
  at iteration 1 (the state starts from noise) and settles after about 6 iterations.
- **Layer 3 dominates,** and its top singular directions carry the effect.
- **Control caveat:** random directions are norm-matched to a random component, not to the ablated one, and
  top singular directions have the largest norms. Part of the c0/c1 effect may be size. The iteration
  pattern does not depend on this: the same component costs nothing in iterations 1–8.
