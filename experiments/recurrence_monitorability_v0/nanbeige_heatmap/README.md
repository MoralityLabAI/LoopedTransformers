# Nanbeige per-pass VPD heat map (2026-11-03, exploratory)

Nanbeige4.2-3B at `451ed48c`, two passes over 22 shared layers. Components: the 64-component paper-1 fit
(`vpd_factors.npz`, sha256 `a1acd3ff…`, 50 VPD steps) on `down_proj` of physical layers 12 and 22. Pass 1's
layer 22 ends the first loop; pass 2's layer 22 is the model's last layer.

- **Task:** one-hop pointer chase, three-shot runbook format, forced choice over the 16 node tokens.
  Clean accuracy 123/176; the 120 correct items are used. Two hops is at chance, so it was not mapped.
- **Ablation:** remove one rank-one component (`(x·V_c)·U_c`) on pass 1, pass 2 or both; record the mean
  change in log p(correct). Control: 32 random rank-one directions per layer, norm-matched.
- **Usage:** mean `|x·V_c|·‖U_c‖` at the answer position per pass.

Run: `bvg-v03` venv (transformers 4.46.3, the checkpoint's remote code degrades under 5.x),
`python -m research_gym.scripts.rmp_nanbeige_vpd_heatmap --factors <npz> --per-hop 120 --max-candidates 400
--random 32 --batch 8`, then `report_nanbeige_heatmap` (bluebeam venv) for `heatmap.png` and
`heatmap_summary.json`. The run was stopped five times by vramorama's power guard (dGPU capped for 600 s)
and resumed from the written conditions each time; all 192 conditions completed.

## Results

| | L12 pass 1 | L12 pass 2 | L22 pass 1 | L22 pass 2 |
|---|---|---|---|---|
| most harmful ablation (nats) | −0.004 | −0.002 | −0.098 | −0.019 |
| random 5th percentile | −0.005 | −0.0004 | −0.016 | −0.0016 |
| components below random p5 (of 64) | 0 | 1 | 11 | 9 |
| usage, pass 2 / pass 1 | | 0.74 | | 0.52 |

- **Layer 12:** the components behave like random directions on both passes.
- **Layer 22:** three components (19, 9, 43) cost 0.06–0.10 nats when removed on pass 1 and nothing on pass 2.
  Removing them on both passes costs the same as pass 1 alone (both vs pass1+pass2 correlation 0.99).
  Pass-1 usage predicts pass-1 ablation effect (r = 0.57 against |effect|).
- **Same weights, different components per pass:** usage on pass 1 and pass 2 is uncorrelated (r = 0.05 at
  L12, 0.06 at L22), and so are the ablation effects (r = 0.01, −0.14).
- **Scale:** the largest effect is 0.1 nats against a clean mean log p of −0.46, from a 50-step fit. This is
  a one-hop map; the multi-hop question needs a model that chases hops (Huginn next).
