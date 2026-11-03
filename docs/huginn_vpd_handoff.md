# Handoff: VPD fit of Huginn-0125's core block, per-iteration heat map

Everything referenced here is public:

- **This bundle:** `https://huggingface.co/datasets/AlephFunk/nanbeige-vpd-shared-weights/tree/main/rmp_local_2026-11-03`
  (scripts under `research_gym/scripts/`, results under `results/`).
- **VPD library:** `https://github.com/MoralityLabAI/param-decomp` at `2df0092d5be4d8e4748b3c66ef8325dedef17549`
  (Python ≥ 3.12, < 3.14).
- **Model:** `tomg-group-umd/huginn-0125` at revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`.
- **Paper-1 Nanbeige fit,** for reference: same dataset, `prime-20260906T204058Z-561/.../vpd_factors.npz`
  (sha256 `a1acd3ffa1fe0c4f4d2566b5e95670112bb152b31a149c05777b46ee6b4a502c`).

## Prompt

> Fit a VPD (`param-decomp@2df0092`) decomposition of Huginn-0125's core-block MLP output matrices and redo
> the per-iteration ablation heat map with the fitted components. Exploratory, not a registered endpoint.
>
> **What already exists (local RTX 5080, 2026-11-03; read `results/*/README.md`):**
> - Huginn applies a 4-layer core block `r` times between a 2-layer prelude and a 2-layer coda. On
>   variable-copy chains (`rmp_huginn_depth_sweep.make_item(rng, hops, shots, task="varchain")`; forced choice
>   over the 10 digit tokens, chance 0.10), three-shot accuracy peaks at r = 16: 0.76 at one hop, 0.47 at two.
>   From three hops it stays at the pick-a-digit-in-the-program level (≈0.25–0.33) at every r.
> - Heat map with the top 64 *singular directions* of each core `mlp.proj` (17920 → 5280), two hops,
>   three-shot, r = 16, 64 clean-correct items: no component matters when removed in iterations 1–8. The same
>   components removed in iterations 9–16 cost up to 0.77 nats (core layer 3, top direction: accuracy
>   0.94 → 0.42). Usage profiles at iteration 1 vs 16 correlate at 0.07–0.24 in core layers 0–2, 0.65 in layer 3.
> - Caveat to fix: random controls were norm-matched to a random component, not to the ablated one.
>
> **Question:** do real VPD components split by iteration, with some used only in iterations 1–8 and others
> only in 9–16, where singular directions spread the work across both?
>
> **Do:**
> 1. Load Huginn at the pinned revision in bf16 under transformers 4.44–4.46. Its remote code
>    (`raven_modeling_minimal.py`) is written for 4.44; under 5.x, check generation before trusting
>    anything. Pass `num_steps` as a plain int. The recurrent state starts from noise, so reseed before every
>    forward. Load check: greedy "The capital of France is" → "Paris" at r = 4 and r = 32.
> 2. Reproduce the r = 16, two-hop, three-shot clean accuracy (≈0.47 on fresh items, seed 20261103 in the
>    depth sweep) before fitting.
> 3. Fit VPD on `transformer.core_block.{1,3}.mlp.proj` (layers 0 and 2 if budget allows). Use one V/U set per
>    matrix shared across iterations, with per-iteration causal-importance masks, matching the paper-1
>    multipass setup. 64 components per matrix. Fit at r = 16 on generated variable-chain programs (hops 1–3,
>    zero- and three-shot mixed), not on the evaluation items. Train long enough that reconstruction and the
>    faithfulness loss plateau; paper-1's 50-step fits were too short to compare across seeds.
> 4. Heat map with `rmp_huginn_heatmap.py`'s protocol: usage per component × iteration at the answer position;
>    ablate each component in iterations 1–4, 5–8, 9–12, 13–16 and all. Use **each ablated component's own
>    norm** for its random controls (≥ 8 per layer). Evaluate on 64 clean-correct two-hop items (seed
>    20261104), separate from the fit data.
> 5. Report, per layer:
>    - the fraction of components whose ablation effect is confined to one iteration half (below the random
>      5th percentile in one half, inside the band in the other);
>    - the correlation of usage between early and late iterations;
>    - a side-by-side with the SVD baseline (`results/huginn_heatmap/heatmap_summary.json`).
>
> **Budget:** about 2–4 GPU-hours on one 80 GB card. Truncated backprop through the 16 iterations is fine;
> record the truncation.
>
> **Artifacts:** before terminating any machine, upload the factors (`.npz` with V/U per matrix plus a
> manifest of module names, r, seed, steps, losses), the heat map JSON/PNG, logs and the exact config to the
> HF dataset above under `huginn_vpd_<date>/`. Verify the upload by listing the files and checking the factor
> file's sha256 against the local copy. Nothing is kept on a pod.

Published copy: HF dataset `AlephFunk/nanbeige-vpd-shared-weights`, folder `rmp_local_2026-11-03/`, commit `abc2973e78386787f714c7a6788f3d421a93f479` (21 files, sizes verified, readable without login).
