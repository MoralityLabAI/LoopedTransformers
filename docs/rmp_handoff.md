# RMP v0 handoff (2026-11-01)

This is the compact entry point for the Recurrence Monitorability Program. Everything listed here is on
branch `rmp-v0-phase2` of `MoralityLabAI/LoopedTransformers`, which is private.

## Where things are

| What | Path |
| --- | --- |
| Program spec | `docs/recurrence_monitorability_program_v0.md` |
| Plans (registrations, with dated change logs) | `docs/rmp_phase1_plan.md`, `docs/rmp_phase2_plan.md`, `docs/rmp_phase2_e5_e6_plan.md` |
| Checkpoint reports | `reports/checkpoint_rmp_phase2_e4.md`, `reports/checkpoint_rmp_phase2_e5_e6.md` |
| Paper ("Shared Weights, Different Roles II") | `papers/rmp_looped_mechinterp/main.tex`, `main.pdf` |
| Technical note (markdown) | `reports/technical_note_looped_mechinterp.md` |
| Result JSON | `experiments/recurrence_monitorability_v0/phase2/*.json` |
| Run records | `experiments/recurrence_monitorability_v0/runs` (seeds 1–3), `.../phase3/runs` (seeds 4–6) |
| Analysis scripts | `research_gym/scripts/rmp_phase2_*.py`; model registry `research_gym/interp/phase2.py` (`MODELS`) |
| Figures | `research_gym/scripts/report_rmp_paper_figures.py` writes to `papers/rmp_looped_mechinterp/figures` |
| Silico scale-up runbook | `docs/silico_rmp_scaleup_runbook.md` |
| Heavy data, not in git | `C:\Users\patri\Documents\Codex\rmp_v0`: `artifacts` (checkpoints, dictionaries), `queue` |
| GPU queue | `scripts/run_rmp_queue.sh`, `scripts/rmp_queue_functions.sh`, `scripts/relaunch_rmp_queue.ps1`, `scripts/rmp_queue_watchdog.ps1` |

## Models

- **LOOP-T:** one block applied 8 times, 0.46M parameters.
- **FF-U:** 8 untied blocks, 1.84M parameters.
- **Gate G0:** 0.98 exact match on all four families. LOOP-T passes it for seeds 1–6 (6/6). FF-U passes for
  s3, s4 and s6 (3/6).

## Results

- **N1 (shared dictionary in the closed loop):**
  - The tied seeds are `faithful_state`, retaining 0.900–0.923 of deep exact match. Seed 1 beats PCA by
    narrow margins.
  - FF-U s3 retains 0.688, below PCA (0.796) and noise (0.778).
- **Transcoders:**
  - The whole-block, position-local transcoder is ill-posed, because attention is 30–42% of update variance.
  - MLP-only transcoders retain 0.94–0.996 of clean deep accuracy on 9 models.
- **E5 DEF:** features pass in 63%, 91% and 78% of eligible classes on the tied seeds. Median set size is 4–8.
- **H3 (description length):** `unresolved`. H2: not supported. Rule ablation: diffuse.
- **N2 / query block:**
  - Early hops (1–2) transfer through the query block in 5 of 6 tied seeds. FF-U s4's transfer is start
    copying.
  - Collision rate is 5.2–6.9% (`query_collision.json`).
  - Rank-32 DAS at the answer position reaches 0.95–0.99.
- **Successor test (registered 2026-10-31, `feature_link.json`):** at most 32 `D-shared` features carry the
  named node.
  - Tied: 4 of 5 eligible seeds pass.
  - Untied: 0 of 2 pass. The untied dictionaries are 4–8× wider, which confounds the comparison.

## Rules that bind future work

- **G0:** do not change it (visits, depths, the 0.98 threshold) without asking the user.
- **Registrations:** do not edit sealed artifacts or existing registrations. Record changes in a plan's
  dated Changes list *before* the run, and label post-hoc work as such.
- **Power and resources:** train on mains power only. When short of VRAM, wait. Never stop another
  session's GPU process, and never change the GPU power cap.
- **Python:** use `C:\Users\patri\.venvs\bluebeam\Scripts\python.exe`, read-only; install nothing.
- **Paper build:** `latexmk -pdf main.tex` with MiKTeX
  (`C:\Users\patri\AppData\Local\Programs\MiKTeX\miktex\bin\x64`). Check written `.tex` for `\r` and for
  collapsed `\\` after scripted edits.

## Open

- **Silico scale-up:** follow the runbook on `Nanbeige/Nanbeige4.2-3B` at revision `451ed48c`, with
  `tomg-group-umd/huginn-0125` second.
  - **2026-11-02, Stage 1 failed the gate.** Silico experiment `exp_01m6dqy2dtehvtp4932aaqy9gy`: the best
    permitted setting (three-shot, two passes) scored 131/2,000 at depths 3–6 (6.55%), against the 0.85
    gate.
  - Load, determinism and capture/replay all passed.
  - Chance with 16 nodes is 6.25%. The same three-shot setting scores 79.2% at one hop, so first tokens are
    valid letters and the failure is depth: two passes without written reasoning do not chase 3 or more
    hops. Next step: `huginn-0125` Stage 1, recorded as an amendment, because the runbook's fallback only
    covered a load failure.
- **2026-11-03, local Nanbeige per-pass heat map** (`experiments/recurrence_monitorability_v0/nanbeige_heatmap/`):
  one hop only. Layer 12 components look like random directions; at layer 22 three components matter on pass 1
  only (up to 0.1 nats), and per-pass usage is uncorrelated between passes. Next: Huginn depth sweep, then an
  SVD-component heat map; a Huginn VPD fit is the Silico candidate.
- **Confirmatory version:**
  - Frozen cells on a sealed split.
  - DEF-selected feature sets, not code-difference ranking, for the feature link.
  - Untied dictionaries at a matched width budget.

## Paper-1 VPD fit (for the Silico add-on)

- **Code:** `MoralityLabAI/VPD` (private) at `69509c3`, script `experiments/nanbeige_looped/run_multipass.py`.
  It uses the vendored `upstream/goodfire/param_decomp`. Settings:
  - `down_proj` at zero-based layers 11 and 21, which are physical layers 12 and 22;
  - one V/U set per matrix, with per-pass intervention masks.
- **Fit:** pod run `prime-20260906T204058Z-561`, 64 components. Its output
  `retrieved/artifacts/nanbeige_looped/bf16_multipass_calibration/vpd_factors.npz` is 6,566,619 bytes, sha256
  `a1acd3ffa1fe0c4f4d2566b5e95670112bb152b31a149c05777b46ee6b4a502c`. The 20 `effect_*.npz` files are
  listed with checksums in the VPD repo's `artifacts/nanbeige_reasoning/dataset_index_v2/assets.json`.
- **Location:** public HF dataset `AlephFunk/nanbeige-vpd-shared-weights` (791 files, 3.1 GB, every Prime run
  directory from 6 to 8 September). The 64-component file is at
  `prime-20260906T204058Z-561/retrieved/artifacts/nanbeige_looped/bf16_multipass_calibration/vpd_factors.npz`.
  Its sha256 matches the index, checked 2026-11-02. Keys `model.layers.{11,21}.mlp.down_proj__V` are
  (10752, 64) and `__U` are (64, 3072), float32.
