# Silico runbook: mechanistic tests of recurrent depth at LM scale

Author: Patrick Dugan, Morality Lab. Written 2026-10-31 for execution on Goodfire Silico by a planner/worker
model. Source study: *Shared Weights, Different Roles II* (branch `rmp-v0-phase2`,
`papers/rmp_looped_mechinterp/`), run on 0.46M-parameter models. This runbook repeats its five tests on a
pretrained recurrent-depth LM.

**How to use this file.**
- Execute stages in order.
- Every rule marked **FIXED** is a decision rule. Do not change it after seeing data. If a rule cannot be
  applied, stop and write the reason into `CHANGES.md` before continuing. Never edit a rule silently.
- Spend compute only on the listed jobs.
- Every stage ends by writing its JSON outputs and a short `stage_N_report.md`.

---

## 0. What the study already established (small models), and what scale-up must answer

| Test | Small-model result | Question at LM scale |
| --- | --- | --- |
| Closed-loop shared dictionary (N1) | Tied loop keeps 0.90–0.92 of deep EM when its state is replaced by the SAE reconstruction after every visit, matching or beating matched-error PCA. Untied stack fails. | Does one SAE shared across all recurrences survive closed-loop use in a 3B recurrent model? |
| Operation surrogate | Whole-block, position-local transcoders fail (attention is 30–42% of the update). MLP-only transcoders with attention live keep 0.94–0.996. | Same, for the recurrent block's MLPs. |
| Mediation (DEF) | 4–8 SAE features restore 90% of a step's answer effect, beating random/PCA/neuron/orthogonal controls; transcoder latents and heads do not. | Same at the answer position per recurrence. |
| Named variable | The intermediate node of a pointer chase is causally transferable mid-loop only for hops 1–2, only when query tokens plus the answer position are patched, and only at rank 32–128. The final answer sits in a rank-32 answer-position subspace. Start-copy collisions 5–7%. | Where, when and at what rank does `cur_node(s)` live in an LM, and do sparse features carry it? |
| Counter | No causally used visit counter. | Same. |

---

## 1. Model choice (FIXED order of preference)

1. **Primary: `tomg-group-umd/huginn-0125`** (Geiping et al. 2025, about 3.5B parameters). It is a true
   recurrent-depth model: a prelude, a recurrent core block iterated `r` times (`r` set at inference), and a
   coda. Use it if Silico supports it.
2. **Secondary: Nanbeige4.2-3B** at the pinned revision `451ed48c3273ecef7ea8faaa43c31ce529763bb1`
   (two passes over 22 layers). It continues the first paper. Use it if Huginn is unavailable, or as a
   second model only after Stage 5 finishes on the primary.
3. **If neither is supported:** stop and report which models Silico supports. Do not substitute a
   non-recurrent model.

**Terminology.**
- An *iteration* is one application of the recurrent core (Huginn) or one pass (Nanbeige).
- `state[t]` is the residual stream leaving iteration `t`, at every position.
- For Huginn, read the code to find where the latent state `s_t` is updated and record it there.
- Write the exact hook points into `MODEL_NOTES.md` before Stage 1.

---

## 2. Task: text pointer chasing with exact traces

**Generator (FIXED).**
- Nodes are 16 labels that each tokenize to a single token for the chosen tokenizer.
  - Test candidates: capital letters `A`–`P`, or two-digit numbers `10`–`25`.
  - Check every label is one token with a leading space, and record the choice.
- The successor map `f` is a uniformly random permutation of the 16 nodes. Pick the start node so the
  requested walk does not revisit a node, i.e. it is a simple path, as in the small study.
- Prompt template, exactly:

  ```text
  Edges: A->K, B->D, ... (all 16, in random order)
  Start: C
  Hops: 4
  Answer:
  ```

  The answer is a single node token after `Answer:`. Score exact match on the first generated token
  (greedy).
- Depths `K` run from 1 to 8.
- Variables: `cur_node(s) = f^s(start)` for `s = 0..K`, and the answer is `cur_node(K)`.
- Splits, by generator seed:
  - `train_states`: 20,000 prompts, for SAE and transcoder training only;
  - `selection`: 4,000 prompts, for every choice;
  - `report`: 4,000 prompts, for every endpoint;
  - `sealed`: 4,000 prompts, generated now, hashed, not read until Stage 7.

  Store each prompt with `start`, `f`, `K` and the full trace.
- **Optional few-shot variant.** Prepend 3 solved examples if zero-shot accuracy is too low. Choose the
  variant in Stage 1 only, by the gate below.

**Accuracy gate (Stage 1, FIXED).**
1. On `selection`, measure EM by `K` and by iteration count (Huginn: `r` in {4, 8, 16, 32, 64}).
2. Choose the smallest `r*` at which mean EM over `K` = 3..6 is at least 0.85.
3. Define the analysed depths `D* = {K : EM(K, r*) >= 0.80}`. This must include at least 3 values.
4. If no `r*` exists, try the few-shot variant. If that also fails, stop: the task is too hard for the
   model.

All later stages use `r*` and `D*`.

---

## 3. Stage 1: feasibility (budget about 2 GPU-hours)

1. Load the model in bf16 and pin the revision hash.
2. Write `MODEL_NOTES.md`: architecture, hook points, hidden size `H`, layers in the core, heads, and
   whether the core MLP is gated (SwiGLU).
3. **Determinism check.** Run the same 64 prompts twice. Logits must match bitwise, or to 1e-3 max abs if
   kernels are nondeterministic. Record which.
4. **Capture check.** For 32 prompts at `r*`, record `state[t]` at all positions for every `t`. Also
   record, for every core layer, the attention output, the MLP input (post-norm) and the MLP output.
   Verify that recomputing the forward pass from recorded parts reproduces the logits.
5. Run the accuracy gate (Section 2) and write `stage_1_report.md`.
6. **Cost receipt.** Wall time and GPU-hours for steps 3 to 5. Extrapolate to Stages 2 to 6. If the total
   exceeds the budget table (Section 10) by more than 2x, stop and report.

---

## 4. Stage 2: a closed-loop shared dictionary (N1) (budget about 12 GPU-hours)

**SAE (FIXED design).**
- TopK SAE on RMS-normalized `state[t]`, pooled over all iterations `t = 1..r*` and all positions.
- Widths {4H, 8H, 16H} (rounded to a multiple of 1024), with `k` in {32, 64, 128}.
- Adam, lr 3e-4, batch 4096 vectors, 50M training vectors drawn fresh from `train_states`.
- Decoder rows at unit norm. Dead-latent auxiliary loss (AuxK, k_aux = 512, weight 1/32), and
  pre-encoder bias initialized at the data mean.
- Seed 0. Seeds 1 and 2 only for the selected configuration (Stage 4 control).
- If Silico's managed SAE trainer offers TopK with these settings, use it. Record any setting it cannot
  match in `CHANGES.md` before training.

**Selection rule (FIXED).**
- Open-loop CE recovered at iteration `t`: replace `state[t]` by its reconstruction at that one iteration,
  run on, and compute `(CE_ablate − CE_spliced)/(CE_ablate − CE_clean)` on the answer token. The ablation
  replaces `state[t]` by its per-position mean over `train_states`.
- Selected size: the smallest width, then the smallest `k`, with CE recovered at least 0.95 at every
  iteration. If none qualifies, take the configuration with the highest minimum.

**Closed-loop test (FIXED).** Measure EM on `report` at depths `D*`, relative to clean, for each
configuration in each condition:
- *closed-loop:* replace `state[t]` by its reconstruction after every iteration;
- *PCA:* project onto the top-`p` principal components of `train_states`, where `p` matches the SAE's MSE
  on `selection`;
- *noise:* add isotropic Gaussian noise whose norm equals that state's reconstruction-error norm.

**Labels (FIXED).**
- `faithful_state`: some configuration has closed-loop at 0.90 or more of clean **and** strictly beats its
  PCA control. Report the margin in examples.
- `open_loop_only`: every configuration with open-loop CE recovered of 0.95 or more has closed-loop at
  0.50 or less, while noise stays at 0.90 or more.
- `fragile_dynamics`: as `open_loop_only`, but noise also falls below 0.90.
- `unresolved`: anything else.

**Outputs.**
- `n1.json`: per configuration, the open-loop CE by iteration, closed-loop EM, PCA EM, noise EM and `p`.
- `dictionaries.json`: paths, hashes, FVU and dead latents.

---

## 5. Stage 3: operation surrogates (budget about 10 GPU-hours)

**MLP-only transcoders (FIXED).**
- For every core layer `l`, train a TopK transcoder from the MLP input (post-norm) to the MLP output,
  shared across all iterations, with a linear skip term.
- Width 8H, `k` = 64, the same training recipe as the SAE, 50M vectors.
- If the MLP is gated, keep the gate inside the black box: the transcoder replaces the whole MLP sublayer.

**Replacement model.** Replace every core-layer MLP at every iteration by its transcoder, keeping attention
live. Report EM on `report` at `D*`, relative to clean.

**Negative control (cheap).** One position-local whole-core transcoder from `state[t−1]` to
`state[t] − state[t−1]`, at the same width and `k`. Report its FVU and replacement EM. The small study
predicts collapse.

**Attention share.** For each core layer, report the share of update variance carried by the attention
term, in the transcoder's units.

**Description length (FIXED accounting).** Count each of these separately; never only their sum:
- alive latents;
- mean active latents per position and iteration times `log2(alive)`;
- skip-matrix parameters.

Compare against a per-iteration (untied) transcoder set only if Stage 5 is finished and budget remains.

**Outputs:** `transcoders.json`, `attention_share.json`.

---

## 6. Stage 4: mediation (DEF) and dictionary controls (budget about 8 GPU-hours)

**DEF, as in the small study (FIXED).** Classes are (depth in `D*`, iteration `t`), at the answer
position.
1. Replace the update `delta = state[t+1] − state[t]` by `delta'` from another `report` example of the
   same depth. The donor is a rotation within the depth group, seed `t`.
2. Restore `P_E(delta − delta')` for the top-`m` components. Score
   `DEF = (s(explained) − s(resampled)) / (s(clean) − s(resampled))`, with `s` the mean log-probability of
   the clean answer.
3. Eligibility: `s(clean) − s(resampled)` must be at least 0.1 nats, otherwise the class is `no_effect`.
4. `m` takes values in {1, 2, 4, 8, 16, 32, 64}.
5. Bases:
   - SAE features at `t+1`, ranked by `|code(clean) − code(resampled)|` times the state's RMS;
   - transcoder latents, ranked by code times the decoder norm;
   - attention heads, ranked by output norm;
   - their union, ranked by raw-unit attribution.
6. Controls at the same `m`: random SAE features, top-`m` PCA directions of `delta`, top-`m` MLP neurons
   by absolute activation, random directions, and directions orthogonal to the basis span.
7. A class is accepted if `S90 <= 16` in some basis and beats every control at that `m`.

**Dictionary controls (FIXED).**
- Seed stability of the selected SAE across seeds 0, 1 and 2: a feature is matched when its best decoder
  cosine in each other seed is 0.9 or more.
- A covariance-matched Gaussian SAE: same size, trained on Gaussian vectors with the mean and covariance of
  `train_states`. Rerun the Stage 5 feature selection with it as the structure-free control.

**Outputs:** `def.json`, `controls.json`.

---

## 7. Stage 5: where the named variable lives (budget about 15 GPU-hours)

This is the headline stage. Pairs (base, source) are drawn from the same split with equal `K`. The
counterfactual for step `s` is `y_node = f_B^(K−s)(f_S^s(start_S))`. The start-copy alternative is
`y_start = f_B^K(start_S)`.
- Exclude pairs where `y_node` is the base answer.
- Report results separately on non-colliding pairs (`y_node != y_start`).
- The control counterfactual is the source's own answer.

**5a. Full-patch scan (FIXED grid).**
- Iteration `t` in 1..r*, step `s` in 1..max(D*).
- Site sets:
  - **A:** the answer position only;
  - **Q:** the query block (the tokens of `Start: X`, `Hops: K`, and `Answer:`) plus the answer position;
  - **E:** the edge-list tokens for the source's path nodes. Optional, and only if Q fails everywhere.
- Patch the full state at that site from the source.
- Report: IIA for `y_node` on non-colliding pairs, the start-copy rate, and the source-answer control.
- Use 1,000 `selection` pairs per cell. This is the selection data.

**5b. Cell choice (FIXED).** On `selection`, choose the best cell per site set: `s <= 2`, `t < r*`, and a
source-answer control of 0.2 or less. Freeze it. Also freeze the best cell with `s >= 3`, if its IIA is at
least 0.4.

**5c. Rank sweep (FIXED).**
- At each frozen cell, fit DAS (a subspace of the flattened site state, orthonormal by QR).
- Ranks {8, 32, 128, 512}, Adam lr 1e-2, 600 steps, batch 64, 3,000 `selection` pairs.
- Evaluate IIA on 1,000 `report` pairs.
- Also fit the final-answer cell: site A, last iteration, `s = K`.

**5d. Start-copy audit.** Collision rate, IIA on non-colliding pairs, and start-copy rate, for every
frozen cell.

**5e. Feature link, the registered successor (FIXED).** At each frozen cell:
- Rank SAE features by the mean `|code(source) − code(base)|` over the site's positions on 2,000
  `selection` pairs.
- Patch only the top-`m` features' code differences through their decoders, at the base state's scale,
  for `m` in {8, 16, 32, 64, 128}.
- Controls: random alive features at the same `m`, all features (the reconstruction difference), and the
  full patch.
- Decision rule:
  - *Eligible:* full-patch IIA on `report` of 0.6 or more.
  - *Linked:* at some `m <= 64`, the feature patch reaches half the full-patch IIA or more **and** beats
    random features by at least 0.20.

**Outputs:** `scan.json`, `das.json`, `collision.json`, `feature_link.json`.

---

## 8. Stage 6: recurrence-specific tests (budget about 6 GPU-hours)

1. **Counter (FIXED).**
   - Candidates are SAE features whose activation at the answer position has a median per-example
     Spearman correlation of 0.9 or more with `t`, over 300 `selection` prompts, and that are not
     visit-index features (iteration explains less than 90% of their variance across examples). Take the
     top 5.
   - Clamp each candidate at iteration `v` to its value at `v − 2`, for `v` in 3..r*.
   - Readout: the share of answers that change from correct to an earlier node on the walk.
   - Supported if some candidate reaches 0.5 or more, with a matched-norm random clamp at 0.1 or less.
2. **Time translation.** Move `state[t]` into iteration `t'` (`t' = t ± 2`) and continue. Report EM and how
   the number of extra or missing iterations shifts the answer. Report only.
3. **Extra iterations (Huginn only).** Run `2r*` and `4r*`. Report the share of correct answers at `r*`
   that stay correct, and whether Stage 5's frozen-cell IIA shifts in time.

---

## 9. Stage 7: sealed confirmation (budget about 3 GPU-hours)

Only after Stages 2 to 6 are written up:
1. Commit `CONFIRM.md` listing the exact frozen cells, ranks, `m` values, selected SAE and the decision
   rules from 5e and Stage 6.1.
2. Hash `CONFIRM.md`.
3. Then read `sealed` and report every registered endpoint on it, once.
4. No re-runs on `sealed`.

---

## 10. Budget and stop rules

| Stage | GPU-hours (A100/H100 class, rough) | Stop if |
| --- | --- | --- |
| 1 Feasibility | 2 | no `r*`; capture does not reproduce the logits; extrapolated total over 2x |
| 2 N1 | 12 | SAE training diverges twice |
| 3 Transcoders | 10 | |
| 4 DEF + controls | 8 | |
| 5 Named variable | 15 | no cell anywhere reaches full-patch IIA 0.3 (report as `unresolved_site`) |
| 6 Recurrence tests | 6 | |
| 7 Sealed | 3 | |
| **Total** | **about 56** | |

- Priority if RU runs short: Stage 5 (a, b, d, e), then 2, then 3, then 4, then 6. Skip 5c ranks 512
  first.
- Use bf16 for model forwards and fp32 for SAE and transcoder training and for every metric.

---

## 11. Deliverables and file formats

- One directory per stage: `stage_N/`, containing the JSON files above, `stage_N_report.md`, a `seeds.json`
  and a `cost.json` (GPU-hours and RU).
- **JSON conventions.**
  - Numbers are unrounded.
  - Every EM is given as `{"correct": int, "n": int, "rate": float}`.
  - Every ratio stores its numerator and denominator.
- **`CHANGES.md`.** Append-only, dated. Every departure goes here, written before its result is read.
- **Final.** `summary.md` with one row per test from Section 0's table: the small-model result next to the
  LM result.
- **Hand-back.** Copy `summary.md`, all JSON and `CHANGES.md` back to the repo under
  `experiments/recurrence_monitorability_v0/silico_<model>/`. The paper's next version cites them.

## 12. Instructions for the worker model

- **Do not invent numbers or summaries.** Every number in a report must come from a JSON file written by
  code in this run.
- **Never look at `sealed` before Stage 7.** Never choose anything on `report`.
- **When a rule cannot be applied as written**, for example because the model has no clean MLP sublayer,
  write the closest faithful alternative into `CHANGES.md` first, mark it post hoc, and continue.
- **Record failures.** Write failed runs, NaNs and unresolved labels as results, not as reasons to retune.
- **Reference implementations.** The small-model code is in `research_gym/scripts/rmp_phase2_*.py`:
  `n1`, `mlp_transcoder`, `e5`, `n2`, `query_patch`, `query_das`, `query_collision`, `feature_link` and
  `h2`. Port their logic; do not copy their hard-coded sizes.
