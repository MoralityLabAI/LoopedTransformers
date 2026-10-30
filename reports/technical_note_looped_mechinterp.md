# What a Looped Transformer Computes, and Where: A Mechanistic Study of Tied Recurrent Depth

*Technical note, Recurrence Monitorability Program v0, Phases 0 to 2 (October 2026). Exploratory: every
rule was fixed before its run, and every later departure is marked post hoc. Branch `rmp-v0-phase2`.*

## Summary

The note compares two small decoder models trained on four algorithmic tasks: graph pointer chasing, gated
pointer chasing, reachability and 4x4 sudoku.

- **`LOOP-T`** applies one transformer block 8 times (tied weights, 0.46M parameters).
- **`FF-U`** stacks 8 distinct blocks (untied, 1.84M parameters).

Both reach 98% exact match on every task.

1. **Sparse dictionaries describe both models' states.** A sparse autoencoder shared across all 8 visits can
   replace the tied loop's state after every visit and keep 90 to 92% of clean deep-chain accuracy. It
   beats matched-error PCA on 2 of 3 tied seeds and ties it on the third. The same dictionary family fails
   on the untied stack: it reaches 69%, and PCA wins at every size.
2. **Transcoders work only if attention is left live.** Whole-block transcoders fail outright, because 30 to
   42% of each update is attention, which a per-position map cannot predict. MLP-only transcoders keep 94
   to 99% of accuracy in both arms.
3. **A handful of features explains each step.** 4 to 8 dictionary features rebuild 90% of a transition's
   effect on the answer, beating random, PCA, neuron and orthogonal controls. Transcoder latents and
   single heads do not.
4. **The intermediate trace is causally real but not compact.** The current node of the chase is
   transferable between examples only for the first one or two hops, and only when the query tokens and
   the answer position are patched together. Even then it needs a 32- to 128-dimensional subspace. The
   final answer, by contrast, sits in a rank-32 subspace of the answer position alone from 1 or 2 visits
   before the end.
5. **No counter, no single rule.** No visit-counting feature that the loop causally uses was found. No single
   transition latent carries a step.

The arms differ in trainability and in how well one shared dictionary fits them, not in the kind of
mechanism found. Tied seeds passed the accuracy gate 6 of 6 times, untied seeds 3 of 6.

## Setup

**Tasks and models.**
- Inputs are 64 tokens: an edge list, a query (start node and hop count), and an answer position.
- Depths 1 to 8 are in distribution. Accuracy is reported on depths 5 to 8 ("deep").
- Training used a depth curriculum and, where needed, deep supervision: a loss on the answer after every
  visit.
- `FF-U` sometimes also needed input re-injection.

The gate `G0` asks for 0.98 exact match on all four families.

**Models analysed.**
- Phase 2: tied seeds 1 to 3 and untied seed 3, the only untied seed of the first three to pass.
- Held-out replication: tied seeds 4 to 6 and untied seeds 4 and 6.

**Determinism.** Training is bit-for-bit deterministic on this machine. When the artifact directory was
lost mid-study, every model was rebuilt and verified against its recorded checkpoint hashes.

**Sites.** `resid[t]` is the residual stream entering visit `t` (layer `t` for `FF-U`). Dictionaries are
TopK sparse autoencoders on RMS-normalized states, with widths 1,024 to 8,192 and `k` of 8, 16 or 32.

## 1. Closed-loop faithfulness of a shared dictionary (N1)

A dictionary that reconstructs each visit well can still describe states the loop cannot run on, since the
tied block re-reads its own output.

**Test.** Replace the state by its reconstruction after every visit and measure deep exact match relative to
clean. Two controls:
- PCA truncation with the same calibration error;
- isotropic noise with the same norm as the reconstruction error.

| Model | Best closed-loop | Matched PCA | Noise | Label |
| --- | --- | --- | --- | --- |
| `LOOP-T` s1 | 0.923 | 0.926 | 0.805 | `faithful_state` (tie with PCA) |
| `LOOP-T` s2 | 0.910 | 0.897 | 0.799 | `faithful_state` |
| `LOOP-T` s3 | 0.900 | 0.821 | 0.769 | `faithful_state` |
| `FF-U` s3 | 0.688 | 0.796 | 0.778 | `unresolved` |

- **Tied loop.** The loop runs on its own dictionary reconstructions for 8 visits without compounding error
  beyond the noise control. The registered negative outcome (`open_loop_only`) did not occur.
- **Untied stack.** For `FF-U`, one dictionary across 8 different layers is a poor basis: open-loop
  cross-entropy recovered is 0.33 to 0.69 at small sizes, against at least 0.92 for the tied loop.
- **Interpretation.** Tied recurrence makes a single shared feature basis natural. Untied depth does not.

**Controls.** All of these were fixed before the runs.
- A planted direction is recovered: decoder cosine 0.84, AUROC 0.99.
- A hash bit no model can compute is never "found".
- Random-initialization models find no trace variable.
- No feature is a pure visit-index feature.
- Found features are much more stable across dictionary seeds (66 to 84% matched) than features in
  general (2 to 14%).

**A caution about association.** Binary trace variables (`latch`, `goal_found`) are found almost as easily by
a dictionary trained on Gaussian data with the same covariance. Finding them shows the variable is
prominent in the state, not that the dictionary learned it. Only `cur_node`, the current node in the chase,
separates trained from structure-free dictionaries.

## 2. Transition functions: attention must stay in the loop

**Whole-block transcoders fail.** The registered transcoder predicted each position's whole-block update
`delta[t]` from that position's `resid[t]`.
- It plateaued at 16 to 27% unexplained variance.
- Its replacement model collapsed to a constant output: 0.17 of clean.
- The cause: the attention term `alpha * beta * attn_out` is 33, 30, 30 and 42% of the update's variance
  (tied seeds 1 to 3, untied). It depends on other positions, so no per-position function can predict it.

**MLP-only transcoders work** (post hoc). These read the MLP input and predict the MLP output, with attention
left live.

| Model | FVU (w4096 k32) | Replacement deep EM / clean |
| --- | --- | --- |
| `LOOP-T` s1, s2, s3 | 0.003 to 0.014 | 0.967, 0.977, 0.938 |
| `FF-U` s3 (8 transcoders) | 0.015 | 0.991 |

**Description length (H3).** One tied transcoder reaches 0.95 at width 2,048 with about 1.7k alive latents
(seeds 1 and 2), against the untied stack's eight at about 5.9k. Measured in bits, the tied transcoder is
about 8 times smaller. Seed 3 tops out at 0.947, so the registered H3 is `unresolved`, missing by 0.003.

## 3. What explains one step

**Method.** For each family and transition:
- the answer position's update is replaced by another example's;
- the top `m` components of the true update are restored;
- DEF is the fraction of the clean-versus-resampled gap in answer log-probability that is recovered.

A class is accepted when 16 or fewer components reach DEF 0.9 and beat every control at that size.

| Model | Classes with an effect | Accepted | Median S90: features, rules, heads |
| --- | --- | --- | --- |
| `LOOP-T` s1 | 19 | 63% | 4, 64, never |
| `LOOP-T` s2 | 23 | 91% | 8, never, never |
| `LOOP-T` s3 | 23 | 78% | 4, 64, never |
| `FF-U` s3 | 20 | 65% | 8, 64, never |

- **Features suffice.** A few dictionary features at the next state explain each transition.
- **Rules and heads don't.** The MLP's contribution spreads over about 64 transcoder latents, and no single
  head suffices.
- **Rule ablation agrees.** Zeroing any one of the top-20 MLP-transcoder latents costs at most 0.4 to 8.8
  points of exact match, so no latent is a step.

## 4. Where the chase lives

**Abstractions tested.** Phase 1 had found an arrival schedule of about one hop per visit for 2 or 3 hops,
then several per visit. Interchange interventions tested three abstractions of pointer chasing: one hop per
visit, doubling, and a fitted schedule.

**At the answer position alone.**
- No intermediate node transfers: a full-state patch recovers at most 0.22 of counterfactuals before the
  last two visits.
- The answer position holds the final answer from 1 or 2 visits before the end, in a rank-32 subspace
  (IIA 0.95 to 0.99, with the full patch at 0.97 to 1.00).
- Doubling never exceeds 0.08.

**At the query block plus the answer position** (post hoc).

| Model | Cell | Full block | Rank 8 | Rank 32 | Rank 128 | "Source answer" control |
| --- | --- | --- | --- | --- | --- | --- |
| `LOOP-T` s1 | visit 2, hop 1 | 0.68 | 0.03 | 0.09 | 0.49 | 0.06 |
| `LOOP-T` s3 | visit 2, hop 1 | 0.77 | 0.06 | 0.41 | 0.66 | 0.05 |
| `FF-U` s3 | layer 3, hop 1 | 0.69 | 0.09 | 0.56 | 0.63 | 0.08 |
| `FF-U` s3 | layer 4, hop 2 | 0.61 | 0.04 | 0.20 | 0.34 | 0.11 |

- **The early hops are real.** The first one or two hops are causally present mid-loop, spread over the
  query tokens and the answer position, and only in a high-dimensional form.
- **Later hops never transfer as a clean node.** Tied seed 2 shows none at all, matching its weaker
  association results.
- **Gated pointer chasing never transfers above 0.6** at these sites.
- **A visit counter does not exist in a causal form.** Visit-monotone features exist on 2 of 3 tied seeds,
  but clamping one to its value two visits earlier changed at most 0.6% of answers. The bar was 50%.

## 5. Implications for monitoring looped models

- **A tied loop is easier to audit with full access.** One feature basis serves every visit and survives
  closed-loop replacement. One MLP transcoder serves every visit at a fraction of the untied description
  length. A few features explain each step.
- **A low-rank monitor watching the answer position will not see the reasoning trace.** It sees the answer
  late. The intermediate hops it would want to watch are causally real but spread across positions and
  32 or more dimensions.
- **Decodable is not the same as used.** Linear probes decode `cur_node(1..2)` at the answer position at
  0.76 to 1.00, but a patch there does not move the computation. Probes there read a correlate.
- **Whole-block transcoders are the wrong tool for loops.** Any replacement-model account of a recurrent
  transformer has to keep attention live.

## Limitations

- **Scale and models.** The models are small: 128 hidden units, 4 heads. The tasks are synthetic, and the
  untied arm passed the gate on only 3 of 6 seeds.
- **Exploratory, with post-hoc parts.** Phase 2 is exploratory, and the MLP transcoders, schedule scan and
  query-block patch are post hoc, though each was recorded before its run. The confirmatory tests (H4, H5,
  H7 on fresh seeds with a locked split) were not run.
- **Unfinished analyses.** The six-lens Loop Delta Reports and patches at edge-list positions were not
  produced.

## Held-out replication (fresh seeds 4 to 6)

Pending: results land with the replication runs started 2026-10-30.

## Artifacts

- **Plans:** `docs/rmp_phase2_plan.md` and `docs/rmp_phase2_e5_e6_plan.md`, with every change dated.
- **Checkpoint reports:** `reports/checkpoint_rmp_phase0.md`, `reports/checkpoint_rmp_phase1.md`,
  `reports/checkpoint_rmp_phase2_e4.md` and `reports/checkpoint_rmp_phase2_e5_e6.md`.
- **Result JSON:** `experiments/recurrence_monitorability_v0/phase2/`.
