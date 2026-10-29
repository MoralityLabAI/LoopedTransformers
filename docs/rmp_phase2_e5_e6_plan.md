# RMP v0 Phase 2 plan, part 2: E5 and E6

Written 2026-10-29, after E4 finished (`reports/checkpoint_rmp_phase2_e4.md`) and before any E5 or E6
code or result. Every rule below is fixed before its first run. A later change is recorded under
"Changes" with its date and reason, before the result it affects is read. Phase 2 is exploratory.

## Models and data

- **Models.** `LOOP-T s1`, `s2`, `s3` and `FF-U s3`, with their `C0` models where a control asks for them.
- **Development data.** `monitor_calibration`: the selection half for choices, the reporting half for
  numbers.
- **Final numbers.** `validation`, restricted to depths 5 to 8 of the three graph families, as in N1.
- **Locked split.** `locked` stays unwritten. It belongs to Phase 3. Where the program says "on locked",
  Phase 2 reports `validation`, labelled exploratory.

## What E4 changed

The registered `D-trans` is ill-posed. It predicts the whole block's update from one position, but 30 to
42% of that update is attention (`trans_diagnosis.json`).

Every program use of `D-trans` in E5 and E6 therefore uses the post-hoc MLP-only transcoders instead
(`mlp_transcoders.json`). These are:

- one for the tied loop, eight for `FF-U`;
- at w4096 k32, the smaller of the two sizes run;
- run with attention kept live.

This applies to Loop Delta Report lens 3, rule ablation, and H3. It is a departure from the program, and
every result that rests on it says so.

## E5: Loop Delta Reports and fidelity

**Reports.** For each model, 200 examples per graph family are drawn from `validation` at depths 5 to 8,
with seed 0. Each example has its sibling, and every transition `t -> t+1` (`t = 0..7`) gets a report.

The six lenses are as in the program, with these choices:

- **Lens 1 (component split).** Per head and MLP, from `ExplicitDecoderBlock.components`.
- **Lens 2 (dictionary difference).** The model's selected `D-shared`.
- **Lens 3 (transition rules).** The MLP transcoder's active latents.
- **Lens 4 (variable difference).** Phase 1's E2 probes.
- **Lens 5 (output difference).** The logit lens on `final_norm`.
- **Lens 6 (geometry).** As specified.

Each update is split into its dataset-mean part, per family and visit, and the example-specific remainder.

**Fidelity.** DEF is computed exactly as the program defines it:

- `delta'` comes from another example of the same family, depth and visit, drawn with a fixed seed.
- `score` is the clean answer's log-probability at the final visit.
- DEF is averaged over examples per (model, family, `t`).

Candidate explanations `E` are ranked in four component bases, each by its attribution to the
example-specific update (dictionary code times decoder norm, or component output norm):

| Basis | Components |
| --- | --- |
| features | `D-shared` decoder directions at `t+1` |
| rules | MLP-transcoder latent output directions at `t` |
| heads | head outputs, each one direction per example |
| mixed | the union of the above, ranked jointly |

`S90` is the smallest `m` in {1, 2, 4, 8, 16, 32, 64} with DEF of 0.9 or more.

**Acceptance.** A report class (model, family, `t`) is accepted when both hold:

1. `S90 <= 16` in some basis.
2. Its DEF at that `m` exceeds each control at the same `m` and norm:
   - random `D-shared` features;
   - top-`m` principal components of the update distribution at that visit;
   - top-`m` MLP neurons by activation;
   - matched-norm random directions and orthogonal directions (`matched_control_directions`).

A necessity test is also reported: remove `P_E` from the true update and measure the drop in score.

**Reported:** the share of accepted (family, `t`) classes per model, with tied and untied side by side.
The program does not make E5 negative-capable, so it has no pass bar.

**H3 (description length).** For each model, report the MLP transcoders' total description length at the
smallest size in the grid {1024, 2048, 4096, 8192} x {8, 16, 32} whose replacement model keeps 0.95 or more
of clean deep exact match. The length is counted in two ways:

- (a) alive latents summed over transcoders;
- (b) mean active latents per position per visit, times the bits to name one latent (`log2` of alive
  latents), summed over transcoders, plus the skip matrices' parameters at 16 bits.

H3 is supported if `LOOP-T`'s total is below `FF-U s3`'s on both counts for every tied seed. It is
contradicted if it is above on both counts for every tied seed. Any other outcome is mixed. If `FF-U` or a
tied seed reaches 0.95 at no size, H3 is `unresolved` for that model.

## E6: concept hunts with causal tests

**N2 (abstraction mismatch, negative-capable).** The program registers three abstractions of pointer
chasing:

- one hop per visit: `cur_node(s)` after visit `s`;
- doubling: after visit `v`, the node `2^(v-1)` hops ahead;
- parallel lookahead: `cur_node` at all steps up to `v` held at once.

Each abstraction's variable at visit `v` gets a rank-`r` subspace (r in {1, 2, 4, 8}), fitted by
distributed alignment search on the selection half. The answer-position `resid[v]` is interchanged
between an example and its sibling.

- **IIA** is the share of examples whose answer becomes the abstraction's counterfactual prediction.
- **Reported cells:** the best `r` and `v` per abstraction, on `validation`.
- **Guards, from the program:**
  - The subspace must also be probe-decodable.
  - A full-residual patch at the same site sets the ceiling.
  - Patched states must stay within the 99th percentile of clean norm and of `D-shared` reconstruction
    error.

N2's strong negative holds for a model when clean accuracy is 0.98 or more and the best IIA over the three
abstractions is below 0.6. It is computed per model, for `pointer_chase` and `gated_pointer_chase`.

**H2 (counter).** "A clock-free tied loop builds a counter it causally uses."

- **Candidates:** `D-shared` features whose activation is monotone in visit across the example (Spearman
  correlation 0.9 or more over visits 1 to 8, on 300 selection-half examples) and that are not
  visit-identity features under control 10.
- **Causal test:** clamp the candidate to its value from visit `v - 2` at visit `v`, for the top 5
  candidates by monotonicity, on the reporting half.
- **Supported** if, for some candidate on every tied seed, the clamp shifts the step-indexed probe readout
  of `cur_node` back by one step or more in at least 50% of depth-5-to-8 examples, while a matched-norm
  random clamp does so in at most 10%.
- **Not supported** if no candidate exists or none meets the bar.
- **Reported** for `FF-U s3` alongside.

**Rule ablation.** Zero one MLP-transcoder latent, at all visits, in the replacement model. For each of the
top 20 latents by attribution, report the drop in exact match per family and depth, and at which algorithmic
step the first error appears. Reported only.

## Order and compute

1. E5 reports and DEF, then the H3 grid (MLP transcoders at the remaining sizes).
2. E6: N2, then H2, then rule ablation.

All GPU jobs run through the RMP queue, with Phase 2 leases. Each step ends with a commit and a report
section in `reports/checkpoint_rmp_phase2_e5_e6.md`.

## Changes

**2026-10-29: E5 details settled before any full E5 run.** A 24-example CPU smoke test had run; its output
was read only to check that the code works, and was not kept.

- **Intervention site.** The update is resampled and explained at the answer position only. Other positions
  keep their clean update.
- **Donor.** The donor is fixed per visit: examples are rotated by one within each depth group, in a
  seed-`t` order.
- **DEF.** DEF is a ratio of means over the class's examples. A class whose resampling changes the mean
  answer log-probability by under 0.1 nats is labelled `no_effect` and left out, because DEF is undefined
  there.
- **Attribution.**
  - Features: `|code(clean resid[t+1]) - code(resampled resid[t+1])|` times the state's RMS.
  - Rules: the transcoder code times `|beta|`.
  - Heads: the norm of the head's output.
  - Mixed: the union ranked by these raw-unit values.
- **Controls.** The orthogonal control is drawn orthogonal to each basis's own span. Neurons are ranked by
  absolute GELU activation, with their output weight columns as directions.
- **Not yet produced.** The six-lens reports themselves are deferred. Only DEF, S90, acceptance and
  necessity are computed.

**2026-10-29: N2 site guard, settled before any full N2 run.** A CPU smoke test (3 fitting steps, read only
to check the code) showed that the full-residual patch at the answer position recovers just 0.14 to 0.24
of counterfactuals. The program already treats the full patch as the ceiling.

- **Usable cells.** A (family, abstraction, visit) cell counts toward N2 only if its full-residual patch IIA
  is 0.6 or more.
- **Verdict.**
  - With no usable cell in a family, N2 there is `unresolved_site`. The answer position does not carry the
    variable, so the test cannot speak.
  - Otherwise it is `negative` when the best usable IIA is below 0.6, and `not_negative` when it is not.
- **Scope.** The interchange is at the answer position, as implemented. Other positions are a follow-up.
- **Abstractions.** Only `hop` and `doubling` are fitted. At the answer, parallel lookahead predicts the same
  counterfactual as one hop per visit, because the answer continues from the latest node. It is therefore
  not separately testable at this site.
- **Probe guard.** The program's probe-decodability guard is deferred. The norm guard is reported as the
  share of patched states above the 99th percentile of clean norms.

**2026-10-29: post hoc, after reading N2.** In the only usable cell, IIA was still rising steeply at rank 8.
N2 is therefore refitted at ranks 16 and 32, with everything else unchanged, into a separate file,
`n2_rank16_32.json`. The registered verdict stays as reported. The refit is labelled post hoc, and it can
only qualify that verdict, not replace it.

