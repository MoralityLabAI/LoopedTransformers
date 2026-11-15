# PEF-08 | Frozen coalition interaction interventions

Registration: 2026-11-15. This plan is committed before PEF-08 intervention scores are computed. PEF-07 outcomes and interaction ratios were already known. The six PEF-07 checkpoints are the *only* model inputs; no retraining or heldout-guided feature choice.

## Provenance and separation

Parent: PEF-07 PR #67, head d9c2087cbe37f4787e3e56ef9d64ab2eb148f214, preregistration a3b441ce5fbc8f8d33914a1a34c883875ea5873c. Model seeds 103/131/167, two tasks (coalition and threshold). The six original 450-update CPU checkpoints must match their byte SHA256 values and state-dict digests from the supplied `sealed_training_ledger.json`. Source SHA256 is `849d039e47dec472b0b4835d24928e9a547e262bf84fd7dfed16741df9911ac1` for the archived implementation. The source split has 160/48/48 context-disjoint groups, each carrying the complete 16-input focal cube. Checkpoint and source identity mismatches abort execution.

PEF-08 uses the *probe* 48 contexts to select intervention coordinates and fit the one frozen h14 ridge readout. The *heldout* 48 contexts are used only to score the already specified interventions and inspect loop trajectories. No heldout feedback changes the coordinates, number of channels, time points, or analysis. The same native h12 readout is retained through each h12 intervention. The h14 fitted readout is trained once on untouched h14 probe states, then fixed across all h14 conditions.

## Loop-by-loop observation

Compute natural memory h(t), t=0..14, where t<=12 denotes the first t chronological source bits and t=13,14 use continuation bits (0,1). For each context, each of the six old-bit pairs i<j, evaluate at focal 0000:

  D_ij h(t) = h(t)[2^i+2^j] - h(t)[2^i] - h(t)[2^j] + h(t)[0].

Report mean hidden L2, and signed/absolute mixed native margin using the *unchanged* trained h12 head across loops 0..12; h14 receives an additional separately fitted ridge score. Intermediate-loop native margins are diagnostic probes using an h12-trained head, not intermediate task accuracy.

## Frozen probe feature selection

For each task, seed, loop t in {4,8,12} and intended pair in {(0,1),(2,3)}, rank 32 channels using only the *probe* cohort:
`score[c]=mean_context(abs(D_ij h(t)[c])) * abs(native_head.weight[1,c]-native_head.weight[0,c])`.
Choose 8 highest channels (ties by lower channel index). Fix one bottom-8 control and eight size-matched random masks produced by `random.Random(8008 + 1000*seed + 10*t + 2*pair_index + task_offset)`, with task_offset=0 for coalition and 100000 for threshold. Masks are generated without using heldout states or labels. The feature score is a heuristic; channel directions need not align with causal factors.

## Causal intervention and comparison

For each unmodified heldout cohort and selected pair p=(i,j), at visit t compute its natural hidden second difference D_p h(t). For all 16 focal states in that context, subtract only the selected channels of `D_p h(t)` when both i and j are one:

  h'_t(mask)[c] = h_t(mask)[c] - 1[(mask & (1<<i | 1<<j))==(1<<i | 1<<j)] * 1[c in C] * D_p h_t[c].

Continue the identical tied block on all remaining natural inputs to 12; at 14 append (0,1). No new gradients, input edits, label edits or weight edits. The subtraction is an explicitly constructed four-corner interaction excision at the selected visit. In particular, the targeted hidden D_p on selected channels is zero by construction *at that visit*. Changes at the final readout, class predictions and other states are the empirical results. The method does not by itself establish that these coordinates are a natural circuit.

Compare top-8 to bottom-8 and the eight random masks, at t=4,8,12. Report heldout original/modified final margin D_p absolute, signed and relative attenuation, h12 classification loss, 16-mask absolute logit disturbance, and h14 frozen-ridge accuracy and margin for the t12 interventions. Report each randomized comparator individually, not just its mean. Include zero-channel identity and all-channel excision as invariant checks in unit tests. The additive-threshold trained task is run identically.

Primary observational contrast (fixed now): for each coalition seed and pair, `[top8 attenuation - mean_random8 attenuation]` at h12 on the fixed native margin. Report all 6 contrasts and the count of seeds where both intended pairs have contrast above zero. A `2/3` count is a descriptive signal, not an independent validation dataset; the seed set and task family were known from PEF-07. Record threshold-arm comparisons and postintervention accuracy separately. Bootstrap context resampling, if performed, is strictly descriptive.

## Boundaries

Pair interactions at the zero focal anchor are local signatures. The function XOR(AND(a,b),AND(c,d)) can produce crosspair interactions at other focal bases. The intervention has a deliberately pair-gated mask, so unaltered zero-anchor crosspair values are a mathematical consequence of its design and cannot count as causal selectivity. No automatic inference about an SAE feature, a uniquely identifiable circuit, pretrained LLMs, or Lean theorem-proving skill. No architecture or checkpoint changes; PEF-07 PR #67 remains untouched. Hosted CI on Python 3.12 requires a runner and the external checkpoint archive, and is not certified by local Python 3.13 execution.

## Invocation

```bash
python -m pytest -q research/pef08/test_pef08.py
python -m research.pef08.pef08 --evidence /path/to/pef07/registered --out /tmp/pef08
```
