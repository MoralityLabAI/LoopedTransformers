# NAN-VOCAB-81 full-vocabulary replay at the recurrent readout

INVERSE65 found a large counterfactual bare-label contrast at physical layer 17
in pass 2, with both target labels beating the other world's bare label on
every focused DEV root. The same checkpoint and physical site across passes
allow an Inkling-style comparison of what the frozen head can read. We replay
the complete vocabulary to locate the competitors hidden by the two-label
contrast, and pair that result with the actual native first-token outcomes.

The unit is one COMP14 pointer graph root. Include all 8 CAL and 16 DEV roots
with focused, copy and erased A/B worlds, in each native decoder precision
(144 archived prefixes per precision). Inspect layer 17 and layer 21 on both
passes, all at the original first-token prefix. Pass 2 layer 17 is the primary
observational readout. The two pass-boundary heads are positive controls for
replay fidelity. Identical erased twins are a negative control for the entire
head replay. CAL and DEV are reported separately. All roots and sites were
previously examined in COMP14 and the follow-up lens and residual work.

Use the locked checkpoint normalization, BF16 head weights and final BF16
rounding in fixed-size CPU tiles. Accumulate the log normalization constant
and maxima over all rows. Map each token to the exact COMP14 first-token parser
class or invalid, with the pinned tokenizer and special tokens retained. A
winning spelling is any token parsed as the target label. Use strict margins
against a competing class, count cross-class ties, and report the deterministic
lowest-token-ID top separately. Compare four nested scopes using a common
score construction where applicable:

- The intended bare label against the other world's bare label.
- The intended bare label against the other fifteen bare labels.
- The best spelling of the intended label against the best spelling of each
  other valid label.
- The best accepted spelling of the intended label against all other tokens.
  Partition failures by invalid or other valid-label competitors, retaining ties.

The second and third comparisons may disagree because their spelling
supports differ. The first two use fixed bare tokens. An unrestricted success
can be carried by a different spelling. Track the largest invalid-token
score and strongest other-label score separately. At native boundaries the
reducer must recover the native winning token, with the original native
logit and log normalizer within 0.125 logits. The two-world bare margin must
also close against the previously recorded INVERSE65 groups. Tile and dense
reduction agree on a synthetic vocabulary with aliases and tied logits.

Pair the layer-17 result with native pass-1 and pass-2 correctness and track
repairs, regressions, always-correct and both-wrong cases. The observations
inform whether a two-label sink is a sufficient proxy for a PercInterp
path's behavioral target. Layer-17 scores use the final head as a lens.
There are no edits, dose, sham envelope, fitted policy, pretrained decoder
forwards or formal comparisons. No mechanistic-resolution or confirmation
claim is in scope. The readout is already exposed, and its evidence class is
descriptive observational.

Use RESOURCE29 concurrent limits with one numerical thread, Below Normal
priority, 5 GiB starting available RAM, 4 GiB reserve, 768 MiB private
commitment limit and a 600-second self-only timeout. Reuse the pinned NumPy
and tokenizer environment without importing Torch.
