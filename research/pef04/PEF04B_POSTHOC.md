# PEF-04B | Matched initialization-to-trained memory study

The PEF-04 preregistered six-seed run [43683120934](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43683120934)
failed its predefined 4/6 old-history transfer criterion. The unexpected
control was that **all six different randomly initialized models scored
48/48** on oldest-bit reconstruction after two natural extra loops.
This follow-up is explicitly **posthoc** and retains that original negative
decision.

The experimental question is whether each trained checkpoint
retains less information than **its own exact initial weights**.
For each seed 101,109,127,131,149,157, reconstruct the original
pre-training model using \`manual_seed(seed)\` and compare its exact
parameter digest with the \`initial_weights_sha256\` saved *before
any heldout evaluation* in the original checkpoint ledger. If the
hash differs, abort. Load the original trained checkpoint using
its pinned SHA256 and replay its original scores exactly.

Fit fresh oldest-bit ridge heads on the same 48 pair-closed
probe histories for the original initialization and trained
checkpoint, and score the 48 heldout histories on natural h8,
h10 after 00, and h10 after 11. The code independently reconstructs
the 10-step causal oldest-bit trajectory under 00 continuation,
recording absolute pairwise L2 effects, state norm, and directional
coherence.

This isolates training-associated changes within *each initialization*
and is independent of the nominally separate random control.
The analysis is retrospective. It does not localize the training
step where memory changed or claim that low L2 distance proves
the absence of all nonlinear information.

All six original binary checkpoints, source manifests and recorded
negative verdict are read-only. New artifacts contain JSON only.
