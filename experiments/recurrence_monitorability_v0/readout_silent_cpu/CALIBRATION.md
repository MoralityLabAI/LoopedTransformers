# Post-hoc cross-task manipulation check

Declared after the locked same-task pilot yielded zero top-1 continuation
changes in all 3x256 selected pairs. It does not replace that primary null
result. No change is made to the original selection or training.

Evaluate the same trained checkpoints and 512 original input draws. At loop
4, choose up to 256 recipients with a donor from a DIFFERENT graph, the same
current oracle answer, and a different next oracle answer. Both native current
heads must be correct. Pick donors with fixed RNG seed 941217; inspect no
patched future outputs when selecting pairs. Patch edge tokens only, retain
the recipient input embedding, and retain its answer token exactly.

Run the same .25/1 donor doses, per-pair matched Gaussian and LayerNorm-null
controls. Report first-step recipient and donor answer agreement, future
changes and patch norms. Conflicting graph memories can produce off-manifold
states. Any change is a finite manipulation response, not evidence of a
hidden belief or a failure of the original same-input natural quotient.
