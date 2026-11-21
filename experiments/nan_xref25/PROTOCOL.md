# NAN-XREF-25: separate decoder plus FP64 physical-geometry audit repair

The original XREF19/XREF22 sources and the separate orientation-only XREF24
condition stay frozen. Original failures remain failures. No model/worker
replay, new grade, dictionary fitting or neural checkpoint change.

XREF24 corrects the three[3072,128] decoder orientations and then encounters
the unchanged strict geometry comparison at rtol1e-10/atol1e-9. The auditor
selects U from a float32 checkpoint, computes its norm in float32, and combines
it with float64 dictionary rows. The actual worker's probe_and_crossref first
casts its physical output directions to float64, so both dot product and norm
use that dtype. The exposed NF4/BF16 discrepancy is several1e-9, consistent
with the mixed-precision norm. Do not widen any tolerance to make it pass.

This distinct condition applies XREF24's mechanically reversible three decoder
transposes, then appends `.astype(float)` to exactly the one U assignment in
the physical geometry loop. That assignment selects learned, initial or Haar
directions, so all controls use the same producer arithmetic. Reverse the
single cast and recover the complete XREF24 transformed AST; all original
comparisons, assertions, tolerances, feature selection, root means and source
admission rules remain unchanged. Also verify the upstream frozen manifests.

Freeze/push this complete source before numerical replay. Pin the exact closed
XREF19 worker receipts AND both failed XREF24 audit receipts. These outcomes
are already exposed; this is engineering validation of an explicit dtype
contract, not confirmation on untouched data. Preserve every dataset, vector,
dictionary,128 features/top4, initial/random/SVD/Haar control, root stratum,
undefined correlation and out-of-fit-depth application. A later mismatch
fails the separate audit and is retained without another model call.

Write final receipts only into ignored runs/nan_xref25. Never create a passing
original XREF19/XREF22/XREF24 audit receipt. Publish complete aggregate cells
only when this separate independent audit passes, with old failures linked.

Before numerical replay, push the explicit supervision amendment: XREF24's
first host stage has ended with both failed audit receipts closed; its future
XREF22 stage is still unstarted. The new supervisor first captures the actual
XREF24/XREF22/STORY23 handles, then waits while ONLY authenticated task-owned
XREF24 supervisor15024 is stopped at its wait boundary. No numerical helper,
model worker, old source or math process is stopped. XREF24's future XREF22
stage becomes unrun under this prospective supervision amendment, not failed
or successful. This prevents concurrent original/replacement host lanes.

Then repair XREF19 NF4/BF16 serially alongside real Prime model work. For
XREF22, inherit the exact waits for actual XREF22 and both STORY23 lanes.
Thus there is at most one substantial numerical host analysis. One CPU thread, Below Normal priority,
same4-hour/helper observation limit with no duplicate run on observation
expiry. The finite72-hour between-stage wait never stops upstream work.

Evidence unit/source population, input qualification, full1706/at-most1248
vector coverage and18/6 physical geometry cells are inherited from XREF24.
No physical neural dose, scientific causal floor, sham floor or calibrated
causal power. Formal comparisons0, verdict REDESIGN_OR_PILOT_ONLY. No causal
reasoning, isolated adaptive advantage, equivalence or donor transfer claim.
The original168-case test stays sealed; the continuous goal stays active.
