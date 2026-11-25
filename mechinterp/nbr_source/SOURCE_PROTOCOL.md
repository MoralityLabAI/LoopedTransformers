# NBR source-only candidate protocol

Parent: PR #351 at 412e32cfd9d52e4f2d61067cbf0d69bf8ebe9145.
The extended FP32 and simulated W4 toy checks passed run 45515740820;
pretrained pilots were in progress when this branch was cut. This stacked
branch does not modify or cancel them.

Freeze source selection before any inference on these candidate puzzles.
Seed 202611250901, Python 3.11; 24 singles-solvable 52-hole and 24
singles-stalled 56-hole unique puzzles per split cal/eval/rep (144 total).
Stalled requires >=8 blanks after synchronous naked/hidden singles.
Generate candidates in the fixed order cal/eval/rep, singles/stalled, slot
0..23; use at most 1200 source-only rejection attempts per slot. Reject
dig failures, wrong difficulty, exact input/solution duplicates, and
collisions of the necessary Sudoku symmetry invariants. Distinct invariant
values certify distinct input orbits under row/band/column/stack
permutations, transpose and global digit relabeling. Collision rejection
is deliberately conservative. Abort if a slot exhausts its cap or the
two exact solvers disagree; no post-inference replacement.

Historical exclusions: reconstruct LS-SAE02..07 from the pinned parent
source; recover LS-SAE11/12 from commit 4af25152b6b0cb0a5a87f86ba9a9f4b477d5a05a;
recover LS-SAE08 test source from Actions run 43851165980, artifact full.json,
SHA256 69c4cd1a2e09760ed2d463e598550bdd1da42fb1bc0ad83a8a95825c47bea7f9;
and LoopScope02 benchmark items and shots. Workflow verifies historical
file hashes. This covers named cohorts only; it is not a claim of
disjointness from every later or unpublished Sudoku campaign or training
data. Historical availability failure blocks admission. Earlier NBR
engineering roster (run 45515740820, SHA256
bcf9b3c47f5703d8e33222bbb66f34e057277476c44a555b896dc5325dd50a5b)
also must be excluded before the candidate is promoted. Do not expose
the candidate to any model until that last comparison and an independent
source review are complete. The source-only receipt rechecks two solvers
and the invariant-based exclusion, with no model imports.

Next: inspect both PRs' CI. Preserve the source artifact and its hash.
Add any omitted historical rosters (including NBR pilot) and reject an
overlap as a source admission failure; do not silently revise an admitted
manifest. Freeze the final roster and code/platform hashes in a separate
registration, including all inferential gates, before starting prospective
NBR inference. Class degeneracy in recovery is an incomplete endpoint;
do not choose replacements using model outcomes. Review the pilot
numerics and add a genuinely independent audit of the final source
manifest. No paper-level scientific claim is licensed by source admission.
