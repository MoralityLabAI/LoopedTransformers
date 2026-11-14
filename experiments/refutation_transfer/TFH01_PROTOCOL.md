# TFH-01 | Hall certificates and structural-zero tactic sampling

November 14, 2026. This is a separate prospective experiment. FC-12/FC-13
training, checkpoint hashes, evaluation scores and baselines are frozen.

## Sources

- OpenAI math family #113, *Entropy and Face Dimension of the Perfect Matching
  Polytope*, 2026-09-23:
  https://github.com/openai/math/blob/main/preprints/Entropy-and-Face-Dimension-of-the-Perfect-Matching-Polytope-September-23-2026/main.pdf
- OpenAI math family #115, *An FPRAS for Cell-Bounded Contingency Tables*,
  2026-09-24:
  https://github.com/openai/math/blob/main/preprints/An-FPRAS-for-Cell-Bounded-Contingency-Tables-September-24-2026/main.pdf

These manuscripts motivate the finite model. TFH-01 independently certifies
its own finite conclusions without assuming their broad asymptotic claims.

## Certified-coverage model

Let n distinct original Lean training states and k tactic families have
incidence E[i,j]=1 ONLY after an original-Lean compiler probe makes
progress. Every unprobed cell is unknown and omitted. A curriculum matrix
X has structural zero condition X[i,j]<=E[i,j], row sum 1, column sum
q[j], and sum_j q[j]=n. Distinct states may be used at most once.

**TFH-1 (Hall cut):** Write N(S) for distinct training states incident to
any family in subset S. Expand each family j into q[j] identical slots.
The max-flow/min-cut or Hall deficiency theorem gives exactly

    delta = max_{S subset families} (sum_{j in S} q[j]-|N(S)|)_+,
    maximum filled quota slots = n-delta.

All 2^k cut witnesses are enumerated. An independent augmenting-path
maximum b-matching must agree; otherwise the audit raises an error.

**TFH-2 (TV floor):** Let p assign one progressing label to each used
distinct training state, with an extra dummy label for unused states,
and let u assign masses q[j]/n to target tactic families. For any S,
p(S)<=|N(S)|/n and u(S)=q(S)/n, therefore

    TV(p,u) >= delta/n.

This quantifies diversity limited by distinct-state capacity, and does not
claim an obstruction to oversampling the same state with replacement.

**TFH-3 (2x2 mixing obstruction):** The legal 3x3 rows are {A,B}, {B,C},
{C,A}, all quotas one. Exactly two schedules exist: (A,B,C) and (B,C,A).
A 2x2 switch is impossible from either. The local-pair-only Markov chain
has two absorbing states and cannot uniformly sample the two-state fibre
from a single initialization. A global three-cycle maps between them.
Target entropy is ln 2; support entropy of a trapped chain is zero. The
TFH-01 sampler uses exact dynamic-program continuation counts, with a
hard 16-state cap, avoiding unsupported mixing-time claims.

## FC-12 consequence

[FC-12 immutable record](FC12_FIRST_RUN_20261114.md) reports 51 distinct
original train goal states and success ATTEMPTS: exact_local=1,
rw_local=13, rcases_local=1, induction_list=30. Success attempt counts
upper-bound distinct eligible state counts. The exact/rcases union has
at most two distinct states. Hypothetical quotas exact=13, rcases=13,
rw=12, induction=13 require 26 distinct targets on those two families.
The Hall shortage is at least **24**, with one-pass TV >=24/51. The
aggregate record does not contain the entire 51x12 incidence graph:
we do not fabricate its entries or claim model improvement. New compiler
probes on existing states or new source goals may expand eligible support.

## Reproducibility and guards

Run from repository root:

    python -m unittest discover -s experiments/refutation_transfer -p 'test_tactic_fiber_balance.py' -v
    python experiments/refutation_transfer/tactic_fiber_balance.py --six-cycle --sample
    python experiments/refutation_transfer/tactic_fiber_balance.py --fc12-published
    python experiments/refutation_transfer/tactic_fiber_balance.py --input train_matrix.json --sample

Training incidence JSON:
    {"split":"train","quotas":{"A":1,"B":1,"C":1},
     "states":[{"id":"...", "allowed":["A"]}, ...]}

The --probe-jsonl adapter additionally requires --quotas JSON and an explicit
split=train on all rows. It only inserts edges for source_kind=original_lean,
compiler_accepted=true and progress=true. The supplied logs must include
every intended state. No heldout test labels, synthetic progression guesses,
new training or model evaluation are part of this campaign. Counting and
uniform sampling stop at 16 states; the Hall cut and max matching remain
available above that threshold for up to 16 tactic families.

The pure mathematical Lean companion is in
patrickdugan/MathCan/MathCan/TacticFibers.lean.

Any FC-14 capability trial requires a separate frozen preregistration,
actual Lean objective, paired native baseline and heldout source-family split.
