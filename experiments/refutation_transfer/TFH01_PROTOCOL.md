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

## FC-12: exact original compiler-probe incidence

The original TRAIN-only 961-compiler-probe archive was recovered, allowing
the full selected four-family training incidence to be reconstructed.

- CI source run: https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43547041634
- GitHub Actions artifact: ID 14095030666; name
  fc12-train-typed-actions-43547041634; archive entry
  fc12_probe/train_candidate_outcomes.jsonl.
- Raw compiler result SHA256:
  c32ebf6d341a000c351b3724b817ec72b3284d82450d986835980b0f51aa626d,
  independently matching the original FC12_probe_receipt.json.
- Original training goal source SHA256:
  403331aba9c9f127f6ac169bb8b732de8a7b56e934f6cb78b4375d1c59732bf0.
  MathCan original-source commit:
  c84e2025334e9a967c1297425a490f98d5b3d189.
- The derived 51-state × 4-family complete positive edge matrix for
  this selected family set is pinned as
  [data/tfh01_fc12_verified_incidence.json](data/tfh01_fc12_verified_incidence.json).
  Each edge means accepted=true AND progress=true for at least one
  actual original-Lean train scratch probe, deduplicated by stage ID.
  Successful scratch progression is not a fully certified theorem.
- Distinct progressing states per family:
  exact_local=1, rcases_local=1, rw_local=10,
  induction_list=27. The published positive attempts had been
  1, 1, 13, 30 respectively.

**TFH-4 (exact frozen-data obstruction):** For hypothetical quotas
exact_local=13, rcases_local=13, rw_local=12,
induction_list=13, the tight Hall witness is

    S = {exact_local, rcases_local, rw_local}
    q(S) = 38, |N(S)| = 12,
    delta = 38 - 12 = 26.
    maximum filled slots = 51 - 26 = 25.
    one-pass TV lower bound = 26/51, approximately 0.5098.

The augmenting-path independent maximum-matching oracle finds exactly
25 filled slots and agrees with complete subset-cut enumeration.
A complete repair under these same four-family quotas needs at least
26 additional distinct training states eligible for at least one member
of S, through new successful original-Lean probes on existing
states or additional source goals.

The earlier bound derived from the published aggregate counts,
delta >= 24 for S={exact_local,rcases_local}, remains valid as
a weaker independently derivable certificate.

An additional balanced 12-family illustration with lexicographically
chosen quotas 5 for the first three families and 4 for the remaining
nine gives an independent 19-slot Hall shortfall (maximum filled=32).
That diagnostic uses all 12 original action families but is not the
four-family witness. Only the four-family exact source matrix is
persisted in this PR.

**Learning scope:** TFH-01 reports curriculum support and combinatorial
sampling barriers only. FC-12/FC-13 first-run heldout learning and
compute scores were negative and remain immutable. A Hall repair
would still require an independently registered heldout proof-gain
experiment.

## Reproducibility and guards

Run from repository root:

    python -m unittest discover -s experiments/refutation_transfer -p 'test_tactic_fiber_balance.py' -v
    python experiments/refutation_transfer/tactic_fiber_balance.py --six-cycle --sample
    python experiments/refutation_transfer/tactic_fiber_balance.py --fc12-published
    python experiments/refutation_transfer/tactic_fiber_balance.py --input experiments/refutation_transfer/data/tfh01_fc12_verified_incidence.json
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
