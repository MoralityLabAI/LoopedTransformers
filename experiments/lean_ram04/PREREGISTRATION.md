# RAM-04 preregistration | first independent-family theorem transfer

**Frozen November 14, 2026, before RAM-04 compiler evaluation or model fitting.**
RAM-01, RAM-02 and RAM-03 receipts, selected weights and released original
proof certificates remain immutable in their original commits. RAM-03's
first prospective suite is already public, and is therefore a **historical
diagnostic**, never the RAM-04 fresh evaluation.

## Question and immutable materials

Can a sub-250k-parameter recurrent tactic scorer, trained only on already
declared MathCan training theorem families, certify at least **one new,
original, family-held-out theorem that contemporaneous native search cannot
certify under identical search and Lean budgets**? The comparator is the
current RAM-03 17-family context-instantiated tactic grammar and persistent
Lean search. All result categories, including negative results, are
published. No unformalized mathematical conjecture is claimed.

Pinned original MathCan source: `patrickdugan/MathCan` commit
`c84e2025334e9a967c1297425a490f98d5b3d189`, Lean
`leanprover/lean4:v4.35.0`. Original source file, theorem, and referenced
gold proof commands are never candidate selection inputs.

Frozen original FC-12 training source: 27 theorem roots, 51 genuine goal
states, source-state SHA256
`403331aba9c9f127f6ac169bb8b732de8a7b56e934f6cb78b4375d1c59732bf0`,
961 compiler-probed outcomes, semantic effect SHA256
`bd713c7fd3d9d2e82f1304d7556f7a25bfa8b1a0e416dbd2e8d271dc4ad047ec`.
Source and original root training artifact comes from completed CI
`43547762837`. The artifact SHA256 digest is
`d91a12d725ca0a611b4de308c80365bfd3888d88449b03b2c5f07137a1c2025a`.

Reconstruct RAM-03's **train-only** expanded descendant-state supervision
from its unchanged collector and exact original config, within this
separate CI training run. Collect at most 10 genuine states per theorem
root, depth 4, max 8 goals. Probe every registered candidate script from
the compiler state. Independently replay **every** candidate label using
fresh Lean sessions before learning. Require positive branching-context
examples in train and development. Never synthesize successful labels,
never overwrite the original 961 effects, and preserve failed/timeout
observations as such. Hash the resulting JSONL, original config, and
independent audit. The reconstruction need not have a byte-identical hash
to the Windows RAM-03 run; require a fresh complete audit and record any
different effect or support counts. If critical coverage is missing,
stop the experiment and publish the source discrepancy.

## The first 12 RAM-04 prospective original theorems

Select the **first three top-level theorem declarations**, in original
file order, from each of the four following families. These four modules
were absent from all previous FC-11/12/13 imported training/held-out and
RAM-03's new prospective modules:

- `MathCan/OpcodeRefinement.lean`:
  `two_step_gross_budget`,
  `two_step_output_aggregation`,
  `boundary_liquidity_transition`.
- `MathCan/FoundryProbe.lean`:
  `two_nonzero_after_swap`,
  `two_currency_zero_at_lock`,
  `erc20_custody_neutral`.
- `MathCan/TOETwoFormTriad.lean`:
  `orthogonal_triad_isotropic`,
  `two_axes_missing_third`,
  `two_fixed_axes_not_positive_isotropic`.
- `MathCan/TOETripletJordan.lean`:
  `exact_source_factor`,
  `exact_lightcone_null`,
  `exact_lightcone_det_zero`.

Only the immutable source tree and theorem declaration names were used
to select these families; none of their future tactic-state traces,
reference scripts or outcome labels were viewed during design.
The upstream files are public, so this is a procedural source-family
holdout with time-ordered frozen model selection, not a cryptographically
secret benchmark. Keep the selector fixed regardless of difficulty.
Require all 12 original source proofs to compile separately as a 12/12
reference oracle, independent of model inference.

## Train and checkpoint freeze

Use the exact existing RAM-03 architecture/encoding and training,
with no score-driven new tactics or hyperparameter search:

- `loop32`: width32, four tied passes, ~85k parameters.
- `loop64`: width64, four tied passes, ~207k parameters.
- `flat64`: width64, zero recurrent passes, ~182k parameters.
- Seeds 17,29,43; select one seed per architecture using the registered
  balanced **training-family development** loss only.
- AdamW LR .002, weight decay .01, gradient clip 1.0, batch16 x
  accumulate4, at most 1,600 updates, validate every100, patience4.
- Fixed RAM-03 five-theorem development split. Development checkpoint
  selection precedes access to any RAM-04 held-out theorem goal.
- Multigoal-input sampling fraction 0.5, weighted four-outcome CE and
  auxiliary goal-count delta loss weight0.2.
- Hash all 18 best/last model checkpoints, optimizer/RNG state and train
  manifest. Train only on true compiler observed actions, with original
  family overlap and state-hash leakage rejection.

The baseline policies are a fixed native candidate order and a frequency
ordering calculated from **training only**. The models and baseline
receive exactly the same candidate pool and search feedback. Native
does not get model logits; frequency does not use development or test
labels. The RAM-03 renderer, grammar, and persistent server are fixed
before RAM-04 held-out evaluation.

## Frozen search and exact theorem certification

Primary: **30 seconds, 48 logical tactic evaluations, up to 8
successful steps per path**, maximum 8 remaining goals, up to two
persistent server restarts. Each request max 3 sec, 0.5 sec reserved for
shutdown. The original source import/session startup, final Lean
compilation, and exact source independent replay consume the budget;
report actual processes and total Lean work separately.

Secondary (registered before any result): **60 seconds, 128 evaluations,
up to 16 steps** for the exact same 12 source statements, native and
development-selected loop64 only. Report secondary separately; changes
to the primary score are forbidden.

Full positive proof: compile the complete original theorem with only
the selected script, no `sorry`, `sorryAx`, `admit`, or trusted extra
axioms. Only `propext`, `Classical.choice` and `Quot.sound` are accepted
by the explicit `#print axioms` gate. Independently replay
byte-identical final Lean source and verify its SHA256. Both certificates
must complete before the time cutoff. Multi-goal induction and case
branches require every sibling goal closed. Preserve any startup
timeouts, unclosed goals, exhausted budgets, errors, and failed models.
Every policy visits its own independent source/server state.

## Success and cost

Primary criterion: at least **one model-only prospectively certified
original theorem** for either recurrent architecture at the primary
matched budget, and no false positive certificate. Also report whether
a recurrence-specific success exists: one solved by loop32/loop64 and
missed by *both native and flat64*, with all budgets identical.
A model beating native by count alone while flat/frequency also match
is a policy-order improvement, not recurrence-specific evidence.

Cost (secondary criterion) is separately measured wall time and
subprocess/evaluation count per original Lean-certified theorem,
including failures. Keep the all-policy detailed case matrix.

Any implementation/scoring repair after first held-out inspection must
preserve original attempts and produce a separately labeled full
comparison rerun; neither frozen theorem set nor model weights/seeds,
grammar or budgets may change. Original RAM-03 source and score
artifacts are never rewritten. Post-hoc FC-13 experiments, if performed,
must be labeled outside this primary result.

## Memory/compute

Support native Windows RTX3050 4GiB VRAM with <=6 GiB measured whole
process-tree RSS, <=2.5GiB allocated CUDA VRAM, >=2 GiB free RAM/commit,
and >=512MiB free dedicated VRAM. If running GitHub Actions CPU:
pin Python, CPU PyTorch, Lean4.35, and memory environment. Compare a
short identical CPU/CUDA training calibration only when CUDA exists.
Use CPU if CUDA throughput is below 1.15x CPU.
Never claim the cloud runner had RTX3050 hardware.
