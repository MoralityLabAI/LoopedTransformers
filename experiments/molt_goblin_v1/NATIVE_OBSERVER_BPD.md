# MOLT-09: native BPD observer-map reuse and training continuation

Stacked on [MOLT-08](https://github.com/MoralityLabAI/LoopedTransformers/pull/406).
Engineering development, no promotion or confirmation. This deliberately uses the
existing AUX-V2 LOOP-T model and pinned native BPDTrainer rather than an SVD
surrogate for factor fitting. The teacher smoke has eight updates and the
behavioral decomposition smoke has two. Neither is a competence or factor
quality gate. The full teacher fit residual and prior BPD calibration state
must be retained. No locked split is opened.

## Reuse and physical coordinates

Fix the teacher checkpoint, one TRAIN anchor, visit 4 (one-based), its complete
reached state, last-token recipient, and learned BPD site blocks.0.mlp.2.
For each distinct TRAIN donor at the same generator depth, form the native
additive swap displacement d=(x_donor-x_anchor) V U at the physical MLP output.
The normal model still applies residual beta; it is never omitted from the
observer. These cross-item swaps are synthetic, not established source-valid
causal interventions. Compare the actual native swap_acts API to the dense
additive hook at every declared readout, including all vocabulary logits after
centering. Original dense weights and the undecomposed remainder stay in the
forward path. Bind the checkpoint, input, state, visit, site/observer contract,
factor file and coordinate chart. A new student checkpoint does not inherit
the teacher's local map as its own Jacobian.

Use an orthonormal output-span chart Q from the fitted U. The exact coordinate
baseline is c=Q^T d, so compression must beat the native factor-span baseline,
not just a 128-dimensional ambient write. Estimate the derivative of all
centered answer-position vocabulary logits from visit 4 through 8 with two
central-difference steps in float64. Reject disagreement between estimates
above 0.005 relative Frobenius norm. Compare rank-four SVD of the stacked
observer with rank-four SVD of only the final readout, inside the eight-factor
chart. There are no restricted-logit or reduced-observer claims. Readouts
outside the declared answer positions are not matched or tested.

For each donor compare actual projected and full swap trajectories. Admit
rank-four coordinates only when every observed visit argmax agrees and the L2
trajectory error is at most 10% of the full swap's displacement from the
clean trajectory; otherwise retain the full native coordinates. This is a
finite teacher-replay gate, not an outward-rounded error certificate. It
requires the original swap at ingestion and makes no promise for a new donor,
new reached state, or changed student weights.

Record all donors (including duplicates), distinct displacement count,
acceptance, errors, full native-coordinate payload, map bytes, and one-byte
record tags. Compute the candidate representation cost with fallback, and
choose all-native coordinates if map storage is more expensive. Do not claim
reuse savings from duplicate displacements. Account for the common Q basis
once and the shared full teacher response target bytes separately. At C=8,
k=4 the serialized local map costs 364 B; without fallbacks a rank-four
record saves 31 B against exact native coordinates after its one-byte tag,
so at least 12 genuinely distinct records are necessary to amortize the map.
The development program retains all compared representations, not a capped
production memory bank. Metadata, Python allocations, IO and optimizer memory
are excluded from this representation calculation.

## Continuation pilot

From the same initial teacher weights, compare ordinary task/trace training,
ordinary training with twice the optimizer updates, full native response
distillation, stacked-observer compressed response distillation, and terminal
observer compressed response distillation. Every arm uses the same frozen
anchor/donor roster, sample stream and optimizer defaults. Each distillation
step runs a clean and a patched student forward, compares their centered
future-logit difference with the *full* frozen teacher swap difference, and
adds the response MSE to the ordinary loss. The compressed arms use the
conditional projection/fallback policy regardless of storage planner choice,
to separate a training mechanism question from memory accounting. Student
training recomputes ordinary forward/backward and never applies a cached
teacher Jacobian as a gradient. Ordinary reinvestment matches forward
example-visits only, with different updates and backward cost. No live model
or full native BPD factor training is claimed for the continuation student.
Evaluation uses the original validation shard and no steering.

The fixed smoke uses one teacher, two TRAIN anchors, 16 distinct donor rows
per anchor, four continuation updates (eight for reinvestment), and two
examples per batch. It establishes engineering parity, recorded representation
cost and executable training, not a performance estimate for a trained
population. Full factor fitting with calibration, competent source checks,
source-valid donor construction, independent model roots, measured resource
caps and strict EvalMath preflight are prerequisites to a prospective
mechanistic or training claim. No protected readout or paid accelerator.

    python -m pytest -q tests/test_aux_observable_bpd.py
    python -m research_gym.interp.aux_gym_v2_observable_bpd --bpd-checkout .native-bpd \
      --data /tmp/molt09/data --run /tmp/molt09/teacher --factors /tmp/molt09/factors \
      --out /tmp/molt09/replay --anchors 2 --donors 16 --rank 4 --updates 4 --batch 2

Workflow produces a temporary receipt artifact. The DataGoblin map allocator
can consume measured native-span costs only after production provenance and
per-record eligibility have been independently checked.
