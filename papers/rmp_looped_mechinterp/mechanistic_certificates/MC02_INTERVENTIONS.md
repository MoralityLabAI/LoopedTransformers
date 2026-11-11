# MC-02: counterfactual continuation certificates

2026-11-11. Additive successor to MC-01 at
`0496846d63c82e19b64173a03d2cf3a0ff74925b`. The implementation imports the existing
SEM-ECHO full-state adapter and frozen LoopedChase checkpoint format. It changes
no model, trainer, or existing assay. The tests and smoke use an untrained model.
Scientific measurement on the external Q4/Q5 checkpoints is still outstanding.

## Mathematical obligation

Let x identify the receiver's complete immutable context, including its graph
and reinjected prompt. Let z identify a donor context. A candidate abstraction
alpha_x maps complete neural states to a symbolic cursor, and T_x advances the
cursor under x's graph. J_(x,z)(h,k) is a specified low-level patch using receiver
state h and donor state k. Two equations are required:

    alpha_x(J_(x,z)(h,k)) = alpha_z(k),                      (lift)
    alpha_x(F_x(h)) = T_x(alpha_x(h)).                      (step)

**Proposition 1 (counterfactual continuation).** Suppose the patched state and
its next H iterates lie in a domain on which the step equation holds. If the lift
equation holds, then for every 0<=r<=H,

    alpha_x(F_x^r(J_(x,z)(h,k))) = T_x^r(alpha_z(k)).         (1)

**Proof.** The lift equation is the r=0 case. Assume (1) at r. Applying the step
equation to the current patched trajectory gives the left side at r+1 as T_x
applied to the left side at r. Substitute the induction hypothesis. QED.

The immutable context remains x throughout. This matters for LoopedChase because
its embedding is reinjected at every loop. Copying a donor state while also
copying its prompt exercises a different transition F_z. MC-02 uses that path
only as a native-replay control and labels its context explicitly.

The equation is a standard causal-abstraction obligation specialized to the
current recurrent model. MC-02 contributes an executable contract and controls;
no novelty claim is made for interchange interventions or semiconjugacy.

## A two-graph challenge with an answer fixed before replay

At visit t, choose receiver and donor examples with correct current native-head
predictions b and d. Their public graphs are A and B. Require

    b != d,        A(d) != B(d),        A(d) != A(b).         (2)

For permutation graphs, b!=d implies the third condition. The code checks all
three explicitly. The predicted patched continuation is

    d, A(d), A^2(d), ..., A^H(d).

The receiver's ordinary continuation begins at b and uses A. Donor replay starts
at d and uses B. Pair selection reads graph inputs and current predictions only.
It never reads a resumed outcome or future network label. Counterfactual oracle
paths are the declared target and remain outside the neural computation.

**Proposition 2 (copy separation).** Under (2), a system that returns the donor's
ordinary first successor, or that ignores the intervention and returns the
receiver's ordinary first successor, fails the required first transition.

**Proof.** The required successor is A(d). Equation (2) excludes B(d) and A(b).
Both are therefore incorrect. QED.

The negative controls are guaranteed to fail by construction. Their rejection
validates the checker and challenge design; it supplies no neural discovery.
All 144 eligible graph/cursor combinations on three-node permutations are
checked at H=4. The true traces pass. Donor copies and ignored interventions each
fail all 144 cases. In 60 of the 144 cases the donor copy still has the correct
final answer. A full trajectory check catches those endpoint collisions.

**Corollary (context-blind limit).** Fix a donor cursor d and m receiver contexts
whose required next cursors are all distinct. A context-blind predictor emitting
one cursor succeeds on at most 1/m of the uniformly weighted contexts. A random
context-blind predictor obeys the same bound in expectation: summing the output
probabilities over m distinct required symbols is at most one. This bound assumes
that its output distribution is independent of the receiver context. MC-02's
initial panel enforces pairwise copy separation, not this balanced m-context
experiment; the corollary is a possible extension.

## Approximate continuation theorem

Suppose the abstract space has metric d_Q, T_x is L-Lipschitz on the relevant
abstract domain, the lift error is at most eta, and the one-step commutation
error on the actual patched trajectory at step j is at most epsilon_j. Then

    e_H <= L^H eta + sum_(j=0)^(H-1) L^(H-1-j) epsilon_j,  (3)

where e_H compares the decoded patched state at H with the predicted abstract
state. Define e_0 by the lift error. The triangle inequality and Lipschitz bound
give e_(j+1)<=epsilon_j+L e_j. Induction expands that recurrence to (3). This
requires bounds on the visited patched states; natural-trajectory averages do
not establish them. In the finite label checker the exact success predicate
requires every decoded state to match, including the post-patch boundary.

The current measurement has finite coverage. It cannot certify equation (1) for
all states or all intervention compositions. A failed cursor-only patch rejects
that lift/readout hypothesis on those states. A distributed algorithm may need
a wider or context-adapted lift. Off-manifold damage is another possible cause.

## Native experiment contract

The first candidate copies the donor's cursor-position residual into the receiver
at the same visit. All other token positions and the receiver's fixed embedding
are retained. Its immediate head logits are checked against the donor's head.
The complete carried state is used during continuation.

Each selected pair runs cursor transfer, sham, full donor with its own embedding,
a constant shift across cursor features, and norm-matched random cursor edits.
The default is eight random directions. Raw edit norms and norms after the native
`inject` LayerNorm are recorded, because matching the first does not imply
matching the second. The shift control tests the LayerNorm invariance in this
architecture. Sham and full-donor continuations must reproduce the separately
captured native state and logits within the declared float32 tolerance.

Only cursor_transfer is the tested semantic lift. Every arm retains its labels
and scores against the same counterfactual target; donor replay is explicitly a
control executed under B, not an A-conditioned intervention. Oracle correctness
under both original contexts is also recorded.

A selection manifest is written and hashed before any intervention. The seeded
cap samples receiver/visit slots, then chooses one eligible donor per slot.
Coverage includes all requested slots and all endpoint-correct slots. No eligible
pairs produces an undefined primary score. Receivers receive equal weight after
averaging their selected visits. Donor reuse couples observations, so the report
makes no iid pair confidence claim. Visit-level counts remain separate.

Q4/Q5 are previously inspected development models. The default future horizon
is 4, with visits 1,2,4,8,16 and seed 940211. This is a development contract;
confirmatory extrapolation claims require new checkpoints after freezing the
whole selection/decoder/intervention pipeline. A future threshold must be fitted
on a separate calibration panel.

## Circuit and source contract

Each candidate trace uses MC-01's existing pointer relation with public statement
(receiver graph, donated cursor, horizon, required final cursor). The private
witness contains the observed intermediate labels and quadratic products. MC-02
checks the finite-field constraints directly in Python. It adds no new Circom
compiler claim, STWO proof, or garbled evaluation.

This audits the hypothetical symbolic continuation. A private witness in that
relation remains meaningful only under MC-01's source assumptions. The runner
retains complete clean/replayed states and exact source/checkpoint hashes, with
native parity controls. Honest execution is trusted; neither hashes nor a passed
symbolic relation cryptographically proves that the supplied activations came
from the model. The full bridge requires a binding to specified numerical neural
inference, or a separately trusted attestation mechanism.

## Commands

```bash
python -m unittest discover -s tests -p test_intervention_certificate.py -v
python -O -m unittest discover -s tests -p test_intervention_certificate.py -v
python -m research_gym.scripts.loopscope_intervention_certificate \
  --controls --out /tmp/mc02-controls
python -m research_gym.scripts.loopscope_intervention_certificate \
  --smoke --items 16 --visits 1,2,4 --horizon 3 --max-pairs 4 \
  --random-controls 2 --out /tmp/mc02-smoke
```

Existing Q4/Q5 checkpoint, with an identity pin from its trusted training receipt:

```bash
python -m research_gym.scripts.loopscope_intervention_certificate \
  --checkpoint /path/to/lam0_s4.final.pt \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --items 64 --visits 1,2,4,8,16 --horizon 4 --max-pairs 16 \
  --random-controls 8 --device cuda --out /path/to/new-mc02-run
```

Missing checkpoints and incorrect hashes fail before execution. The output
folder must be empty, preventing accidental replacement of a frozen manifest.
`--smoke` emits status `UNTRAINED_SMOKE` with scientific_summary=null. Smoke
fallback pairs are explicitly ineligible and excluded from every scientific
aggregate. No model is trained by this package.

Files: selection.json, clean_trace.npz, replays.npz and metrics.json. The NPZ
archives contain the full replay tensors for independent inspection. The result
JSON records every per-arm future and constraint failure.

## Primary sources and repository dependencies

- Geiger, Lu, Icard and Potts (2021), Causal Abstractions of Neural Networks:
  https://arxiv.org/abs/2106.02997 .
- Geiger et al. (2022), Inducing Causal Structure for Interpretable Neural Networks:
  https://proceedings.mlr.press/v162/geiger22a.html . Alignment and interchange
  intervention targets precede this implementation.
- Circom constraint-generation specification:
  https://docs.circom.io/circom-language/constraint-generation/ .
- Native LoopedChase and SEM-ECHO adapter at
  80d2ce8d073c6b634eab56f94c3f24c1a3c3a9e1, plus MC-01 at
  0496846d63c82e19b64173a03d2cf3a0ff74925b. Existing output-witness PR #6 has a
  distinct assay for future sensitivity under unchanged current outputs.

The proofs above are handwritten. No Lean kernel check has been performed.
