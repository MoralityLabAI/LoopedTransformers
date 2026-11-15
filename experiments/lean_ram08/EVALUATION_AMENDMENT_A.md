# RAM-08 evaluation amendment A | hosted runner dispatch fault, before test scoring

**November 15, 2026. This amendment was recorded after source-only training and before any RAM-08 original heldout theorem was scored.**

The original [RAM-08 preregistration](PREREGISTRATION.md), commit
`b8001e50e89401b44c8acb276f4c15e02c4b5b05`, fixes the old and
new MathCan source revisions, exactly 12 PEF theorem declarations,
20-family shared action grammar, three non-neural comparators, nine
small models/seeds, development-only model selection, complete theorem
proof gate, and strict 30s/48/8 primary and 60s/128/16 secondary
budgets.

The original two-job [CI run 43689874169](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43689874169)
completed its old-MathCan **TRAIN** job successfully. It independently
replayed **20,930 original MathCan compiler candidate effects**, with
zero compiler replay disagreements. Positive original MathCan exact
examples were obtained from three distinct real theorem roots:
`Intervention.obsEqOn_trans`,
`TOEMixedPrincipal.independent_kinetic_and_spectral_gates`, and
`TOENonAbelian.zero_mixed_nonzero_source`. There were eight
recorded positive candidate occurrences representing five distinct
(theorem, state-hash, script) observations. Thus the preregistered
two-root/three-unique-positive diversity gate passed.

All nine original small CPU models were fitted and the selected
development-only checkpoints were frozen and uploaded in the
original `ram08-frozen-audited-train-43689874169` artifact,
GitHub artifact ID `14152063191`, original ZIP SHA256
`a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc`,
size 20,496,574 bytes. Its model selections and exact checkpoint
SHA256 are:

- `loop32` seed29, `4a88bc80456743a89a70882911060cc90e76785a5ae1d435a0e46a8f8c09778f`
- `loop64` seed43, `5cbd327904714ac698897e1c2333cae9ec4125aeb70e7c56cd6236db8a737dac`
- `flat64` seed29, `2aa101edbd56f6563ebb2edcfc8c61c07dcc027559ea4cdc551187d6ec5303d8`

The exact original train corpus SHA256 is
`f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f`.
Hosted CPU calibration measured 53.6123 optimizer steps/second.

The dependent `first-new-family-original-proofs` job was reported
**failed with no executed steps** in the run's attempt records.
GitHub's returned job-step list was empty, and its job logs were
unavailable (BlobNotFound). Retrying only the failed job also
ended with **no executed steps**. There is consequently no
reliable evidence that the original heldout MathCan checkout or
any of its goal states was opened, and **no scored model or native
theorem result** exists from this initial dependent job. Its failure
is treated as a runner-dispatch/infrastructure blocker, not a
model/search outcome. The original completed train job and
artifact remain immutable.

To execute the original frozen preregistered test independently,
the dedicated [RAM-08 evaluation-only workflow](../../.github/workflows/lean-ram08-eval-frozen.yml)
downloads the **original** SHA-frozen training artifact by its
source run ID, verifies the original three selected checkpoint
byte hashes, and checks out only the pinned new MathCan PEF
revision in this fresh runner. It keeps the original code,
original new theorem names, shared candidate grammar and strict
full-score budgets. It must run all **12 × 6 primary** and
**12 × 4 secondary** searches, plus a separately independent
Lean source replay of every accepted proof. No model refitting,
data-provenance change, test-driven script changes, partial-case
resumption, or result credit from the empty failed job is allowed.

If this independent evaluation also encounters a hosted
runner failure, preserve that failure and leave the original
training artifact intact. This is an **operationally disclosed
whole-suite first test execution**, not a scientific rerun
after observing heldout model results.
