# RAM-04 evaluation amendment A | fatal Lean subprocess exit recovery

**November 14, 2026. Post-score engineering disclosure.**

The original RAM-04 preregistration was committed at
`ae7be99b6367434f0dd724ae232cfa218ed99342` before any new
held-out compiler states or model scores were used. The train-only
GitHub Actions run `43593270279` completed and independently audited
4,973 compiler action effects. Nine CPU model checkpoints were frozen
before the test job began; their exact SHA256 hashes are in
`TRAINING_RECEIPT_20261114.md`.

The **first prospective scoring attempt was incomplete**:
[GitHub Actions #43593270279](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43593270279)
failed after the first three complete prospective theorem cases and
part of the fourth. At `primary/04/flat64`, the persistent Lean server
terminated during a local simplification tactic with the compiler
error `maximum recursion depth has been reached`. The original
RAM-03 `Session._response` raised a RuntimeError before returning an
action outcome. The original RAM-03 search handled bounded
`ProbeTimeout` restarts but did not recover from unexpected Lean
process exit during a request.

The incomplete original outcome is an **operationally invalidated,
partially observed score**. Its proof cases are excluded in their
entirety from the final prospective comparison. The full original
partial artifact `ram04-first-prospective-evaluation-43593270279`
(artifact ID `14113737113`, ZIP SHA256
`e7717f9c541cc569056d7ac72d1a3a3556f60685b52600b32035b5917bf97851`)
is retained for review, including compiler logs and generated sources.
Neither early successes nor the particular failed case determine the
published model ranking.

## Recovery implementation

A small RAM-04-only `CrashAwareSession` adapter translates a terminated
Lean subprocess during an otherwise valid registered tactic request
into a **recorded unknown compiler outcome** that enters the existing
RAM-03 bounded process-restart path. It preserves the original
candidate prefix, the fatal Lean stdout error excerpt and its local
source/file evidence. No failed or successful training label is
fabricated from the server exit. The search retains the original
request timeout ceiling, no more than two server restarts, all
original budgets, original candidate family ordering and original
argument generation.

Any compiler exception unrelated to a terminated Lean process
continues to propagate. The adapter never changes the Lean executable
options or increases recursion depth, so the theorem semantics remain
identical.

## Required re-evaluation

Run the **entire 12-case × 5-policy primary comparison** and the
registered **entire 12-case × 2-policy secondary comparison**, including
all theorems that happened to pass before the failure, using only the
original frozen CPU checkpoints from training CI `43593270279`.

The following are unchanged:

- original MathCan source commit, held-out theorem list and fixed order;
- all 4,973 audited train-only labels and development split;
- exactly the selected loop32 seed17, loop64 seed43, flat64 seed43
  best checkpoints with the SHA256s in the training receipt;
- 17-family parameterized candidate grammar, native/frequency policies,
  model architecture and utility ranking;
- primary 30-second,48-evaluation,8-transition budget; secondary
  60-second,128-evaluation,16-transition budget;
- full original proof, explicit axiom gate and independent source replay.

Every restarted server and compiler exit must be visible in per-case
evidence. The re-run uses a new, separately named GitHub Actions
workflow and output artifact. Any future operational repair requires
another complete comparison rerun and separate disclosure. Model-only
prospective theorem gain remains the previously preregistered
success criterion.
