# RAM-03: branching supervision and bounded Lean search

RAM-03 preserves RAM-02 and adds a separately audited corpus of whole-goal-list
transitions, explicit goal-count supervision, balanced branch sampling and a
persistent Lean probe backend. Mathematical success requires a separate complete
theorem compile and independent exact-source replay. See PROTOCOL.md and RESULTS.md.

The feature branch starts at RAM-02 a3e7be130334ecf86b8f0befc912f4864418d385.
All work and large artifacts are under `D:\Research Engine\RAM-03`.
RAM-02's original branch, checkpoints, source and scored receipts remain unchanged.

## What changed

- Collect only from the 27 original FC-12 training theorem roots. No original
  target reference proof commands or expert-prefix actions enter collection.
- Observe entire ordered sibling-goal lists and visible definition constants.
  Probe every candidate and independently replay every new label in a fresh
  Lean process before fitting. Keep old observations immutable.
- Hold out five whole theorem groups, including branching examples. The network
  receives active context/conclusion, pending conclusions, candidate/family and
  goal count. Predict outcome class and clipped net goal-count decrease.
- Train Looped32, Looped64 and Flat64 with seeds17/29/43; half each effective
  batch is drawn from observed multi-goal states. Select weights and one seed
  per architecture by balanced development loss alone.
- Generate visible-definition unfolding, list induction/generalization, case
  splitting, core integer simplification and all-goals simplification candidates.
  Native and learned policies share the exact same registered grammar.
- Probe in a persistent Lean session that restores the original root for each
  full candidate prefix. Lean elaboration isolates stdin/stdout, so transport is
  atomic JSON mailbox files with bounded I/O retries, not a pipe. Reset messages
  and heartbeats between requests. No cross-policy verdict cache.
- Primary:30 seconds,48 logical evaluations,8 transitions per path. Secondary:
  60 seconds,128 evaluations,16 transitions. Both were frozen before scoring.
  Startup, final compilation and independent replay are charged to the budget.
  Count actual processes separately from logical evaluations. Reserve0.5s for
  shutdown; require all branches to close and every replay to succeed.

The prospective set is twelve declarations selected by source order across
FlashAccounting, V4Transient, TOEFormSectors and TOEMultipletKinetics. FC-13's
sixteen already published cases remain a separately labeled post-hoc regression.
Do not treat gains over RAM-02's historical native6/16 as learned-policy gains
unless they also exceed the contemporaneous native comparator at equal budgets.

## Environment and limits

The measured run reuses Python3.13.7, PyTorch2.6.0+cu124 and psutil7.1.3 through
the existing RAM-02 venv. No dependency upgrades. Lean is4.35.0; MathCan is pinned
to c84e2025334e9a967c1297425a490f98d5b3d189 and the original gym to
e8a780254b5cf41d479095f4f22bcfc4aa76ed72.

`runtime.py` imports RAM-02's immutable resource guard by exact file path without
altering Python module search precedence. Caps:6 GiB process-tree RSS,2.5 GiB
allocated CUDA memory, at least2 GiB available RAM/commit and512 MiB dedicated
GPU headroom. Monitor RSS/available memory every0.1s and nvidia-smi every2s;
torch separately records its allocator peak. Abort on resource errors. No
torch.compile, GPU compilation or bulk GPU dataset preloading. Compare identical
FP32 CPU/CUDA updates and require1.15x CUDA speedup; otherwise train on CPU.

On Linux/WSL2 use matching Linux Python/Lean installations, colon-separated
LEAN_PATH and the same pinned dependencies. Compile ProbeServer.lean locally;
do not assume Windows binaries or virtual environments work inside WSL.

## Commands

For this machine, `run_windows.ps1` builds the helper, runs tests and executes
collection, full independent audit, training, both proof comparisons and analysis
in a new timestamped directory. It preserves all earlier runs.

For individual stages, set PYTHONPATH to the pinned Compactification-Theory
checkout and LEAN_PATH to this experiment directory plus MathCan's existing
`.lake/build/lib/lean`. Put the pinned Lean bin directory on PATH; set
LEAN_NUM_THREADS=2, OMP_NUM_THREADS=2, MKL_NUM_THREADS=2 and PYTHONHASHSEED=0.
Redirect TEMP/TMP to the artifact drive. From this experiment directory:

```powershell
lean ProbeServer.lean -o ProbeServer.olean
python -m unittest discover -s tests -v
python smoke.py --lean <lean.exe> --out <new-smoke-dir>
python smoke_search.py --lean <lean.exe> --out <new-search-smoke-dir>
python collect.py --data <original-fc12-train> --mathcan <pinned-MathCan> --lean <lean.exe> --out <new-data-dir>
python train_local.py --data <audited-data-dir> --out <new-training-dir>
python evaluate.py --mathcan <pinned-MathCan> --data <audited-data-dir> --training <training-dir> --lean <lean.exe> --out <new-evaluation-dir>
python analyze.py --data <audited-data-dir> --training <training-dir> --evaluation <evaluation-dir> --out <new-analysis-dir>
python report_results.py --analysis <analysis-dir> --out RESULTS.md
```

Install dependencies into an isolated environment only when missing:
`torch==2.6.0` from the official CPU or CUDA12.4 wheel index, plus `psutil==7.1.3`.
The current measured environment required no installation.

## Integrity and artifacts

Original state SHA256:403331aba9c9f127f6ac169bb8b732de8a7b56e934f6cb78b4375d1c59732bf0.
Original effect semantics:bd713c7fd3d9d2e82f1304d7556f7a25bfa8b1a0e416dbd2e8d271dc4ad047ec.
The loader checks these before collection. New transitions have a separate
dataset fingerprint, schema, complete before/after observations and source hash.
The independent audit checks every new acceptance verdict and canonical result,
including state/definition agreement. A timeout/transport failure is unknown,
never fabricated into a negative training label. Exact cross-split state overlap
is excluded from optimization with a receipt; theorem groups stay intact.

All scratch/session results are untrusted. Full release sources import the
original context and selected tactics without ProbeServer or sorry. Audit only
propext, Classical.choice and Quot.sound; reject all other dependencies. Archive
every independent replay and byte hash. Toy smoke proofs are excluded from
research counts. Full data, mailbox files, optimizer/RNG checkpoints and compiler
logs are under gitignored `runs/`; compact receipts and deduplicated accepted
Lean sources are tracked for review.

Interrupted collection attempts remain preserved. Engineering fixes addressed
Lean's isolated streams, Windows sharing races and a Python import-path collision;
none used prospective outcomes to change model parameters or search policy.

The first evaluation attempt was invalidated after a heartbeat option was found
between attached documentation and theorem declarations. The repaired wrapper
keeps documentation attached, requires each probe process to exit cleanly and
refuses reused mailboxes. A subsequent audit caught cumulative heartbeat usage;
the outer session is unlimited while every request retains an independent
100,000-heartbeat limit and 3-second wall timeout. A complete additional replay
of all 4,973 unchanged labels passed before the corrected evaluation. Data bytes,
weights, seed selection, grammar and budgets stayed frozen. These engineering
repairs occurred after initial prospective outcomes were observed; the corrected
comparison is explicitly a rerun. See RESULTS.md for the measured outcome.
