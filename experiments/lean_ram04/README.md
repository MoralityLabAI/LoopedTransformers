# RAM-04: exact/rw/rcases supervision and new prospective proofs

RAM-04 branches from frozen RAM-03 c1265ea1718680be4827c2528ca0b39d5603d6b1.
RAM-02/03 sources, data, receipts and 36 checkpoints are preserved. All campaign
artifacts remain under D:\Research Engine\RAM-04; large files stay gitignored.
See PROTOCOL.md, the two explicit amendments and RESULTS.md for measured results.

## Supervision

Collect new labels from the 27 original TRAIN theorem roots plus 52 authored,
statement-only curriculum tasks. Hold out whole natural theorem groups and
complete authored template groups. Authoring a useful training statement does
not count as proving an original MathCan theorem. Report natural/authored family
support separately. Never consume original target proof commands or teacher
prefixes, and never relabel the old 961 or 4,973 observations.

Lean supplies accessible local names, types, proposition status, head constructor
and first function domain. Generate bounded local applications, symmetry,
projections, rewrite orientations/at-all-locations and And/Exists/Or elimination.
Named introductions retain accessible witnesses. Every controller gets the same
candidate pool. Independently replay every new acceptance verdict, complete
goal list and local metadata in fresh sessions with clean compiler exits.
Unknown/timeouts never become fabricated negative labels.

Require at least eight positive and negative rows per target family in both
splits before fitting. Remove exact visible-state overlap from optimization.
Check conflicting labels on identical visible inputs. Save support by origin.
Inputs exclude theorem/file/template/source-origin IDs, original commands,
prefix metadata and future outcomes. Pending contexts and referenced local types
are represented explicitly. Fixed sinusoidal position gates preserve argument
order; token hashing/truncation still lose information.

## Training and comparison

Looped32, Looped64 and Flat64, seeds 17/29/43. Batch 64 via microbatch 16 x4:
each target family contributes eight positive and eight negative samples; the
remaining 16 split single/multi-goal inputs evenly. AdamW .002/.01, clipping 1,
maximum 1,600 updates, validation every 100, patience 4. Weighted outcome CE plus
.2 SmoothL1 goal delta. Select using the mean of single/multi/exact/rw/rcases dev
losses only. Preserve all seeds, last/best optimizer and RNG states, class/family
metrics and checkpoint hashes. CPU/CUDA calibration uses identical loop64 updates
and a frozen 1.15x speedup threshold.

Only the twelve new MathCan theorems in config.json count toward proof coverage.
Their names were selected from positions 2/4/6 in four unused source modules.
Goals and reference proofs remain unobserved until implementation and selected
weights are frozen. Primary: native, frequency, loop32, loop64, flat64; 30 seconds,
48 logical evaluations, 8 path transitions. Secondary: native/loop64; 60 seconds,
128 evaluations, 16 transitions. Both use 8-goal caps, 3-second requests and at
most two server restarts. Startup, final complete compile and independent exact
replay are charged. No cross-policy verdict cache; no late proof gets credit.

An accepted proof must compile the complete original theorem with only selected
tactics, then independently compile the same source bytes. Explicitly audit
dependencies against propext, Classical.choice and Quot.sound. Reject sorryAx,
custom axioms, partial branches and scratch-only closure. Synthetic smokes and
training curricula are excluded from prospective coverage. Preserve negatives.

## Windows and resource limits

Reuse RAM-02 Python 3.13.7, PyTorch 2.6.0+cu124, psutil 7.1.3. Lean 4.35.0 and
MathCan c84e2025334e9a967c1297425a490f98d5b3d189; original gym
e8a780254b5cf41d479095f4f22bcfc4aa76ed72. No dependency upgrades or torch.compile.
The isolated recovered Lean executable is byte-identical to the preserved one.
Narrow tactic imports and synchronous elaboration with two Lean workers passed
preflight; broad import/startup and first-collection failures remain preserved.
The underlying full-import stall's cause is not established.

Enforce 6 GiB process-tree RSS, 2.5 GiB allocated CUDA, 2 GiB available RAM/commit
and 512 MiB dedicated GPU headroom. Sample every .1 seconds and nvidia-smi every
2 seconds. Record exact torch allocator peak separately. GPU holds microbatches
only. Redirect TEMP/TMP and all checkpoints to D:. Abort safely on resource errors.

Run the full campaign into a fresh timestamped directory:

```powershell
& "D:\Research Engine\RAM-04\LoopedTransformers\experiments\lean_ram04\run_windows.ps1"
```

Individual stages, from this experiment directory with PYTHONPATH pointing to
the pinned gym and LEAN_PATH to this directory plus MathCan/.lake/build/lib/lean:

```powershell
python build_probe.py --lean <lean.exe> --out <new-build-directory>
python -m unittest discover -s tests -v
python smoke.py --lean <lean.exe> --out <new-smoke-directory>
python smoke_families.py --lean <lean.exe> --out <new-family-smoke-directory>
python smoke_search.py --lean <lean.exe> --out <new-search-smoke-directory>
python collect.py --data <original-fc12-train> --mathcan <MathCan> --lean <lean.exe> --out <new-data-directory>
python train_local.py --data <audited-data> --out <new-training-directory>
python freeze.py --training <training-directory> --out <training-directory>/freeze.json
python evaluate.py --mathcan <MathCan> --data <audited-data> --training <training-directory> --lean <lean.exe> --out <new-evaluation-directory>
python analyze.py --data <audited-data> --training <training-directory> --evaluation <evaluation-directory> --out <new-analysis-directory>
python report_results.py --analysis <analysis-directory> --out RESULTS.md
```

On WSL/Linux use matching Linux Python/Lean, colon-separated LEAN_PATH and the
same pinned dependencies; Windows environments/binaries cannot be reused there.
Compact receipts and accepted sources are tracked. Checkpoint paths and hashes
are in receipts/checkpoints.json. Full machine-readable data, metadata, failures,
replays and the streamed SHA256 artifact manifest remain in runs/.
