# ARG-03 dev-only follow-up: time-local controls and final-R grading

**Prospective local experiment.** The original frozen protocol and the 168-case ARG-03 held-out test remain unchanged and sealed. All historical campaign receipts retain their November 15 provenance.

The existing Fable audit found that structured and random context disturbances had different state-displacement magnitudes at subsequent loops. The new time-local control starts from the same clean h_t, replaces only unscored context positions using the propagated structured state, and compares against a freshly sampled random context perturbation matched to its total per-case state L2. All three current readout positions are bit-identical before one more loop.

## Local commands (LoopedTransformers checkout, Windows PowerShell)

Set paths for the existing checked-out EvalMath, ARG-02 Compactification-Theory branch, and cached Nanbeige weights:

~~~powershell
$E = "C:/path/to/EvalMath"
$C = "C:/path/to/Compactification-Theory"
$R = "runs/dark-fibers/arg03/finalr_dev_20261117"
$CPU = "$HOME/.venvs/bluebeam/Scripts/python.exe"
$GPU = "$HOME/.venvs/bvg-v03/Scripts/python.exe"
$env:ARG02_SNAPSHOT_NANBEIGE = "C:/path/to/models--Nanbeige--Nanbeige4.2-3B/snapshots/451ed48c3273ecef7ea8faaa43c31ce529763bb1"

& $CPU -m mechinterp.dark_fiber_gyms.arg03_preunseal.prepare_final_r --evalmath-root $E --compactification-root $C --out-dir "$R/roster" --seed 2026111617

& $GPU -m mechinterp.dark_fiber_gyms.arg03_nanbeige --compactification-root $C --public-jsonl "$R/roster/development_finalR_fit_public.jsonl" --out-dir "$R/nanbeige_fit" --device cuda:0 --limit 0

& $GPU -m mechinterp.dark_fiber_gyms.arg03_nanbeige --compactification-root $C --public-jsonl "$R/roster/development_finalR_valid_public.jsonl" --out-dir "$R/nanbeige_valid" --device cuda:0 --limit 0

& $CPU -m mechinterp.dark_fiber_gyms.arg03_preunseal.score_final_r_future --evalmath-root $E --compactification-root $C --roster-dir "$R/roster" --fit-tape "$R/nanbeige_fit/answer_tape.jsonl" --valid-tape "$R/nanbeige_valid/answer_tape.jsonl" --out "$R/finalr_dev_future12.json"

& $GPU -m mechinterp.dark_fiber_gyms.arg03_preunseal.run_tlocal_dev --compactification-root $C --public-jsonl "$R/roster/development_finalR_valid_public.jsonl" --out-dir "$R/tlocal_valid" --device cuda:0 --limit 0 --width 8 --sigma 0.02 --seed 20261117

& $CPU -m mechinterp.dark_fiber_gyms.arg03_preunseal.score_tlocal_future --evalmath-root $E --compactification-root $C --roster-dir "$R/roster" --tlocal-tape "$R/tlocal_valid/tlocal_dev_tape.jsonl" --output "$R/tlocal_valid/grade_future12.json"
~~~

Use --limit 2 for initial GPU pilots, --limit 0 for the complete 48-case development partitions. The time-local runner has a dry-run mode. The fresh final-R generator refuses to overwrite its output directory. The original Nanbeige model-only runner stores 64 candidate scores at loops 1/2/4/8 and 16x3x4 raw operator-site logits for source-separated grading. Its weights and oracle source pins remain those of the previous ARG-02 campaign.

## Measurements and interpretation

The time-local runner emits per-case parity certificates, matched state-L2 doses, next-step full-vocabulary logprob divergence, and 64-program score changes at t+1 for t in {2,4,8}. Current full-vocabulary readout must be exactly equal up to 1e-6 and random-vs-structured L2 matching within 0.5%. Report paired continuous effects separately from operator-subspace rank effects; depth choice and amplitude must remain fixed before examining outcomes.

The independent future grader regenerates the new 48-FIT and 48-VALID development cases under the original source locks and verifies their public hashes. It fits a twelve-bias operator-site model only on FIT, then reports future-correct@12, accepted@12, and top-one future correctness on VALID against baseline 1/2/4/8, open-loop, equal-compute, and all present model controls. This adds task-based evidence to the original source-program-rank diagnostics. The 48 validation cases have different source-program identities from FIT.

The new time-local tape is model-only and scores the one-step continuation at t+1. The separate score_tlocal_future module grades all such t-local interventions on development VALID using the exact original EvalMath search and grader. It rejects incomplete tapes, nonmatching per-case SHA, dose or readout-parity violations, and any held-out test path. No real checkpoint inference has been run by this new follow-up code.

The next held-out test remains gated by explicit separate human approval under the frozen 168-case protocol. These development controls are preparatory and do not change the earlier confirmatory hypotheses.
