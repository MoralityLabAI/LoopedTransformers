# OLR-08B: adaptive proof-frontier allocation (development registration)

Status: control infrastructure; no native results and no architecture promotion.
Stacked on PR #318 at 8931f9d9cf0b064cea5cb3ad0760fe2dcd5197fa.
OLR-08A's original failed gate is immutable.

## Why this rung

OLR-08A's hand prior closes 4/18 hard CAL roots, counted prior 3/18,
and counted prior at 1,200 attempts 5/18. The counted prior solves two
FinRange roots at two attempts each but misses a root that the hand prior
closes at 83 attempts. That is a posthoc signal for measuring strategy
allocation, not evidence for a learned policy. The smoke-observed evaluation
root is not pristine; this campaign opens none of the 275 evaluation roots.

## Fixed execution contract

Reuse the pinned clean Lean checkout and repaired OLR-08A Session server.
Run the native scope sentinel and human/menu checks before search.
Each root/arm owns a new native process. Multiple queues within one
portfolio share one native meter and immutable root snapshot; all attempted
tactics, premise requests and certifications are charged. Literal tactic
dedup only. No cross-path state merge or learned state identity.
Fresh independent certificate replay, including recorded exact? warmup,
is required for every credited closure.

Policies: S1 uniform, S2 hand prior, S3 counted prior, S3c cost-aware
counted prior, RR deterministic four-queue round robin, UCB-style heuristic
with a fixed open-child progress proxy and raw-heartbeat cost penalty.
The UCB heuristic is not a trained utility predictor. It observes only
completed attempts in the current episode; a failed closure certificate
does not count as progress. All queues share a single total budget.

Budgets: 75/150/300/600/1200 attempted tactics. RPC and raw-heartbeat
caps scale proportionally from 1200 and 512M at 300 attempts.
The budget grid is spent-cohort exploratory only. Depth 3 is the
primary protocol. Depth 5 and wider premise caps (64 rw/16 apply requested,
40 rw/10 apply retained) are separate spent-cohort diagnostics.

Memory tiers: 0/64/256/1024/4096 MiB aggregate serialized menu-cache
payload cap, divided by worker count. The exact cache keys only live
session snapshot IDs and stores successfully enumerated menus as immutable
JSON bytes. No timeout responses, tactic results or certificates are cached.
A cache hit avoids an otherwise repeated native premise request; timing,
hits, misses, evictions and payload high water are logged. Python overhead,
Lean snapshots and queues are subject to the independent 12 GiB process-tree
cap. The tiers do not force allocation. Cache sizes may be indistinguishable
if the workload never fills the small tier, particularly because cross-queue
reuse may occur mainly at the root. This is the transparent memory control,
not a BPD feature-map implementation.

## Phase A: spent diagnostic

Run only the previously scored calibration registry from 08A (including
the disclosed smoke overlap). Re-run menu validity, execute all 40 specs,
and report each solve curve, paired effects and native charges. The stage
uses the immutable 08A TRAIN-fitted counted prior. It cannot establish
fresh efficacy or justify opening evaluation. Do not revise the new
calibration contract using its outcomes without a new registration.

## Phase B: prospective calibration feasibility and headroom

The candidate list is calibration_roster.json: all 64 nested Init/Data
files in the pinned inventory after excluding the 69 earlier spent files,
all files in the original 08A registry and declared fresh roster, and
all six SInt files. Root-level modules are not included because the
current 08A trace selector expects family directories. Family overlap
is expected. Files are clusters, not proven independent samples.

Use the original 08A deterministic T/R selection rules and exclude all
previous exact goals/declaration names (including evaluation registry
metadata, never its outcomes). No resampling after menu results.
Freeze the selected registry and all exclusions before any native
menu/search outcome. Commit the roster and registry seal externally
before supplying the digest to the fresh-calibration command.
The new cohort must include at least 120 valid menu-hard roots in
at least 12 file clusters; at least ten T-human audits and >=95% valid
human closures. Otherwise stop after the menu feasibility report.
Registry insufficiency stops even before menu outcomes.
No fresh EVAL code path exists.

Run all four 300-step controls, RR/UCB at all five memory tiers, and
S3 at 1200 as the offline reference (15 total specs). Require the
strongest 300-step control to solve <=80% and the longer reference to
improve by >=10 percentage points. Compare over the identical roots.
Compute retrospective union over the controls as an additional optimistic
portfolio bound. It uses future outcomes and is never deployable.
Report all file-cluster bootstrap comparisons as exploratory when the
control is selected on the same cohort.

A passing gate authorizes a separately registered learning campaign,
not a model promotion or automatic training. The thresholds belong to
08B and do not alter 08A. Preserve the original evaluation and SInt.

## Phase C: proposed learning rung (not executed or implemented here)

On a separate reused TRAIN roster, collect matched metered search prefixes
for every prior and independently certify continuations. Store only causal
prefix features, remaining budget and exact outcome receipts. For an
unresolved prefix use a censored target; never call an unvisited action
mathematically false. Fit a budget-conditioned closure-hazard and expected
cost predictor. A direct MLP, recurrent reader, matched-compute
direction/shuffle shams and exact memory baseline are mandatory.

BPD features must be fit on TRAIN only using a pinned BPD implementation.
Compare learned low-rank component maps, raw hidden features, random
rank-matched maps, and exact receipt memory. No component score can
certify a proof. Use the same RAM tiers, report feature construction,
maintenance, inference, full native wall time and matched total budget.
Freeze policy and gates before a separate confirmation. Do not scale
to 500K or 5M on the present data.

## Commands

From the repository checkout (PowerShell; set paths to local installation):

    $g = "research/lean_cgrt_hrm400/olr08b"
    $source = "D:\lean4-419-source"
    $exe = "research\lean_cgrt_hrm400\olr08\lean\.lake\build\bin\lgym.exe"
    python -m unittest discover -s $g -p "test_*.py" -v
    python "$g/campaign.py" --mode spent --source $source --exe $exe --out C:\olr08b\spent --threads 8 --workers 2 --ram-gib 12
    python "$g/campaign.py" --mode freeze-new --source $source --exe $exe --out C:\olr08b\cal --roster "$g/calibration_roster.json" --threads 8 --workers 2 --ram-gib 12

After auditing and committing the frozen registry/digest, use the exact digest
from registry.sha256 (do not overwrite or edit the frozen registry):

    python "$g/campaign.py" --mode fresh-calibration --source $source --exe $exe --out C:\olr08b\cal --expected-registry-sha256 <committed_digest> --threads 8 --workers 2 --ram-gib 12

The Linux command needs an enforced cgroup memory.max <=12 GiB, e.g.
prefix with systemd-run --user --scope -p MemoryMax=12G. Use an isolated
process environment. No blanket python.exe kills. Keep outputs on a drive
with adequate space. Native runs require the 08A binary and psutil.
A checkpoint can be resumed only under the identical contract.

Return summary.json, contract.json, resource.json, registry seals and
per-root SHA-256 checkpoints (including replay records). Archive the
run directory without rewriting its bytes. Report failures rather than
relaxing the gate.
