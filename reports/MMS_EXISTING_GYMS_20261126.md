# MMS-01: opt-in mechanistic memory scaling in existing trainers

Development implementation only. No claim of improved task accuracy or causal
BPD discovery. A zero budget preserves existing training and RMP cell hashes.
A nonzero budget holds a bounded BF16 bank of detached, previous-step output
cotangents, addressed by training example ID and visit at a shared MLP write
site. The current V/U map selects top-k normalized write directions by product
of read amplitude and historical gradient alignment. An admitted feedback
term is norm-matched to a fraction of the current output gradient. Same-step
and expired entries are excluded. Cache collision tags prevent mistaken
identity hits. Dense and rolled-coordinate sham use the same admission
criterion within an arm; admissions may diverge between trained models.

Modes: observe (cache and no feedback), dense (historical gradient),
bpd (component-filtered), sham (coordinate-rolled component signal).
The ordinary baseline has mms-mib=0. Rank-16 SVD is explicitly a seed proxy.
For learned BPD maps, pass a torch-save dictionary with CPU V[in,C], U[C,out]
from the exact named matrix and model lineage via --mms-map; shape and
SHA-256 checked. Weight cosine is a drift diagnostic, not a proof of behavioral
fidelity. Native training with live learned factors is separately implemented
in the BPD RP03 follow-on. RMP selects blocks.0.mlp.2 and the middle visit by
default; RAM02 selects cell.2 at the middle pass. Only CPU tied arms supported.

## Existing gym rollout
| Gym | Status | Useful memory scope |
|---|---|---|
| RMP core4, masked pointer chase, modular recurrence | integrated via run_rmp_training_cell.py | full sequence gradient at a chosen visit; largest scaling candidate |
| RAM02 compact Lean candidate ranker | integrated via train_local.py | exact training candidate IDs; tiny dataset, engineering check only |
| Lean-CGRT Mathlib TRM/HRM | deferred | carries and distinct supervision clocks need explicit capture adapter |
| SAE dictionary training | deferred | detached cached features already have different objectives |
| LBF exact solver, LS Collatz enumeration | unsuitable for gradient hook | verifier/cache resource experiments separately |
| DataGoblin MOD7-FB/state | deferred | operator/state feedback; finite support cannot justify multi-GiB bank |
| BPD RP02/RP03 | native follow-on, preserve frozen results | live V/U maps, fresh development roots |

## Run
Add to the existing RMP CPU CLI command:
```
--mms-mib 512 --mms-mode bpd --mms-rank 16 --mms-topk 4 --mms-ratio 0.2 --mms-refresh 128 --mms-max-age 32768
```
Sweep 0, 128, 512, 2048, 4096 MiB sequentially, only after measuring RSS,
commit, available RAM and CPU cost at the smallest budget. Use matching
model initialization and sample stream. For RAM02 pass the same flags to
train_local.py; use a new output directory. Device calibration remains
baseline-only, while enabled fitting uses CPU. No automatic host cap increase.
The bank limit is tensor payload + tags, not total RSS; pages can be sparse.
Checkpointing the cache adds disk I/O and may increase transient memory;
RMP latest checkpoints retain cache/maps exactly, so keep enough disk/headroom.
The current RMP max-seconds pause includes in-loop work, but final checkpoint
and evaluation may exceed the soft budget: use external process CPU/wall
receipts and count all preparation, serialization and evaluation.

## Evaluation
Run independently at fixed threads, initial checkpoint, training exposure
and process CPU-second budget, separately report time-to-competence. Record
actual bytes, occupied rows, hits, collisions, admitted updates, refresh cost,
peak process-tree RSS/commit, step time and held-out accuracy. Do not infer
capacity gain when banks have the same occupancy or zero hits. Include plain
task, observe, dense, BPD proxy, native BPD export and matched-dose sham;
the coordinate roll is a weak directional control, not a calibrated causal
null. Preserve all failures, use independent source roots and training seeds,
fresh held-out lineages, and EvalMath resolution preflight before claiming a
causal mechanism. No protected cohorts are opened. Start with a learnable
pointer chase task; modular-recurrence depth-one coverage defect stays a
separate problem.

## Checkpoint and CPU accounting amendment
The checkpoint helper now hashes files in 4 MiB chunks (same SHA-256 bytes).
Enabled RMP cells save their latest full cache every --mms-checkpoint-every
updates (default 2000) and on exit; named model-only snapshots still retain
the original schedule. This changes crash recovery frequency only in MMS
cells. A 4 GiB checkpoint at the existing 40 MiB/s cap can take about 102 s
before hashing/fsync. Keep at least space for both old and temporary files.
--max-cpu-seconds limits each run_cell invocation at an update boundary,
including initialization inside run_cell but excluding CLI pool loading and
mandatory exit evaluation/checkpoint. Invocation timing receipts include exit
checkpoint/evaluation, so report actual budget overshoot; use an external
whole-process timer for the complete CLI. Historical runs' accumulated CPU
ledger starts when this code first resumes them.

Map admission amendment: the absolute 0.5 weight cosine gate applies only to
a frozen exported map. Rank-truncated SVD explained energy is rank-dependent;
for example rank-16/128 random initialization can have cosine below 0.5.
SVD and native live maps require a nonzero reconstruction cosine, and all
feedback still requires positive current/historical gradient alignment.
Neither gate establishes causal fidelity.

## MMS-02 multi-site
RMP now accepts --mms-sites mlp,attention,qkv,mlp_in and
--mms-schedule round_robin|adaptive. A shared global LRU page pool caps BF16
tensor payload + tags across every site, evicting old pages before allocating
new ones. Each optimizer step selects one site and one visit; all microbatches
use that choice. It adds no forward/backward pass. Adaptive uses historical
admitted rows per sample (observe: hit fraction), plus deterministic exploration
every fourth update. This is a coverage/alignment heuristic, not validated
learning utility. Use fixed round-robin for matched mechanistic controls.
Saved RMP latest checkpoints include pool eviction order and schedule state;
old default single-site IDs and checkpoints are unchanged. Multi-site currently
uses current SVD proxies; the single default site supports learned BPD exports.
