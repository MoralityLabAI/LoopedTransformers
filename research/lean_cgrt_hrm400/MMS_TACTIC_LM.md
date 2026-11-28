# Mathlib MMS-04: native recurrent byte-tactic training

Development implementation, not a new Mathlib competence result. The existing
lcgrt.tactic_lm trainer accepts bounded CPU memory feedback for loop, HRM and
TRM encoders. The original zero-budget identity and resume remain unchanged.

Physical sites: fast.ff, fast.att; HRM additionally fast.cross, slow.ff,
slow.att, slow.cross; TRM trm0.ff, trm0.att, etc. Each update selects a single
physical site and one differentiable visit; the page pool bounds the combined
BF16 gradient payload + tags. In HRM the slow clock executes ceil(T/slow_every)
visits. In TRM only the final differentiated recursion (n+1 network calls)
is hooked. Earlier no-grad recursions and the ACT detach remain intact.
TRM identity uses the *actual carried train row*, not the discarded draw, plus
supervision stage. Loop/HRM keys also include sampled depth. Fresh admissions
and optimizer updates remain different ledgers.

With TRM MMS, checkpoints save y/z, data, halt mask, stages and original row
IDs, together with EMA and the MMS cache. Matched zero-budget resumes should
use --exact-carry-resume. Activation-checkpointed and GPU MMS are rejected in
this version. Historical continuation forks are rejected for MMS; create
fresh paired runs. The pack's SHA-256 and map manifest hash are part of the
enabled identity. Hash-pinned exported maps are accepted via
--mms-map-manifest; unspecified sites use the SVD seed proxy.

Run from the existing lcgrt working directory. Example small CPU pilot:

```sh
python -m lcgrt.tactic_lm --data mathlib-c3 --preset tlm-trm-XS --cfg trm_sup=4 --seed 17 --steps 1000 --batch 16 --threads 4 --device cpu --eval-every 250 --log-every 250 --val-items 256 --val-batch 8 --save-every 1000 --mms-mib 512 --mms-mode bpd --mms-sites trm0.ff,trm1.ff --mms-schedule round_robin --out RUN
```

Compare 0/128/512/2048/4096 MiB, observe/dense/sham/filtered feedback; use
the same fixed sites and initialization for matched controls. Use a fixed
depth for the first loop/HRM sweep; variable-depth histories are split by
depth. For a width-176, 384-position output, a BF16 record plus tags is
135,184 bytes; 4 GiB therefore holds at most 31,771 records across such
sites before indexing/allocator overhead, excluding the model. Repeated
row-and-stage coverage is a necessary admission criterion. If the larger
bank has no more hits than the smaller one, no scaling contrast exists.
Use small RAM targets initially and measure actual occupancy, peak host RSS,
commit, CPU core-seconds and checkpoint cost. Native host guard remains;
a page budget is not an RSS limit. Keep free disk for latest, lkg and temp.
The --max-cpu-seconds budget is a soft per-invocation training limit; dataset
loading is outside that interval, exit checkpointing may overshoot. Compare
fixed exposure and fixed CPU separately. Existing --stop-at enables
reproducible matched budget continuation without altering the recipe.

Use the existing native verifier for development only, e.g.:

```sh
python -m lcgrt.mathlib_eval --data mathlib-c3 --ckpt RUN_A RUN_B RUN_C --splits val --per-split 120 --per-file 4 --beam 8 --workers 2 --threads 2 --proposal-jobs 1 --out DEV
```

Ensure each run has a best-on-val checkpoint (EMA for TRM); a very early CPU
window may have no best checkpoint yet. Use fresh theorem-lineage development
cohorts for scientific conclusions; the historical 588-case c4 archive is
not eligible for tuning. Record candidate-pool coverage and kernel-certified
close@k, plus fresh-admission and compute budgets. Apply EvalMath preflight
to independent training roots and theorem lineages before opening a sealed
confirmation. The engineering tests use tiny synthetic token data, not Lean
proofs. No new certified proof or causal BPD learning benefit is claimed.

## Repetition admission before a full-corpus sweep

Uniform draws from the full ~193k-row corpus can make exact row/stage hits
too rare to identify a memory scaling effect. The optional
--train-pool-rows 2048 --train-pool-seed 314159 selects a fixed sorted subset
of the existing TRAIN split, without consulting labels. It works in both
zero-budget and MMS arms. Identity binds original packed train SHA, pool
seed, row count and index SHA. Default full-corpus sampling is unchanged.
Use this as an exposed development stress test only, not a claim of
full-Mathlib generalization. At TRM sup=4, the subset has at most 8192
row-stage pairs per selected recurrent visit/site. A 2 GiB bank can hold
more of them than a 128 MiB bank if actual occupancy/hit counts differ.
Pilot with the same site and batch across arms before multi-site and full
corpus; record zero-hit/saturation failures and retain all arms.
