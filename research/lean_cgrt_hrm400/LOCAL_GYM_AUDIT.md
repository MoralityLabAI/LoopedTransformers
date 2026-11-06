# Local gym audit (v1 gym at e134b54)

Date: 2026-11-05. Machine: MSI Stealth A18 AI+ (A3HWIG), AMD Ryzen AI 9 365, 31.1 GB RAM, **NVIDIA RTX 5080 Laptop GPU, 16 GB** (`nvidia-smi`; the v1 README's 4 GB RTX 3050 is not in this machine). Lean 4.19.0 (commit 6caaee842e94, the version pinned by `experiments/lean_reasoner/lean-toolchain`), installed from the official release to `D:\lean_cgrt\toolchains`.

Every number here comes from a committed receipt:
- [`results/v1_audit/probes.json`](results/v1_audit/probes.json), produced by [`audit/v1_probes.py`](audit/v1_probes.py) (CPU, torch 2.11.0);
- [`results/v1_audit/lean_probe.json`](results/v1_audit/lean_probe.json), produced by [`audit/v1_lean_probe.py`](audit/v1_lean_probe.py) (Lean 4.19.0).

## What v1 implemented and measured

`gym.py` trains `TwoClockHRM` on generated Horn problems and exports teacher proofs to Lean. No v1 training result was recorded in the repository. LR-01 (`experiments/lean_reasoner`) is separate pilot evidence. It used a 20,161-parameter policy (the README says 20,161; `PLAN_SILICO_400M.md` says 5,161) and reached 100% standard / 92.01% depth-OOD completion with four passes over three seeds. Its corpus passed Lean 4.19.0 in CI (`results/kernel_receipt.json`). The model's rollouts were replayed by a Python fragment checker and by the CI workflow. LR-01 is kept frozen as a regression experiment and is not part of this campaign's results.

The repository's other recurrent work: the TRM/LDT hybrid neighbourhood v1 stopped negative at the screen stage. v1.1 was mixed: `trained_trm_ldt_fixed` macro utility 0.6873 was the sole Pareto point, but its "TRM" components are gym analogues, not trained recursive models (`reports/rlm_trm_ldt_hybrid_neighborhood_v1_1.md`, `reports/benchmark_comparison.md`). The loop-algebra reports record tied-recurrence gradient norms of 149–166 at high loop counts versus at most 1.565 untied (`reports/looped_transformer_algebra_v0.md`). That record is the reason v2 post-normalises the recurrent state and supervises every slow tick.

## Findings

| # | Claim checked | Measured | Status |
|---|---|---|---|
| 1 | Preset parameter counts | smoke 100,995; 5m 4,835,715; 15m 15,393,603; 30m 30,152,643 | counts correct; **the halt head (4 tensors) never receives a gradient**, because `forward` never calls it |
| 2 | Genuine fast/slow separation | slow state `z` is zeroed on every call; both blocks act on the same rule tokens with the same mask; 4 fast + 2 slow block applications per decision | **two update rates inside one decision; nothing persists across proof steps** |
| 3 | Teacher leakage into inputs | features are observable state only (no teacher, depth or label inputs) | no leak found. But every atom has at most one producer, so the proof is unique and search-free: backward reachability solves 60/60 depth-OOD problems |
| 4 | Proof-length-plus-one budget | `budget = len(teacher) + 1`, same for all modes; CGRT mode spends 3× the model calls under the same action budget | budget applied consistently; **CGRT is not compute-matched** |
| 5 | Split by theorem / family | 3 of 24 validation problems are structural (WL-hash) duplicates of problems in the first 2,000 training steps; all three families appear in training and evaluation | **no family split, no dedup** |
| 6 | Resume vs uninterrupted | 20 steps vs 10 + 10: max weight difference 0.0 (bitwise equal); resuming with a changed lr **and** batch is accepted silently; no last-known-good copy; no hash check on load | resume exact; **identity check incomplete** |
| 7 | CGRT explores alternatives | branches permute rule order, a symmetry of the equivariant network: argmax identical in 24/24, max logit deviation 3.0e-7; greedy and CGRT solve the same 9/24 with 131 vs 393 calls | **CGRT is a no-op at 3× cost** |
| 8 | Curriculum | depth unlocks by micro-batch index: at accumulate 8, depth 5 unlocks at optimizer step 188 instead of 1,500 | **bug** |
| 9 | Lean export and certification | v1 corpus (64 theorems) passes under Lean 4.19.0. **v1 `audit` certifies `theorem thm_bad : False` proved from a user `axiom`** when a line break follows the keyword (the literal-string check looks for `"axiom "`) and an `#eval` prints a fake "does not depend on any axioms" line, which the auditor counts. `set_option` (e.g. `debug.skipKernelTC`) and `#eval` were allowed. Statements are not bound to a dataset, so a proof of a weaker statement under the same name passes. A tab after `axiom` does *not* work: Lean 4.19 rejects tabs | **certification hole (confirmed)** |
| 10 | README install path | the README calls `lean`; elan had Lean 4.35.0 but no default toolchain, so `lean` failed | fixed by pinning 4.19.0 (`lean/lean-toolchain`, `lcgrt/config.py`) |

## Fixes

In `gym.py` (v1 stays runnable as the pilot):
- curriculum clock = optimizer step (#8);
- resume identity includes lr, batch, accumulate, depth interval and steps (#6);
- the CGRT eval output labels the measured no-op (#7);
- `audit` uses a token regex (`axiom|sorry|admit|set_option|#eval|macro|…`) and parses `#print axioms` per declaration name (#9). After the fix the adversarial file is rejected (`results/v1_audit/lean_probe.json`, `fixed/adversarial`).

Findings 1–5 and 7 are design limits. They are addressed by the v2 package `lcgrt/` rather than patched into v1:

| v1 limit | v2 (`lcgrt/`) |
|---|---|
| unique proofs, no search | generator with several producers per goal: dead-end, deep-dead and cycle decoys, plus valid-but-longer alternatives ([`lcgrt/fragment.py`](lcgrt/fragment.py)) |
| Python decides legality | Lean decides every goal, tactic outcome and error; Python only orchestrates ([`lean/Main.lean`](lean/Main.lean), [`lcgrt/extract.py`](lcgrt/extract.py)) |
| no split hygiene | canonical-hash dedup before splitting; held-out families, depths 7–12, and unseen 3-premise dependencies ([`lcgrt/dataset.py`](lcgrt/dataset.py)) |
| CGRT no-op | contexts the network is not invariant to (recurrence depth, failure-history visibility), a learned branch scorer, a shuffled-score control, Lean-call-matched budgets ([`lcgrt/search.py`](lcgrt/search.py)) |
| teacher-only Lean export | every proof the model finds is compiled as a standalone file; statements bound by construction; tactic whitelist ([`lcgrt/certify.py`](lcgrt/certify.py)) |
| halt head unused | halting head trained per slow tick and used for adaptive depth ([`lcgrt/model.py`](lcgrt/model.py)) |
| resume / LKG | sha256 sidecars, latest→lkg rotation, full identity hash, deterministic data order ([`lcgrt/train.py`](lcgrt/train.py)) |

## Defects found while building v2 (fixed before any reported result)

- `importModules` in Lean 4.19 needs `loadExts := true`, or notation such as `∧` does not parse.
- Elaboration errors inside tactics are logged and replaced by `sorry`: `apply nonexistent` "closed" a goal. The REPL now runs without error-to-sorry and treats any logged error as failure; the kernel op reports `sorryAx`.
- `Meta.SavedState.restore` does not restore the name generator. New goal mvars reused assigned ids, and `apply` reported open goals as solved. Each request now gets its own name prefix.
- The heartbeat counter is process-global; `initHeartbeats` is now set per request.
- Rendering every rejected candidate's error message pretty-printed the whole context (540 ms per expand at 85 hypotheses). Messages are now rendered only for sampled failures (7.4 ms per expand).
- Typed-attention bias via index gather had an atomic-scatter backward (424 of 520 ms per step); replaced by a one-hot matmul, shared across the tied fast steps.
- Resume on CUDA failed (`RNG state must be a torch.ByteTensor`), because `map_location` moved the RNG state to the GPU. Found by the first real run; CPU tests had missed it.

## Defects found during campaign c1 (fixed; affected runs re-run and kept as evidence)

- **HRM slots got untrained global features at evaluation.** Training passes zeros, so the glob projection keeps its random initialisation, but the search passed real features. This added a random vector to every HRM slot; flat models never read the features. It cost HRM-S about 2 points on `test_mirror` (47.8 vs 45.8 greedy at b1). The five affected evaluations are kept as `results/c1/evals/*_globbug.json.gz`; the reruns are the reported numbers.
- `hash()` is salted per process, so the random baselines and shuffled-CGRT permutations would not have replayed. Seeds are now CRC32.
- vramorama's default restart budget (8) ran out on routine thermal pauses (HRM-L at step 2,650/3,000; two c2 runs). The run resumed from the same verified checkpoint, and the budget is now 60 per job.
- Two CGRT evaluation jobs pointed at a renamed run's branch-weights path and failed on start. They are re-queued.

## Evaluation protocol changes (recorded before any v2 result)

- Success = Lean-certified completion (standalone file, axiom audit), not graph closure.
- Budget = node expansions (Lean calls), identical for all policies. Reported at `b1` = shortest + 1 (the v1 rule), `b2` = 2× shortest, `b4` = 4× shortest + 4. Model forward passes and analytic FLOPs are reported separately, so CGRT's extra forwards are visible.
- Evaluation uses the final checkpoint. `val` is monitored but never used for selection; test splits are only touched by evaluation jobs.
