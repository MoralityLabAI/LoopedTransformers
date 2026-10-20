# Compute–Retain–Reuse benchmark for recurrent (looped) models, with an adaptive HRM-style controller

Standalone suite (PyTorch + NumPy/SciPy/Matplotlib). No prior "recurrent-recovery" code was found in the
repositories on this machine, so the suite is self-contained; its pieces (`tasks`, `model`, `train`, `collect`)
are small enough to swap for an existing recurrent model.

```
crrbench/
  tasks.py       state machines: cyclic (Z5 sum), dfa (rotate + two partial-reset maps), storyworld + sweepweave (real worlds), s3/s4 (optional)
  model.py       weight-tied transformer block looped K times; answer read from the last token after every loop
  train.py       the three regimes: final | random_intermediate | random_exit
  collect.py     per-loop logits and hidden states for any split
  metrics.py     COMPUTE (first-correct loop) and RETAIN (retention curves, stable-solve, forgetting hazard)
  reuse.py       REUSE: delayed state freezing; native zero-shot resume + independent decoder probe
  controller.py  Q-learning controller over {CONTINUE, PRESERVE, HALT} (HRM-style halting head)
  evalctl.py     frontiers, FLOP-matched fixed-depth baseline, calibration, premature / excess computation
  flops.py       analytic inference FLOPs
  stats.py       paired tests over seeds, Holm, cluster bootstrap
  run_cell.py    one (task, regime, seed): train -> compute/retain -> reuse -> controller
  run_suite.py   whole grid with persistent worker processes (restarts workers on CUDA errors, caches models)
  analyze.py     tables.md, summary.json, figures
tests/           fast CPU sanity tests
data/storyworld_worlds/   snapshot of 12 train-split branching worlds (ConstitutionalAlignment, storyworld_branching_world_v1)
data/sweepweave_worlds/   4 SweepWeave-format worlds from the GPTStoryworld repo
REPORT.md                 results and verdict (read this first)
```

## Run

```bash
PY=<python with torch, numpy, scipy, matplotlib>
$PY tests/test_crrbench.py                                        # sanity
$PY -m crrbench.run_cell --task dfa --regime random_exit --seed 0 --out results/quick --config quick   # smoke test, ~2 min
$PY -m crrbench.run_suite --out results/full --seeds 6 --workers 4 --train-only   # train + cache models
$PY -m crrbench.run_suite --out results/full --seeds 6 --workers 4                # evaluate (reuses cached models; add --device cpu --threads 4 on a busy GPU)
$PY -m crrbench.analyze --run results/full                                        # -> results/full/analysis/
```

## Definitions

Models read a string, loop `K_eval = 32` times (trained with `K_train = 8` loops), and emit a state prediction after
every loop. Loop depths 1,2,4,8,16,32 are logged (predictions + last-token hidden states, `results/*/logs/*.npz`);
predictions at *every* loop are used for the metrics.

* **Compute**: first loop `k*` at which the prediction is correct (and whether it is ever correct).
* **Retain**: `P(correct at k*+Δ | first correct at k*)`; `P(correct at k+Δ | correct at k)` for k ∈ {1,2,4,8,16};
  stable-solve (correct at every loop ≥ k*), lost-after-solve, one-step forgetting hazard.
* **Reuse** (delayed state freezing): for each sample the state is frozen at loop `k* + d`, d ∈ {0,1,2,4,8}
  (samples solved by loop ≤ 24, identical sample set for every d, so d is a paired contrast).
  * *probe reuse*: an independent MLP decoder (trained on fresh in-distribution frozen states, not the model's head)
    recovers the machine state; 4 new symbols are then applied with the machine's own transition rule.
    Accuracy = the continuation is right. It measures whether the information needed to continue is still present.
  * *native reuse*: the frozen state is pasted into the first positions of a longer sequence, fresh embeddings for
    4 new symbols are appended, and the same frozen core runs M = 8 more loops (zero-shot). A from-scratch run with
    the same total loops is reported alongside.
* **Training regimes** (same unroll K_train, optimiser, steps, data):
  `final` = CE at the last loop only; `random_intermediate` = each sample supervises one uniformly drawn loop;
  `random_exit` = each sample draws an exit depth e (log-uniform on 1..K) and is supervised at every loop 1..e.
* **Held-out lengths**: train on N ∈ {2,3,4,6,8}; evaluate fresh strings at N ∈ {4,8} (`id`), unseen interior N ∈ {5,7}
  (`interior`), and longer N ∈ {10,12} (`extrap`); `heldout` = interior ∪ extrap.
* **Tasks** (`cyclic dfa storyworld sweepweave`, 6 seeds each): `cyclic` (sum mod 5), `dfa` (rotation + two random non-injective maps, 5 states), `storyworld`
  (12 Storyworld branching worlds from `ConstitutionalAlignment/experiments/storyworld_curriculum_v1/worlds/train`;
  token layout `[world, start-encounter, a_1..a_N]`, action `a` takes option `a mod #options`, terminal encounters
  restart the world; stochastic reaction selection of SweepWeave-style worlds is not modelled).

### Controller

After each loop the controller (a 2-hidden-layer Q-network, conditioned on the compute price λ) sees the loop's
last-token vector, softmax confidence/margin/entropy, argmax stability, the loop index and its own output-register
state, and picks:

* `CONTINUE` – keep computing; the register tracks the live prediction;
* `PRESERVE` – latch the current prediction in the register (or keep an existing latch) and keep computing;
* `HALT` – stop and return the register.

Reward = 1[returned answer correct] − λ·(loops used). Q-values are fitted by Q-learning with a target network on exact
simulations of the frozen base model's trajectories (the base model does not depend on the controller's actions);
`Q_halt` is additionally regressed on 1[answer correct if halted now] at every visited state, so it is a calibrated
probability estimate by construction. The controller is trained on fresh strings of the *training* lengths and
evaluated on the held-out lengths. One network serves all prices λ; sweeping λ traces an accuracy–FLOPs frontier.

**FLOPs.** `loop_flops(T,d) = 24·T·d² + 4·T²·d` per loop plus the read-out; the controller's own MLP is charged per loop.
Fixed-depth baselines pay no controller overhead. FLOPs are reported as fixed-depth loop-equivalents (κ).

**Matched baselines.** For a controller operating point with mean cost κ the fixed-depth baseline is the random mixture of
⌊κ⌋ and ⌈κ⌉ loops with the same mean FLOPs (expected accuracy = linear interpolation of the fixed-depth curve).
A non-learned confidence-threshold early-exit baseline and an oracle halting bound are reported for context.

**Statistics.** The unit of inference is an independent training seed. Per (task, regime) the endpoint is the mean, over
the pre-specified budgets κ ∈ {2,3,4,6,8,12}, of (controller accuracy − fixed-depth accuracy at equal FLOPs).
Tests: paired t, exact sign-flip permutation, Wilcoxon, Holm across task×regime; pooled estimate via a two-level cluster
bootstrap (task×regime clusters, seeds within).

`sweepweave`: the four SweepWeave-format worlds in the GPTStoryworld repo (`data/sweepweave_worlds`); state = encounter, symbol = (option, reaction) pair,
the stochastic reaction draw being treated as an observed input; dead ends restart the world. Start encounter is a token, as for `storyworld`.

## Notes
* Windows/MSIX: when Python reports an `AppData\Local\Packages\...\LocalCache\Roaming` path, files may be unreadable through it;
  `tasks.real_path` / `_readable_dir` map back to `AppData\Roaming`.
* Training and evaluation are separable: models are cached in `<out>/models`; delete `<out>/cells` to re-evaluate (new controller settings, probes) without retraining.
* NPU: not used (AMD XDNA needs the Ryzen AI Software stack, absent here; training/Q-learning cannot run on it).
