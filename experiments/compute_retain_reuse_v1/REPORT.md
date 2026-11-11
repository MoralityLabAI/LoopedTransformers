# Compute–Retain–Reuse benchmark: results and verdict on the adaptive controller

Run: 4 state-machine tasks × 3 training regimes × 6 seeds = 72 independently trained looped transformers
(~0.2M parameters, one weight-tied block, trained with 8 loops, evaluated for up to 32).
Everything below is produced by `python -m crrbench.analyze --run results/full`; full tables are in
`results/full/analysis/tables.md`, figures in `results/full/analysis/`, raw per-model results in `results/full/cells/`,
logged predictions and hidden states at loop depths 1,2,4,8,16,32 in `results/full/logs/`.

## Bottom line

**The controller does not give a statistically supported gain over static recurrence once the static baseline is allowed a sensible depth, and it does not survive the length shift.**

| comparison (accuracy at equal inference FLOPs, controller − static) | pooled over 12 task×regime clusters | 95% cluster-bootstrap CI |
|---|---|---|
| trained lengths, pre-specified budgets κ∈{2,3,4,6,8,12} | **+0.049** | [+0.005, +0.107] |
| **held-out longer lengths**, pre-specified budgets (primary) | **−0.024** | [−0.080, +0.038] |
| held-out lengths vs the best fixed depth within budget B=8 | −0.094 | [−0.141, −0.051] |
| trained lengths vs the best fixed depth within budget B=8 | −0.014 | [−0.022, −0.007] |

* The only place the controller clearly helps is where static recurrence **loses its answer**: models trained with the **final-loop** loss
  (cyclic +0.29, dfa +0.08, storyworld +0.10, sweepweave +0.17 on trained lengths; all Holm-significant, 6/6 seeds each). Fixed-depth accuracy for
  those models peaks near loop 8 and collapses afterwards, and the controller's PRESERVE latch plus HALT keeps the answer. But a practitioner who picks the right
  fixed depth gets the same accuracy: against the best fixed depth the controller is level or slightly *worse* (−0.004 to −0.019 for those cells).
* For models trained with **random-intermediate-loop or randomized-exit supervision** (which already retain answers) the controller is flat to
  negative on trained lengths (≈0 to +0.011 for cyclic/dfa, −0.0004 to −0.033 for storyworld/sweepweave).
* **Under length shift it fails**: the controller is trained on the training lengths only; on unseen lengths its P(correct) is overconfident, so it
  stops too early (premature-stop rate 0.56–0.72 on Storyworld random-supervision models, accuracy −0.21 vs fixed depth at equal FLOPs). Positive held-out gains
  again appear only for the final-loop models (cyclic +0.19, dfa +0.06, sweepweave +0.10).
* The controller is **better calibrated than raw softmax** confidence (ECE softmax − ECE Q_halt = +0.060 [+0.036, +0.084] on trained lengths, +0.052 [+0.023, +0.081] held-out; 72 models), and it beats a
  non-learned confidence-threshold early-exit baseline on trained lengths (+0.082 [+0.032, +0.144]) but not on held-out lengths (−0.004 [−0.054, +0.051]).

Inference caveat: with 6 seeds per cell the exact sign-flip p-value cannot go below 0.031, so per-cell claims use the paired-t p with Holm correction
(sign-flip p are in the tables) and the pooled estimates use a cluster bootstrap over the 12 task×regime cells. Seeds are the only unit of replication; confidence
intervals do not cover choice of task family, model size or loop budget.

## 1. Compute and retain

(Table 1 in `tables.md`; figures `fig1_accuracy_vs_loop.png`, `fig2_retention_curves.png`, `fig2b_retention_from_depth.png`.)

* **Compute.** All regimes solve the trained lengths (solved-ever ≥0.97, median first-correct loop 1 for random-loop regimes, 2–6 for final-loop). On held-out longer
  lengths solved-ever drops for the random-loop regimes (cyclic 0.80–0.82, Storyworld 0.40–0.62, SweepWeave 0.74–0.81; final-loop models stay at 0.86–0.99 because they sweep through many loops) and first-correct moves later.
* **Retain.** Answers are lost on loops beyond the trained 8 for every regime, but at very different speeds. Stable-solve rate (correct at every loop from k* to 32) on trained
  lengths is ≈0.01 (cyclic), 0.03 (sweepweave), 0.24 (storyworld), 0.38 (dfa) for **final-loop** training versus 0.53–0.86 for random-loop regimes on cyclic/dfa/storyworld
  and 0.23–0.27 on sweepweave. One-step forgetting hazard: 0.27 vs 0.013–0.019 (cyclic). Final-loop supervision gives *peaked* accuracy-vs-depth curves
  (chance at loop 1–2, best near loop 8–12, back to chance for cyclic by loop 20); the two random-loop regimes give flat plateaus with slow decay.
  Random-intermediate retains a little better than randomized-exit on most tasks.
* So "compute then lose it" is a regime-dependent property that final-loop training makes extreme and loop-randomized training largely fixes inside the trained depth range.

## 2. Reuse and delayed freezing (0, 1, 2, 4, 8 extra loops after the first correct prediction)

(Table 2; figure `fig3_reuse_vs_delay.png`.) Samples are those solved by loop ≤24; the same samples are used at every delay, so the delay contrast is paired.

* **Probe reuse** (independent decoder trained on frozen states; continuation over 4 new symbols with the true transition rule). On trained lengths it is ≈0.98–1.0 and flat.
  On held-out lengths extra loops make the state **more decodable for Storyworld** (random-intermediate +0.185 [+0.143, +0.228], Holm p=0.002; randomized-exit +0.151, Holm p=0.045; final-loop +0.037 n.s.),
  **less decodable for cyclic** (−0.14/−0.03/−0.04, p≈0.003–0.006, Holm p 0.06–0.11), and flat for dfa and sweepweave (|Δ|≤0.03, n.s.).
  Pooled over the 12 cells: +0.014 [−0.033, +0.060], i.e. **no general benefit of extra iterations for reusability**.
* **Native reuse** (frozen state pasted into a longer sequence, same core, 8 loops, zero-shot). It is poor at every delay (chance for cyclic; 0.27–0.83 elsewhere), and **gets worse with delay**: pooled
  −0.055 [−0.101, −0.021]; strongest for sweepweave final-loop (−0.26, Holm p=0.017).
  Zero-shot native reuse fails because the core never saw "old state + fresh tokens" configurations; the benchmark reports it as the honest measure of reusing a saved hidden state without retraining.
* Caveat on the Storyworld probe gain: samples are selected on being correct at their first-correct loop, which includes lucky guesses whose state is not yet well formed; extra loops can then
  appear to "improve" the state through selection (regression to the mean). The effect is sizable only where held-out accuracy is low, so treat it as suggestive.

**Answer to "does the representation become more reusable after additional iterations?":** not in general. It does for Storyworld on unseen lengths (probe), it hurts for cyclic (probe) and for native resumption everywhere, and is neutral for dfa/sweepweave.

## 3. Controller, calibration, stopping errors and compute (Section 3–4 of `tables.md`)

* Policy: Q-network over {CONTINUE, PRESERVE, HALT}; PRESERVE is used for 7–73% of visited steps (it is a free hedge once the live answer looks right) and HALT for 6–86%, depending on task and price.
* **Calibration.** ECE of Q_halt 0.003–0.07 on trained lengths (softmax: up to 0.33 for final-loop models); on held-out lengths it degrades sharply on Storyworld random-loop models (0.71–0.85, i.e. confidently wrong) while staying 0.03–0.32 elsewhere.
* **Premature stopping** (halting before the first correct loop; held-out, λ=0.004): 0.01–0.16 for cyclic, dfa and final-loop models, 0.18–0.27 for sweepweave random-loop, **0.56–0.58 for Storyworld random-loop**.
  On trained lengths it is ≤0.04 everywhere.
* **Excess computation** (loops beyond the first correct loop when right): 4–15 loops at λ=0.004, 0.15–5 at λ=0.016; futile loops spent on never-solved items are large on held-out lengths (≈11–17 loops at λ=0.004 for cyclic/dfa).
  Compute prices up to 0.5 were needed to reach cheap budgets; the first sweep (λ≤0.12) could not go below ≈4 loops, which is why the first analysis showed empty cells (archived under `results/full_v1/`).
* **Accuracy under fixed FLOP budgets** (`fig4_accuracy_vs_flops_*.png`): the controller frontier hugs, rarely exceeds, the best static curve, except right of the static peak for final-loop models.

## What was and was not done

* **Methods fixed before the results:** loss regimes, held-out length sets, FLOP accounting, matched-baseline construction, budgets κ∈{2,3,4,6,8,12}, the unit of inference (seed), and the controller design.
* **Changed after seeing results (flagged in the tables as post hoc):** the compute-price range (first run capped at λ=0.12), the "range-adaptive" and "best-within-budget" endpoints, and the headline Holm correction using paired-t p-values (the exact sign-flip p is capped at 0.031 with six seeds).
  The first-run analysis is archived in `results/full_v1/` and gave the same qualitative conclusion.
* **Storyworld data.** Two sources, both reduced to a deterministic state graph. `storyworld`: the 12 train-split `storyworld_branching_world_v1` worlds from `ConstitutionalAlignment/experiments/storyworld_curriculum_v1` (copied to `data/storyworld_worlds`),
  with a random start encounter shown as a token (without it each world is a fixed clock, and the answer depends only on the string length). `sweepweave`: the four SweepWeave-format worlds found in the GPTStoryworld repository
  (`data/sweepweave_worlds`), where the stochastic reaction draw is an input symbol. Scripts, effects and desirabilities are not evaluated, and only 12 and 4 worlds exist, so these are state-tracking tasks over real narrative graphs, not Storyworld play.
* **Not found:** an existing "recurrent-recovery experiment" in the repositories on this machine; the suite is standalone.
* **HRM-style:** the controller is an HRM-ACT-style Q-learning halting head over a frozen looped core; the core itself is a single-level looped transformer, not a two-timescale HRM.
* **Limits.** Small models, short strings (N≤12), 8 training loops, 1,500 optimizer steps; length extrapolation is partial for every model (that is the point of the held-out sets but also limits what "solved" means); depth beyond 8 loops is an extrapolation of the trained recurrence.
  The best-within-budget comparison picks the maximum over candidates on the test data for both sides, which is symmetric but optimistic for both. The controller was not tuned per task.
* **NPU.** Not used: the machine's NPU (AMD Ryzen AI 9 365) is reachable only through AMD's Ryzen AI Software, which is not installed, and training/Q-learning cannot run there. Per your choice the suite stays on GPU/CPU.
* **Execution notes.** Training ran on a GPU shared with other jobs; a driver hang (`nvidia-smi` lost the device) interrupted one evaluation pass, and the last 32 evaluations ran on CPU from the cached models. Results do not depend on the device used for evaluation, apart from floating-point noise.
