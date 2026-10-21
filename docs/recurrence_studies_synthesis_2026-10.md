# Recurrence in transformers: what the October 2026 studies jointly establish

Three studies run in October 2026 on branch `add-recurrent-recovery-laco`: `experiments/compute_retain_reuse_v1` (CRR), `experiments/recurrent_sae_study` (SAE) and `experiments/recurrent_recovery_laco_v1` (LaCo). Numbers are quoted from the studies' own tables, with the source file after each.

## The shared question

The Recurrence Monitorability Program (`docs/recurrence_monitorability_program_v0.md`) asked whether moving serial computation into tied latent iterations makes a model harder to monitor, and whether the per-iteration trajectory is a stable, causally faithful object or only looks that way through a sparse autoencoder. Its Phase 2 plan (`docs/rmp_phase2_plan.md`) turned that into a closed-loop dictionary-replacement test on sub-2M-parameter models. The October studies step back to the capability premise underneath: does looping a block add computation that a fixed depth lacks, and if so, can it be retained across loops, reused from a saved state, controlled by a learned halting policy, and read out at the feature level? CRR tests control and retention, SAE tests readability and causality of the loop state, and LaCo asks whether a looped block can stand in for deleted layers in a real language model.

## The three studies

| Study | Model scale | Task family | Seeds | Pre-registered endpoint | Headline effect [95% interval] |
|---|---|---|---|---|---|
| CRR | ~0.2M params, one weight-tied block, trained at 8 loops, run to 32 | state tracking over cyclic (sum mod 5), 5-state DFA, Storyworld and SweepWeave graphs; N<=12, held-out N=10,12 | 6 per task x regime, 72 models | controller minus fixed-depth accuracy at equal FLOPs, mean over budgets {2,3,4,6,8,12}; held-out lengths primary | -0.024 [-0.080, +0.038] held-out; +0.049 [+0.005, +0.107] trained lengths (`REPORT.md`; `results/full/analysis/tables.md` section 3) |
| SAE | d=128 looped models from the CRR grid plus one own; 12-layer d=128 mini-LaCo replica | same automata; Qwen3 path written, never run | 2 per task x regime (own 1); 2 replicas; 3 controller runs; intervals over examples, not seeds | none frozen; five research questions scored by AUROC gain over a confidence baseline and intervention accuracy change | sparse-code probe AUROC gain for "answer will be lost" +0.125, above 0 in 11/11 runs in-distribution; ablating the 8 best loss-predicting features rescues 0-5% of lost answers vs 0-14% for random edits (`reports/REPORT.md` section 0) |
| LaCo | Qwen3-0.6B-Base, 28 layers; LaCo collapses layers 8-15 into one, 21 left | text KL to teacher; ARC-E/C, PIQA; synthetic FSM, addition, multiplication, variable chains, in-distribution and longer | 1 seed; 64-example task cells, 400 multiple-choice items; differences under 3-5 points called noise | none; looped block D vs params-matched LoRA C and vs real layers at equal FLOPs | algorithmic direct accuracy D T=1..8 0.445-0.451 vs C 0.453 vs pruned B 0.334; text KL D T=8 0.690 vs C 0.364 (`reports/tables.md`, main comparison) |

## What all three support

1. **Loops beyond the trained depth subtract capability; they never add it.** CRR final-loop cyclic models score 0.920 at loop 8 and 0.196 at loop 32 (CRR `tables.md` Table 1, `id` group). In SAE Table 1 the same models end at 0.111 and 0.254, and the recovered mini student falls from 0.840 at its trained 5 loops to 0.157. LaCo's D run at T=16, past its trained loop count, has text KL 4.08 and direct accuracy 0.012 (LaCo `tables.md`). Inside the trained range LaCo accuracy is flat across T=1..8 while KL and knowledge scores worsen past T=2-4.

2. **Whatever benefit appears comes from adaptation or the supervision regime, not from iterating.** In LaCo, D's gain over the pruned model is already present at T=1; one shared adapter matches eight loop-specific ones (KL 0.82 vs 0.94); and training with 1 loop is best on text KL (0.46 vs 0.94 at 8 loops) and scratchpad accuracy (0.51 vs 0.31), with direct accuracy within noise (LaCo `tables.md`, ablations). In CRR, retention is set by the training regime: stable-solve on cyclic is 0.007 under final-loop supervision and 0.641 under random-intermediate supervision (Table 1). The controller wins only on final-loop models whose fixed-depth curve collapses (cyclic +0.29 on trained lengths); against the best fixed depth within budget it is level or worse everywhere: -0.014 [-0.022, -0.007] on trained lengths, -0.094 [-0.141, -0.051] held-out (CRR `tables.md`, best-within-budget). SAE-derived controller inputs add only +0.007 to +0.037, and no controller reaches the best fixed depth on DFA (SAE `REPORT.md` section 4.7, `tables.md` Table 8).

3. **The loop state is readable and locally steerable, but what predicts an outcome is not what causes it.** Ablating the features that write the predicted class moves the readout in 37/45 cells; ablating the features that best predict answer loss rescues almost nothing (SAE `REPORT.md` sections 0 and 4.5). Linear CKA dissociates from function out of distribution, with CKA 0.87 at 22% accuracy (section 4.4). This bears directly on the program's worry that a trajectory can look stable only through a dictionary.

4. **Length shift is where every mechanism fails.** The CRR controller is confidently wrong on longer Storyworld strings (premature-stop rate 0.56-0.58, accuracy -0.21 vs fixed depth; CRR `REPORT.md`). The SAE sparse-code probe's advantage for "answer will be gained" turns negative in 4 of 11 runs on longer strings. LaCo recovers longer-input accuracy to 0.074 at best against a teacher at 0.166 (LaCo `tables.md`).

5. **A saved hidden state is not reusable as is.** Native resumption worsens the longer the state is looped before freezing, pooled -0.055 [-0.101, -0.021] (CRR `REPORT.md` section 2), and resuming scores below a from-scratch run of equal total loops in 3 of 5 SAE runs (SAE `tables.md` Table 6).

## Where the studies disagree or cannot be compared

- **Scale and inference.** CRR and SAE are at d=128 with up to six seeds, Holm correction and cluster bootstraps. LaCo, the only real language model, has one seed, no interval, and 3-5 points of noise per cell. The SAE study's Qwen3 pipeline was never run on real checkpoints, so there is no feature-level evidence at 0.6B.
- **What was looped.** CRR and SAE models are trained from scratch as loops; LaCo loops a block that was one feed-forward layer, adapted for 800 steps. LaCo's finding that loop specificity does not matter and CRR's that supervision across loops matters greatly come from objects too different to compare.
- **Task realism.** The automata are solved at loop 1 or 2 by the random-loop regimes (Table 1, median first-correct loop), so they leave loops little to do. LaCo's tasks are realistic, but no condition moves its knowledge benchmarks (0.49 vs teacher 0.61), so knowledge recovery is untested, not refuted.
- **Task family as a moderator.** Within CRR, extra loops make the frozen state more decodable on Storyworld (+0.185 [+0.143, +0.228]) and less on cyclic (-0.14); pooled claims hide this.
- **Post hoc endpoints.** CRR's sharpest negatives come from the best-within-budget endpoint, which was added after the results.

## The strongest objection and the experiment that answers it

The objection: none of the three studies gave recurrence a task that needs it. The random-loop CRR models answer at the first or second loop on trained lengths, LaCo's block was never a loop before pruning, so the joint result may only say that iteration is useless where one step suffices. CRR's held-out rows hint at this: on longer strings the median first-correct loop moves later (cyclic 1.5 to 2.0, Storyworld 1.0 to 3.0; CRR `tables.md` Table 1, `extrap` group) while accuracy at loop 32 still falls. The models iterate further on longer inputs but cannot hold the result.

The answer: train looped and untied models at matched FLOPs, six seeds each, with random-intermediate supervision, on the program's pointer-chase tasks, where required depth is explicit and held-out depths exceed trained ones. Gate-passing LOOP-T models for those tasks exist on branch `rmp-v0-phase2`. Register one endpoint before the held-out split is read: accuracy at held-out depths when the loop runs to the required depth versus the trained depth, with the untied model as control. If running past the trained depth solves deeper instances the untied model cannot, the October negative is task-bound; if not, claim 1 extends to tasks built to need recurrence.

## Next experiments, cheapest first

1. **Re-analysis, no training.** For every CRR cell, tabulate accuracy at each loop from 8 to 32 on the N=10,12 split and ask whether any model's best depth on longer strings exceeds 8. The logged predictions exist under `results/full/logs`.
2. **Retrain only the CRR controller** with held-out lengths mixed into its training strings; the 72 base models are cached. This tells whether the length-shift failure belongs to the controller or to the state.
3. **Two more LaCo seeds** for conditions B, C-matched, D T=1 and D T=8, plus the fairer variant the report names, D with LoRA on the other 20 layers. This puts an interval on the only real-model result.
4. **Run the SAE study's Qwen3 pipeline** on regenerated LaCo D checkpoints, repeating the class-writing-feature ablation and the code-cosine-to-teacher measurement at 0.6B.
5. **The required-depth experiment above**: looped versus untied at matched FLOPs on pointer chase, six seeds, endpoint registered in advance.
