# Results: recurrent recovery of LaCo-pruned Qwen3-0.6B-Base

Setup (see `REDUCTIONS.md` for every deviation): teacher A = Qwen3-0.6B-Base + task LoRA (28 layers). LaCo collapses layers 8-15
into one (m=7, 21 layers left, 0.99 vs 1.22 GFLOPs/token). C = LoRA on all remaining layers; D = collapsed layer replaced by the
shared recurrent block with 8 loop-specific rank-16 adapters, per-loop norms, per-loop gates (2.90M trainable, C matched at
2.84M). 800 steps (400 for ablations), one seed. Full numbers: `tables.md`; figures: `figs/`.
Teacher-logit KL is measured against A. "LONG" = longer than any training example.

## Headline

| | B pruned | C LoRA (matched) | D T=1 | D T=2 | D T=4 | D T=8 | A teacher |
|---|---|---|---|---|---|---|---|
| text KL to A (nats, lower better) | 0.759 | **0.364** | 0.466 | 0.459 | 0.517 | 0.690 | 0 |
| knowledge: ARC-E/C + PIQA acc_norm | 0.488 | 0.492 | 0.479 | 0.482 | 0.487 | 0.450 | 0.611 |
| algorithmic ID, direct answer | 0.334 | **0.453** | 0.445 | 0.451 | 0.445 | 0.438 | 0.475 |
| algorithmic LONG, direct answer | 0.039 | 0.072 | 0.057 | 0.059 | 0.061 | 0.074 | 0.166 |
| algorithmic ID, scratchpad | 0.115 | **0.725** | 0.477 | 0.504 | 0.502 | 0.416 | 0.723 |

## Interpretation

1. **Extra recurrent compute recovered neither knowledge nor algorithmic reasoning in this setup.** D's gains over B appear already
   at T=1-2 and are what ordinary adaptation gives; adding loops leaves in-distribution accuracy flat (0.44-0.45 for T=1..8) and makes
   text KL and ARC/PIQA worse beyond T=2-4. The only place more loops line up with a gain is LONG direct accuracy at T=8 (0.074 vs 0.057 at
   T=1), but that is ~9 questions out of 512 and equal to C (0.072): not distinguishable from noise.
2. **Repeating a block is not the same as the deleted layers** (condition E): with untrained adapters the same collapsed block applied
   T times degrades monotonically (text KL 0.76 -> 0.81 -> 1.64 -> 5.9 for T=1,2,4,8; direct accuracy 0.33 -> 0.00). Training adapters
   makes T up to 8 usable (KL 0.69 at T=8 vs 5.9) but never better than T=1-2. T=16, beyond the trained loop count (last adapter reused), collapses.
3. **At matched FLOPs, real layers beat loops.** Keeping T-1 more LaCo layers (m = 7-T+1) plus params-matched LoRA gives text KL
   0.29 / 0.19 and ARC-PIQA 0.52 / 0.54 at the FLOPs of D T=2 / T=4, against 0.46 / 0.52 and 0.48 / 0.49 for D. Even the *untrained* B at m=4
   (KL 0.33, MC 0.52) beats trained D at T=4.
4. **At matched trainable parameters, D does not beat C.** C is better on text KL and on scratchpad accuracy (0.725 = teacher, vs
   0.42-0.50 for D) and ties on direct accuracy. Caveat: C's parameters are spread over all 21 layers, D's sit in one block, so this
   compares "adapt everything" with "adapt one looped block"; D with the same budget cannot repair the rest of the network, and D at T=1 is
   already worse than C. A fairer follow-up would add LoRA on the other layers to D.
5. **Knowledge vs algorithm.** What recovery exists is mostly *algorithmic and in-distribution* (direct accuracy 0.33 -> 0.45 for both C and D; teacher 0.475),
   while knowledge benchmarks barely move for any condition (0.49 vs teacher 0.61), consistent with the idea that collapsed layers
   carry stored knowledge that a re-used block plus small adapters cannot rebuild. Length generalisation (LONG) is recovered only ~25 % by any method.

## Ablations (D, 400 steps, evaluated at the largest trained T; `figs/fig4_ablations.png`)

- **Adapter rank** (2..32): higher rank is monotonically better on text KL (1.08 -> 0.88) and scratchpad accuracy (0.18 -> 0.33), ~flat on direct accuracy.
- **Loop count**: training with 1 loop (a single adapter; KL 0.46, scratch 0.51) beats 2/4/8 loops on every metric at its own T; more loops degrade.
- **Sharing**: one *shared* adapter (0.38M params) is at least as good as 8 loop-specific ones (KL 0.82 vs 0.94, scratch 0.41 vs 0.31); frozen / shared / per-loop norms
  and the gate make no measurable difference. With *no* adapter the 8-loop block is destroyed (KL 5.1). So loop specificity is not what helps.
- **Objectives**: CE-only, KL-only and KL+CE are within noise. Matching residual states to the teacher (`--hid final|steps`) lowers T=8 text KL (0.94 -> 0.86 / 0.80)
  but does not change task accuracy. Intermediate-state *tokens* (scratchpad) only help scratchpad accuracy (no scratch: 0.04); direct-answer accuracy is unchanged
  (0.44), i.e. no transfer of intermediate supervision to latent computation here.

## Caveats

- One seed; task sets are 64 examples per cell (ablations 32), MC sets 400/150 items: differences under ~3-5 points are noise.
- Own LaCo search (single range, `last` merge) stands in for the LaCo reproduction project, which had no final recipe yet; results may differ with its recipe (`--laco-recipe`).
- 0.6B only; 8B was not attempted (shared GPU, little free RAM). Short training (800 steps); loop adapters for T=8 get ~1/4 of the steps at each T.
- Latency/memory were measured on a GPU shared with other jobs; latency is noisy (e.g. one 1.3 s outlier for C m=6) - only the memory column and FLOPs are reliable.
- Teacher is task-tuned Base, so "knowledge" recovery is measured against a teacher whose ARC/PIQA score (0.611) slightly exceeds the untuned Base (0.565).
