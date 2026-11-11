# Readout-silent recurrent-state intervention pilot

Locked before training/evaluation in this session, 11 November 2026.

Purpose: test whether the current native answer-head output is a sufficient
statistic for subsequent recurrent computation in a trained tied-attention
transformer. Use LoopedChase from research_gym/scripts/loopscope_toy.py.

Existing Q4/Q5 checkpoints are excluded from Git and unavailable in this
runtime. This CPU pilot trains fresh small checkpoints; it makes no claim
about the original Q4/Q5 weights or a pretrained language model.

Training: seeds 101,102,103; n_nodes=6, d=64, layers=1, heads=4;
4 recurrent visits, 2000 AdamW steps, batch 64; mean per-loop cross entropy,
learning rate 0.002 with cosine decay to 5%, 100-step warmup; weight decay
0.01; betas=(0.9,0.98), gradient norm cap 1.0; float32 CPU, one thread.
No graph-structure holdout: evaluation draws use independent generator
seeds, and graph values can recur in this finite population.

Evaluation: 512 fresh item draws, seed 941117; trace 16 loops; continuation
horizon 4. Candidate pairs have the SAME graph/start/input embedding, visits
1<=t<u<=12, matching oracle node, and both native endpoint top-1 answers
correct. Choose one pair uniformly per eligible item with fixed seed 941118;
choose up to 256 items uniformly. Selection must not inspect future outcomes.

Patch only graph-edge token states, leaving the answer token exactly intact.
The entire native current-logit vector must match bit for bit. Doses 0.25 and
1.0; sham, matched-norm Gaussian direction on edge tokens, matched-norm
per-edge constant-coordinate shift (removed by native injection LayerNorm),
and full-state donor replay as a separate plumbing control. Fresh random
directions use seed 941119. Do not tune them using results.

Primary: fraction of selected pairs with changed future native top-1 answer
within four visits, per seed and stratum. Report continuous probability TV,
current-logit equality, mean task accuracy and perturbation norms as well.
Within-training vs cross-boundary vs beyond-training visit pairs are reported
separately. Later clean-future correctness may define descriptive strata only.

Qualification gate: native head accuracy at training depth 4 must be >=90%
on all 512 evaluation draws before a checkpoint supports a trained-capability
claim. Weak models are retained and labeled. Control violations invalidate a
run. Three seeds are a pilot; there is no population-level frontier-model claim.

An edge transplant may create an off-trajectory state. A changed future
falsifies sufficiency of the present readout over the tested intervention
class. It alone does not establish a semantically meaningful circuit, faithful
state abstraction, or latent consciousness. Compare to matched-size damage.
