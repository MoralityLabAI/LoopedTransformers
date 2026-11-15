# PEF-04 preregistration: independent-seed latent memory transfer through natural recurrent continuation

**Date:** 2026-11-15. **Registration commit:** this file's first commit on
\`research/pef04-prospective-pairclosed-transfer-20261115\`, before
PEF-04 model training, probe fitting, or PEF-04 heldout scoring.

## Lineage and falsifiable question

PEF-03 (LoopedTransformers PR #50) established exploratory task-irrelevant
oldest-bit memory in a trained, tied, 16-dimensional causal attention model
on synthetic eight-bit histories. Its follow-ups PEF-03N/F/G used
posthoc selected checkpoints and sometimes off-manifold hybrid patches.
PEF-04 tests **six new independently initialized/trained checkpoints**
and a **natural recurrent continuation** through two further steps.

Primary question: does oldest-history information, never supervised in
trunk training, remain linearly decodable after two additional steps
on natural trajectories at a frequency that reproduces across seeds?

The primary readout is a newly fitted downstream **linear ridge head**;
the recurrent trunk is frozen before fitting this head. A downstream
readout trained after freezing establishes use by a later task. It
does not establish that the original four-class head would use this
information, or an ability to learn a new task without a probe.

## Fixed architecture and training

- Import \`research.pef03.pef03.BitMemoryLooper\` from the PEF-03 source;
  tied \`MiniAttentionMlpResidualBlock\`, memory width 16, two attention
  heads, MLP ratio 2, alpha 1, beta 0.2. At each visit input bit token
  is ephemeral; only the memory token carries forward.
- Eight chronological independent fair input bits. Supervised trunk
  label is the *two most recent bits*, four-way cross-entropy.
  Oldest bit index 0 is never a target or auxiliary loss.
- Six newly selected seeds: **101, 109, 127, 131, 149, 157**. Same
  architecture and data for all; no checkpoint filtering or best-seed
  selection.
- CPU only, torch==2.10.0, Python 3.12, FP32 math. Use 1 CPU thread,
  \`torch.use_deterministic_algorithms(True)\` when supported. For
  each seed exactly 250 AdamW steps, lr 0.003, weight decay 0,
  batch 64, gradient norm clip 1.0. Fixed history-pool sampling
  generator seed \`seed*7919 + 59\`; model init \`manual_seed(seed)\`.
  No learning-rate search, rerun-based selection or early stopping.
- Train only on the **parameter-training pair pool** below. Write
  checkpoint with torch state_dict and initializer/final weight hashes
  before constructing any downstream probe or heldout output. Reload
  through PEF-03 \`load_frozen\` and verify hash, optimizer steps,
  source split digest and checkpoint SHA256.
- Separate matched **random-initialized trunk** generated from
  \`seed + 50000\`, with no gradient updates. A new ridge head can be
  fitted to its states under the same probe/heldout protocol.

## Pair-closed split and leakage control

Index histories by their last seven bits (128 distinct seven-bit
suffixes). For each suffix, include BOTH chronological length-eight
histories \`(0,*suffix)\` and \`(1,*suffix)\` in the same split.
Sort 128 suffixes lexicographically by
\`SHA256(b"pef04-v1-pair:" + bytes(suffix)).digest()\`,
then break ties by the suffix bytes. Allocate the first **80 pairs**
(160 histories) to trunk parameter training, next **24 pairs**
(48 histories) to downstream probe training, final **24 pairs**
(48 histories) to heldout scoring.

Each suffix occurs in exactly one split. Both oldest-bit
counterfactuals remain in that split. The heldout suffix and its
flipped counterpart are absent from trunk parameter training
and probe-head training. No source-prior prediction of the oldest
bit is possible from any function of the newer seven bits on this
perfectly balanced evaluation cohort: its maximum accuracy is 50%,
and it cannot correctly predict **both members of any one pair**.

The split receipt must store the sorted suffix hashes, original
training/probe/heldout indices, SHA256 of the partition JSON,
and verify pair closure and disjointness before training.

## Natural continuation and downstream heads

At visit eight, hold the fully trained recurrent model fixed.
For each length-eight history, compute an unmodified h8 trajectory.
Continue two further \`model.step\` calls using **identical appended
bits for both members of each pair**. Primary continuation is
\`(0,0)\`. Secondary, independently frozen continuation is \`(1,1)\`.
No head or trunk weights are updated by those continuation bits.

For each seed, stage and continuation fit a one-dimensional
intercept-augmented ridge head to the *oldest bit only* on the 48
probe-training histories. Use PEF-03's \`fit_ridge\` and
\`predict_ridge\`, feature standardization on probe split only,
penalty=1.0, threshold at 0.5. The primary is h10 after \`(0,0)\`;
h8 and h10 after \`(1,1)\` are secondary diagnostics.

Also fit separate heads to the random-initialized trunk states
using exactly the same probe and heldout histories. Report the
random-trunk baseline per seed, even if it equals or exceeds the
trained-trunk result.

## Registered metrics and decision rule

For every seed and readout arm report:
- ordinary heldout oldest-bit correct count out of **48**;
- count of the **24** suffix pairs with both members correctly
  decoded using the same frozen downstream ridge head;
- fraction of pairs where the prediction changes under the
  oldest-bit-only natural counterfactual;
- mean and minimum Euclidean distance between frozen naturally
  generated memory states for those paired histories;
- heldout original four-class task accuracy after eight steps,
  random-initialized task-head accuracy, and training loss;
- downstream probe-only coefficients (including intercept) SHA256,
  checkpoint SHA256, and unchanged-after-evaluation check.
- Exact suffix-only baseline: 24/48 correct, 0/24 pairwise both.

**Primary success criterion:** for the h10 \`(0,0)\` arm at least
**four of six new seeds** must achieve **at least 36/48** correct
and **at least 12/24** jointly correct pairs. In addition, at least
four of six trained trunks must achieve **at least 41/48** correct
on the original supervised four-class task. The random-init control
distribution will be reported regardless of criterion outcome.
The result is a descriptive six-seed finite study; this rule is
an experimental gate, not a calibrated statistical p-value.

Failure to meet the criterion is a recorded negative or mixed
outcome. No change in seeds, training steps, split, thresholds,
continuation, or checkpoint choices is authorized by evaluation.
If numerics produce nonfinite states, stop and retain an integrity
failure instead of silently discarding that seed.

## Integrity, compute and claim boundary

- CI run trains exactly six independent **fresh** trunks, on CPU,
  with a 15-minute job timeout; no remote GPUs or paid cloud.
- No original PEF-03 checkpoint is reused for the prospective
  success criterion. Previous PEF-03 seed17/29 scores are
  historical context only.
- The final-stage intervention is a **natural source counterfactual**
  (flip original bit 0 and rerun all 10 visits). No intermediate
  state splicing, head-nullspace patch or synthetic off-manifold state
  enters the primary analysis.
- Check SHA256 of frozen source checkpoints before and after
  all probes. Artifact includes checkpoints and exact JSON
  evidence; PR includes source/protocol/tests only.
- Model type remains a small synthetic recurrent attention architecture;
  the result neither establishes a theorem for pretrained LLMs nor
  confirms the PEF dyadic affine precision law for neural weights.
