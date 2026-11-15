# PEF-05 preregistration | Training-window causal gradient rescue

**Registration:** 2026-11-15, first commit on \`research/pef05-gradient-rescue-20261115\`, before the first PEF-05 gradient intervention, outcome or heldout scoring. **Base:** PEF-04 PR #55 commit \`0aebea73278a9f71159f64e718faa0acf79069fc\`.

## Why this experiment

PEF-04's prospectively registered oldest-bit-transfer criterion was missed. In a separately labeled posthoc replay, some models lost oldest-bit linear decodability during the optimizer updates between steps **125 and 150**, despite retaining the supervised newest-two-bit task. The current experiment tests a *specified, train-only directional gradient intervention* during exactly that window.

This window and the seed set were selected from the prior PEF-04 data, so they do not represent independent discovery. PEF-05's own new partition, intervention assignment and final endpoint decision are frozen here before PEF-05 training.

## New pair-closed split and prospective continuation

- Every chronological 8-bit history is represented by a seven-bit recent suffix and an oldest bit in {0,1}. Sort the 128 suffixes by \`SHA256(b"pef05-v1-pair:" + bytes(suffix))\`, breaking ties lexicographically. Allocate the first **80 pairs** to trunk and anchor training, next **24 pairs** to downstream ridge fitting, last **24 pairs** to final heldout. Both oldest-bit alternatives are always assigned together. Train and probe use only their partitions.
- PEF-04's prior mathematical setting includes the same 256 possible length-eight histories. This is a newly frozen **within-PEF-05** heldout split rather than a claim that its entire stimulus universe has never appeared in preceding research.
- Train-only old-bit preservation objective is evaluated after two further naturally generated visits with fixed continuation **(0,1)**. The *primary* final oldest-bit readout also uses natural h10 continuation **(0,1)**, a full ten-visit sequence absent from PEF-04's earlier fixed (0,0)/(1,1) readouts. Secondary final diagnostic uses h8 with no continuation.
- The source-only heldout baseline remains exactly 24/48 correct and 0/24 jointly correct oldest-bit pairs because both oldest-bit labels are equally represented for every seven-bit suffix.

## Fixed model, seeds and optimizer stream

- Import the existing 16-dimensional, two-head, tied \`research.pef03.pef03.BitMemoryLooper\`; train only the **newest two** of eight bits as a four-way classification task. The oldest bit is never a supervised label.
- Four prospectively fixed seeds: **101, 109, 127, 157**. Seeds 101/127/157 had weak oldest-bit retention previously; seed 109 is a strong-retention control. All four are analyzed. No checkpoint selection.
- Set \`torch.manual_seed(seed)\`, single-thread CPU FP32 PyTorch **2.10.0**, \`torch.use_deterministic_algorithms(True)\`; AdamW learning rate **0.003**, weight decay **0**, batch **64**, gradient norm clipping **1.0**, **250** total updates. Batch sampling uses \`torch.Generator().manual_seed(seed*7919+59)\`, a predetermined common sequence for all arms within a seed.
- Run steps **1 through 125** with the unmodified supervised gradient, then freeze the complete model weights, optimizer moments and batch-generator position. Fork **three** exactly identical starting states and optimizer states. All arms consume identical future supervised training batches.
- Arms: \`baseline\` (unmodified clipped task gradient), \`protect\` (first-order projection against a decreasing oldest-bit separation score), \`sham\` (random gradient-component removal matched in Euclidean norm to the protection removal at the current state). Branch updates **126 through 150 inclusive** use their selected rule. Updates 151 through 250 revert to the unmodified clipped task gradient.
- On the common state at step 125, compute the signed mean difference \`mean(h10(1,s)-h10(0,s))\` across all 80 **training** suffix pairs, using continuation (0,1). Fix its unit vector \`u\` as a seed-specific anchor through all arms; require a positive norm.
- At intervention steps, compute \`M(theta)=mean_s <h10(1,s;theta)-h10(0,s;theta),u>\` over the **first 16 training suffix pairs**, taking a gradient \`m=grad_theta M\`. The score and its gradient depend only on the training split and never on downstream probe or heldout data. The same score/gradient computation occurs in protect and sham.
- With the already clipped supervised task gradient \`g\`, let \`p=m/||m||\` if nonzero, \`c=max(0,<g,p>)\`. The protect arm uses \`g'=g-cp\`, so \`<g',m> <= 0\` up to numeric tolerance. Under an infinitesimal first-order SGD step, this prevents reduction of \`M\`; AdamW's adaptive moments and finite step sizes are measured and may violate exact preservation.
- The sham arm chooses a fixed-generator random unit vector \`q\` constrained to \`<g,q>=c\`, where the perpendicular component is sampled from a deterministic Gaussian seed \`seed*100003+step\`; it applies \`g'=g-cq\`. When \`c=0\` or \`||m||=0\`, both modified arms leave \`g\` unchanged. Thus protect and sham remove **the same current-step Euclidean gradient-norm amount** for each arm's own state. Both use the same extra train-only score-gradient computation; q has no target-specific direction.
- Fail integrity on nonfinite values, mismatched common starting hashes/optimizer-state hashes, nonidentical batch indices across arms, a sham norm-matching violation above **1e-5 absolute**, a protect first-order dot product above **1e-5** when active, or any change of a frozen source checkpoint during downstream probing. No arm can look at probe/heldout data until all three 250-step checkpoints for that seed have been written.

## Registered readouts and primary decision

For each seed and arm, fit the same intercept-augmented standardized linear ridge to the oldest bit on the 48 **probe-only** histories at h10(0,1), penalty **1.0**, threshold **0.5**. Evaluate the **48 heldout histories forming 24 complete paired counterfactuals**. Compute ordinary oldest-bit correct count, both-members-correct pairs, prediction flips, mean and minimum oldest-bit memory-state L2 distance, original four-class supervised task accuracy, full checkpoint SHA256, exact gradient intervention counts and removal norms. Record the task-only source baseline and pre-split initialization performance.

Secondary metrics are h8 readout, train-only anchor memory score at steps 125/150/250, supervised training loss, random-init accuracy, and per-visit old-bit natural causal separation. These secondary values may explain results but cannot change the primary decision.

**Registered causal rescue criterion:** across vulnerable seeds **101, 127, 157**, at least **two** show *protect* oldest-bit h10(0,1) correct count ≥ *sham*+**8** and both-members-correct pairs ≥ *sham*+**4**, while the original supervised task stays at least **40/48** and no more than **3/48** below the sham arm. The retention-control seed **109** must lose at most **4/48** oldest-bit correct predictions relative to sham. A failure is recorded as \`criterion_not_met\` with every arm's results intact. The \`baseline\` arm documents whether the previously observed spontaneous forgetting reproduces on this newly allocated dataset.

The primary research claim concerns the effect of the registered gradient intervention in this bounded architecture. Any rescue must also be compared against an untrained initializer, and success is not by itself proof of general LLM reasoning improvement or a precise map of hidden SAEs.

## Compute and reproducibility

- CI CPU-only, fixed PyTorch 2.10.0 with a 15-minute job timeout. The same source code and optimizer setup run on GitHub-hosted CPU workers. No GPU, paid cloud, changed previous PEF-04 checkpoints or merge.
- Save **all four common step-125 branch checkpoints** and **all twelve final step-250 arm checkpoints**, including optimizer state and training-only intervention receipts. Freeze before probing. Keep binaries in hosted artifacts; commit only source/protocol/tests and lightweight audit documents.
- No posthoc threshold changes, extra seeds, adaptive interval extension, hyperparameter search, or best checkpoint selections on failure. Any follow-up requires a new experimental label.
