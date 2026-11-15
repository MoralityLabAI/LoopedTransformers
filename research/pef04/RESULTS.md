# PEF-04 | Registered result and retrospective mechanism audits

**Registration precedes the experiment:** [PEF-04 preregistration at commit a82af51](https://github.com/MoralityLabAI/LoopedTransformers/commit/a82af51abb249b76026e91b34a87be6799d5ae69).
The original model is the PEF-03 16D tied causal-attention block with
an ephemeral bit token and a persistent memory token. Six new seeds
(101,109,127,131,149,157) each received 250 fixed CPU AdamW steps
supervising the **newest two bits** of an eight-bit history. The
oldest bit was never a training target.

**Original preregistered CI:** [run 43683120934](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43683120934),
artifact \`pef04-preregistered-six-seed-frozen-checkpoints-and-receipts\`,
ID \`14149561491\`. All six checkpoints and split manifest were
SHA256-verified before heldout scoring.

The dataset is *pair-closed*: all 128 seven-bit suffixes are divided
into 80 trunk-training, 24 probe-training, and 24 heldout suffix pairs.
Both oldest-bit alternatives for each suffix stay together. The
heldout 48 histories have 24 complete counterfactual pairs that
were unseen by trunk training and downstream probe fitting.
A suffix-only decoder is **exactly** correct on 24/48 and
correct on both members of **zero** pairs, regardless of its
strategy. See MathCan/PairedHistory.lean in [MathCan PR #44](https://github.com/patrickdugan/MathCan/pull/44).

## 1. The registered negative outcome

The primary downstream head predicts the original oldest bit using
a probe-only fitted ridge on the **naturally generated** h10 state,
after the original eight bits and two more fixed \`00\` iterations.
No activation patching occurs in the registered score.

| Seed | Original two-bit task /48 | Oldest-bit transfer /48 | Both-oldest-bit pairwise /24 | Different random-initialization control /48 |
| --- | ---: | ---: | ---: | ---: |
| 101 | 40 | 26 | 2 | 48 |
| 109 | 47 | 48 | 24 | 48 |
| 127 | 43 | 35 | 11 | 48 |
| 131 | 46 | 48 | 24 | 48 |
| 149 | 48 | 43 | 19 | 48 |
| 157 | 48 | 24 | 0 | 48 |

**Registered decision: \`criterion_not_met\`.** The oldbit threshold
was at least four seeds with >=36/48 and >=12/24; three attained
it (109,131,149). The original task threshold of >=41/48 was met
by five of six. The fact that all six random-initialized controls
decoded 48/48 was unexpected and motivates the retrospective
diagnostics below. This is a fixed six-seed finite study, with no
inferential p-value or model-selection correction.

## 2. PEF-04B: exactly matched original initializer

**Retrospective diagnostic selected after PEF-04**, preserving the
negative preregistered verdict. [SHA-pinned run 43683668508](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43683668508),
artifact \`pef04b-original-own-init-vs-trained-same-seed-receipts\`
(ID \`14148818519\`).

Each **own original initialization** was reconstructed using its
registered random seed and checked against the
\`initial_weights_sha256\` saved in the pre-evaluation training
ledger. Every original trained checkpoint and its h10 score was
independently replayed exactly. The downstream ridge head was
refitted on the same probe-only histories for each version.

| Seed | Own initialization oldbit /48 | After training /48 | Initialization mean natural pair L2 | Trained mean pair L2 |
| --- | ---: | ---: | ---: | ---: |
| 101 | 48 | 26 | 0.6200 | 0.0921 |
| 109 | 48 | 48 | 0.3465 | 2.0299 |
| 127 | 48 | 35 | 0.8788 | 1.9014 |
| 131 | 48 | 48 | 0.5647 | 2.7746 |
| 149 | 48 | 43 | 0.4732 | 0.6520 |
| 157 | 48 | 24 | 0.4062 | 0.0234 |

Every original initialization was perfectly linearly decodable
after the two natural continuation steps; four of six trained
versions lost some linear decodability. This demonstrates
optimization-associated alteration of old-history representation,
subject to finite-data and finite-model limits. The hidden channel's
causal effect remains nonzero even in poor-performing seeds, so
"forgotten" here means *lost linear decodability under the fixed
fitting protocol*, not proof of zero information.

## 3. PEF-04C: exact deterministic training-stream replay

**Retrospective diagnostic selected after inspecting PEF-04B.**
[Run 43684000507](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43684000507),
artifact \`pef04c-exact-source-optimizer-trajectory-receipts\`
(ID \`14149263903\`).

Six training streams were replayed with the same optimizer, seed,
data order and 250 updates. Every terminal model's parameter
SHA256, final loss, source-task accuracy and oldest-bit score
matched the original registered source. Intermediate states were
sampled only at optimizer steps
\`0,10,25,50,75,100,125,150,175,200,225,250\`.
No early-stop checkpoint or replacement final weight was selected.

| Seed | Step 125 oldbit /48 | Step 150 oldbit /48 | Step 250 oldbit /48 | First sampled checkpoint below 36/48 |
| --- | ---: | ---: | ---: | --- |
| 101 | 41 | 35 | 26 | 150 |
| 109 | 47 | 48 | 48 | None |
| 127 | 47 | 37 | 35 | 175 |
| 131 | 48 | 48 | 48 | None |
| 149 | 45 | 48 | 43 | None |
| 157 | 48 | 34 | 24 | 150 |

The strongest amplitude collapse is **seed 157**: mean
old-bit causal pair distance shrank from 2.1687 to 0.0773
between steps 125 and 150, around 28-fold, while the supervised
task improved from 45/48 to 48/48. By step 225 its oldest-bit
linear readout was 24/48, with zero jointly correct pairs.

For **seed 101** the mean old-bit distance declined to 0.0921 by
step 250. Directional coherence across paired differences also
fell from 0.813 at step 125 to 0.410 at step 250.

For **seed 127**, the final mean old-bit causal separation
remained large (1.9014), while oldest-bit linear decodability
fell to 35/48. Its directional coherence fell from 0.6305
at step 125 to 0.4816 at step 250. This is consistent with
reorientation or increased conditional overlap rather than
simple amplitude erasure. The present diagnostics cannot
attribute the change to one circuit or gradient update.

Seeds 109 and 131 are useful high-retention controls under
exactly the same training protocol.

## Scientific boundaries and next experiment

All PEF-04B and PEF-04C measurements were **selected after**
the original preregistered negative result and reused the
original heldout data for retrospective temporal diagnostics.
Their intermediate scores establish properties of those fixed
training streams. They do not constitute a new heldout model
selection or a fresh preregistered capability result.

The next controlled study should preregister weight/gradient
interventions on the **training intervals where old-bit
distance collapses**, compare a strong-retention seed as a
matched control, and score new source families with hidden
target and source-only baselines. A principled experiment can
separate recurrent contraction of an old-bit direction from
redistribution into state directions inaccessible to a fixed
linear decoder.

No task improvement for a pretrained LLM or a Lean reasoner
is claimed here. All experiments concern one tiny recurrent
attention architecture and finite synthetic binary sequences.
