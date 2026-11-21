# PEF-07 | Prospective 12-bit coalition and additive-control transfer

Registration date: 2026-11-15. This document's standalone first commit
is the preregistration, preceding PEF-07 implementation, model training
and heldout scoring. Parent context: PEF-06 (mathematical PRs
Compactification-Theory #146 and MathCan #52; posthoc frozen-model PR #64).

## Hypothesis and independently generated task family

The earlier PEF-03 through PEF-06 source family used eight-bit histories,
256 possible words, and heavily reused interventions. This new experiment
uses twelve chronological binary inputs (4096 possible words). The oldest
four bits (a,b,c,d) are focal historical signals. The remaining eight
bits (context) are uniformly distributed distractors, with no task
information beyond their identity.

Two **separately trained, exactly matched** task arms:

- COALITION target y = (a AND b) XOR (c AND d), a Boolean interaction
  involving two disjoint historical bit pairs.
- THRESHOLD target y = 1[a+b+c+d >= 2], realized by a purely additive
  **teacher score** a+b+c+d-1.5 followed by a threshold. Its output
  can have pair-only class changes despite zero *teacher logit*
  four-corner interaction.

Both task labels ignore all eight context bits. The target distribution
is exactly 6/16 positive for COALITION and 11/16 positive for THRESHOLD
on every complete focal quartet. The corresponding context-only Bayes
accuracy ceilings are exactly 10/16 (62.5%) and 11/16 (68.75%).

## Frozen source split and counterfactual integrity

Enumerate the 256 eight-bit context words. Sort by
SHA256(b"pef07-v1-quartet:" + bytes(context)) then lexicographic
context bytes. Assign the first **160 contexts** to parameter training,
next **48** contexts to downstream probe training, remaining **48**
contexts to heldout evaluation.

For each assigned context, include **all 16** four-bit histories
(a,b,c,d), ordered by integer mask 0..15, so no single or paired
old-bit counterfactual ever crosses a split. The exact split sizes
are 2560 trunk-training histories, 768 downstream-probe histories,
768 heldout histories. The split is sealed before any model training.
Each heldout quartet has exact target-label balance and is unseen
at the eight-bit context level by both training and probe fitting.

The source family overlaps the 8-bit *alphabet and possible local
subwords* with earlier research; the 12-bit complete input words
and 8-bit context-closed cohort are newly defined here.
This is a prospective heldout result **within this new task family**,
not a proof of general language-model transfer.

## Architecture, optimizer, seeds

Import only the preexisting repository
lsa.mini_transformer_probe.MiniAttentionMlpResidualBlock.
An input bit is embedded as a transient token at position 0 and
the recurrent memory occupies position 1. The same tied causal
attention/MLP operator is visited exactly **12** times. Only the
memory token persists.

Architecture: 32 hidden channels, four attention heads, MLP ratio 2,
residual alpha 1.0, beta 0.2. A trainable learned bit embedding,
two trainable role vectors, trainable initial memory and a final
**two-logit** classification head. No other targets or probes
receive gradients during trunk training.

Training seeds: **103, 131, 167**, each trained separately for the
COALITION and THRESHOLD task (six models total). Exact initialization
torch.manual_seed(seed+0 if coalition else seed+10000). CPU FP32,
torch==2.10.0+cpu, torch.set_num_threads(1),
torch.use_deterministic_algorithms(True), environment variables
OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
PYTHONHASHSEED=0, DNNL_MAX_CPU_ISA=AVX2. The backend/processor ID
must be reported; pinning oneDNN ISA does not guarantee all cross-host
bitwise weights (PEF-05R).

Optimizer: AdamW lr=0.003, weight_decay=0.0, exactly **450** updates,
batch size **128**, loss is binary two-class cross-entropy, gradient
clip max norm 1.0. Parameter-training minibatches are sampled
from the 2560 parameter-training pool only using a local
torch.Generator().manual_seed(seed*7919 + 59 + (0 for coalition,
100000 for threshold)). No hyperparameter search, early stopping,
seed exclusion, learning-rate schedule or test feedback.

All six final 450-step state_dict checkpoints are saved with
initial/final parameter SHA256, full checkpoint bytes SHA256, exact
source split SHA256 and scalar first/final training loss **before
any probe-head training or heldout scoring**. Individual model
checkpoint hashes identify the actual scored weights.

## Primary evaluation and source controls

1. For every heldout example score the native trained head after
   **12 natural visits**, and store correct count /768,
   balanced accuracy, per-class recall, the source-only
   context baseline and a matched seed-specific random
   initialization control (untrained task head).
2. Continue the frozen model for **two additional natural visits**
   with bits (0,1) for all histories, producing h14. Fit an
   intercept-augmented ridge head (penalty 1.0, train-only
   standardized features, threshold 0.5) to the same task label
   on the **768 separate probe histories**, then evaluate on
   the 768 heldout histories. Fit the same ridge head to
   randomly initialized trunk h14 states and score identically.
   The h14 fitted readout is a **new downstream task head**, and
   does not imply the original trained native head uses the
   surviving information after 14 visits.
3. On every heldout context fix a=b=c=d=0 and evaluate the
   naturally computed two-bit flip effects of all six focal-bit
   pairs. Count the *teacher* pair-only changes and actual
   native-head pair-only changes. For the native two-logit
   difference z=logit_1-logit_0, compute four-corner
   D_ij z = z(x^ij)-z(x^i)-z(x^j)+z(x).
   Also compute D_ij h12 in L2, under the same heldout context.
4. Teacher COALITION at the four-bit-zero base has two
   pair-only active pairs (01,23) out of six; each cross-pair
   is inactive. THRESHOLD has all six pairs active from
   four-bit zero, while its *additive teacher score* has
   zero four-corner differences. This is the preregistered
   threshold-only negative control.
5. Report the mean absolute native margin D_ij for the two
   intended coalition pairs separately from the four
   cross-pairs at all-zero focal histories. Record normalized
   ratio = (mean intended abs+D)/(mean cross abs+D+1e-6)
   and the **same ratio in the THRESHOLD arm** of each
   seed. This is a descriptive mechanism contrast that
   does not identify a unique learned AND/XOR circuit.
6. Report h12 class-only pair effects as well as the additive
   logit surrogate z_add_ij = z_i+z_j-z_0, and count
   how often its predicted class agrees with the actual
   double-flip class on the all-zero focal context.
   This controls threshold-only pair effects.
7. For each trained model report the sum of six pairwise
   mean h12 second-difference hidden-state L2 values and
   matched random-initialization controls for final-head
   and h14 probe accuracies. No activation splicing.

## Preregistered decision

**Primary task-learning success** requires at least **two of three**
COALITION seeds with native h12 heldout accuracy >= **80%**
(>=615/768) and h14 frozen-trunk downstream-probe accuracy
>= **75%** (>=576/768). All three THRESHOLD model scores,
class baselines, random-trunk scores and all six checkpoints
must be reported regardless of result. Missing this criterion
is an explicit negative result.

**Independent diagnostic** (not required for primary success):
for at least two COALITION seeds, the h12 intended-pair
absolute mixed-logit ratio at the four-bit-zero source
exceeds **1.25** and exceeds the same seed's THRESHOLD
ratio by at least **0.20**. This is a preregistered
diagnostic with a separate pass/fail value, not an
alternate success pathway.

Report per-seed actual task, downstream probe, mixed-logit
pair ratios, teacher pair-change pattern, additive-surrogate
agreement, random controls, CPU backend and all checkpoint
SHA256s. Do not change any threshold after inspecting results.

## Trust and compute

A single CPU GitHub Actions job has timeout **20 minutes** and no
paid GPU. Optional local CPU replay runs the exact same source
with the same provenance. Model checkpoints and exact JSON
receipts are uploaded as artifacts; code and protocol go to a
new PR. The existing PEF-06 PR #64 and older negative registered
PEF-04/05 outcomes remain untouched.

A resulting signal is evidence about six small synthetic trained
tied-attention models and a new 12-bit finite source family.
It is not evidence of pretrained-LLM generalization, memory
consciousness, safety behavior or Lean reasoning improvement.
