# PEF-07 | Prospective twelve-bit coalition transfer

[PRE_REGISTRATION.md](PRE_REGISTRATION.md), sealed separately at
[commit a3b441c](https://github.com/MoralityLabAI/LoopedTransformers/commit/a3b441ce5fbc8f8d33914a1a34c883875ea5873c),
defines the fixed data split, six seeds, model, tasks, training
schedule, source-only baselines and success criterion.

## Source and learned tasks

The eight-bit PEF-03 through PEF-06 history universe had only
256 possible inputs. PEF-07 creates **4096 distinct twelve-bit inputs**:
four oldest target bits (a,b,c,d) and eight independent context
bits. Each context is assigned in its entirety with all sixteen
possible old-bit interventions to one of the train, probe, or heldout
sets. There are 160/48/48 disjoint contexts, yielding 2560/768/768
source histories, respectively.

Two tasks are trained separately using exactly matched tied-attention
architectures and three fixed seeds:

    COALITION: (a AND b) XOR (c AND d)
    THRESHOLD: indicator(a+b+c+d >= 2)

The THRESHOLD teacher uses the additive *prethreshold* logit
a+b+c+d-1.5. It is a control for pair-only class changes caused
by thresholding additive scores. Both tasks ignore the eight
context bits. Their maximal context-only accuracies on the complete
heldout focal-hypercube groups are **62.5%** and **68.75%**.

## Frozen experiment

Each of six models has 32 hidden channels, a weight-tied two-token
causal-attention/MLP recurrent block, twelve visits, and a two-class
task head. Training uses exactly 450 CPU AdamW updates on the
parameter-training split. All six checkpoints and SHA256 receipts
are saved before any fitted probe or heldout scoring.

The primary task score uses the trained native head at visit 12.
A separately trained ridge readout for the same task is fitted on
the probe-only states after two **natural further visits** with
input bits (0,1), then evaluated on the 768 heldout histories.
The two additional visits are out of the original training horizon.

An untrained random-initialized trunk undergoes the exact same
ridge fitting procedure, isolating prior representational
capacity from task-training effects.

The heldout mechanism audit measures actual four-corner logit
and hidden-state differences for all six pairs among the
oldest four bits. At focal source 0000, the COALITION
teacher activates only pairs 01 and 23; the THRESHOLD
teacher activates all six pairs, though its additive
prethreshold score has exactly zero mixed differences.
The additive-logit surrogate is a further threshold-only
negative control for each observed class change.

## Decision and limits

See the registration for both frozen criteria. The primary
success rule requires two of three coalition seeds with
native head >=615/768 and fitted h14 head >=576/768.
A separate mixed-logit diagnostic checks whether the intended
pairs are more selective than cross-pairs and the matched
threshold task. Both failures and positive results remain in
the saved JSON.

This experiment measures generalization over **unseen
eight-bit contexts in a newly defined 12-bit synthetic task**.
The same four-bit teacher logic applies to all contexts, and
the results do not establish spontaneous generalization to
unseen Boolean functions or pretrained language models.

Run:

    python -m pytest -q research/pef07/test_pef07.py
    python -m research.pef07.pef07 --out /tmp/pef07_registered

The workflow pins Torch 2.10.0 CPU and oneDNN maximum AVX2 ISA,
reports processor provenance, and uploads the six immutable
checkpoint artifacts and the complete source/score ledger.
