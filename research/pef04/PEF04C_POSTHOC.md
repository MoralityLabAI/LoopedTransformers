# PEF-04C | Exact training-stream replay and historical-memory collapse

This is an explicitly **posthoc** continuation of the preregistered
[PEF-04 negative result](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43683120934).
The retrospective scan was selected after observing six random-init
old-bit readouts at 48/48 and the train-versus-own-initialization
differences measured in PEF-04B. It has no independent prospective
threshold or checkpoint-selection decision.

The diagnostic re-executes the EXACT original training streams using
the original six seeds, 250 AdamW steps, same fixed pair-closed data,
same model architecture, and same optimizer and sampling generator.
At prespecified retrospective sample steps
\`[0,10,25,50,75,100,125,150,175,200,225,250]\`, it briefly
evaluates a downstream ridge head fitted on the original probe split
and scores only the original heldout pairs. Evaluation never changes
the optimizer state, gradients, or weights. After step 250 it requires
a **bitwise identical original final parameter SHA256** and exactly
matching terminal loss, original-task accuracy and oldest-bit
decoding score. Any mismatch invalidates the replay.

Each sample records old-bit accuracy out of 48, jointly correct
counterfactual pairs out of 24, mean natural 00-continuation
oldest-bit hidden-state separation, its normalization by activation
norm, directional coherence of differences, and original task
accuracy. Scoring at intermediate steps is retrospective;
evaluation reuse means the best intermediate checkpoint cannot
be cited as a prospectively validated generalization result.

Preview observations on the unchanged training streams:
- seed 157: oldbit **48/48** at step125, **34/48** at step150,
  **24/48** at step225 and step250; causal mean pair L2 drops
  2.1687 at step125 to 0.0773 at step150, while the supervised
  task improves from 45/48 to 48/48.
- seed 101: oldbit **41/48** at step125 to **35/48** at step150,
  then **26/48** at step175 and step250.
- seed 127: oldest-bit pair separation remains large at the
  endpoint while its linear decodability worsens. Thus small
  response amplitude is not a complete explanation of lost
  linear prediction.
- seed 109 and 131 retain excellent oldest-bit linear decoding;
  these branches provide retained-memory controls at the same
  optimization budget.

This can identify intervals of training during which memory
changes sharply, but cannot determine a unique circuit or causal
optimizer update. A subsequent experiment would need preregistered
interventions on a chosen layer/gradient component, with heldout
test data not reused in selecting a checkpoint.

The source run and six checkpoint SHA256 digests are pinned in the
hosted workflow; the original PEF-04 negative verdict remains
unchanged. Artifacts contain only JSON trajectory receipts.
