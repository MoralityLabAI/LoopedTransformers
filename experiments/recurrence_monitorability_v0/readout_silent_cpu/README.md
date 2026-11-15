# Readout-silent interventions in a trained looped transformer

11 November 2026. Executed CPU pilot, with frozen selection and a separately
labeled post-hoc manipulation check. This starts the neural transfer of the
OIC observations without extending the finite modular-sensor search.

## Primary result

The locked same-input experiment found **zero discrete continuation changes**
after edge-state transplantation in 256 selected pairs per seed. Every seed
had 100% native task accuracy on 512 fresh input draws at each loop 1 through
16, after training at four loops. The result is limited to this small task
and the tested finite perturbations. Continuous probabilities changed slightly.

| Seed | Parameters | Accuracy at loops 4 / 16 | Same-task full edge patch | Maximum future TV |
|---:|---:|---:|---:|---:|
| 101 | 53,062 | 100% / 100% | 0 / 256 | 0.0001081 |
| 102 | 53,062 | 100% / 100% | 0 / 256 | 0.0125074 |
| 103 | 53,062 | 100% / 100% | 0 / 256 | 0.00000201 |

The full current native logit vector was identical, bit for bit, before and
after all edge-only patches. The cursor state is untouched. Selecting donor
and recipient visits requires the same input graph/start and oracle node,
plus correct current native answers. Selection never uses future predictions.
One pair is sampled per item. The same 512 draws are reused across seeds, so
768 cases are a descriptive pooled count rather than 768 independent graphs.
There is no graph-identity holdout from the finite training distribution.

The 256 pairs in each run include 6 within training depth, 158 crossing the
training-depth boundary, and 92 with both visits beyond training depth.
All unpatched four-step continuations on selected pairs are correct.

## Post-hoc cross-task control

After recording the null result, a separate calibration copied edge states
from a different graph with the same current oracle answer and a different
next oracle answer. The recipient input embedding and answer token stayed
fixed. The calibration was declared before computing its intervention outcomes.

| Seed | Full cross-task edge patch: changed future | Matched-norm random edges | Next answer follows donor |
|---:|---:|---:|---:|
| 101 | 23 / 256 | 1 / 256 | 6 / 256 |
| 102 | 21 / 256 | 2 / 256 | 9 / 256 |
| 103 | 9 / 256 | 0 / 256 | 0 / 256 |

Totals are 53/768 changed continuations for the cross-task patch and 3/768
for matched-size random directions. Quarter-dose patches produce zero discrete
changes. Sham and LayerNorm-null controls have zero changes in every run.
Full cross-task patches have mean L2 size 1.13 to 1.16 times the receiver's
complete state norm; same-task donor patches average 0.15 to 0.39 times it.
This is a large, potentially off-manifold intervention. Only 15 first-step
answers follow the donor graph. This supports sensitivity to an unobserved
state intervention, with limited evidence of semantic transfer.

## What was actually trained

The native LoopedChase model has tied self-attention and an MLP block, with
input reinjection at each recurrence. The model learns graph pointer chasing;
there is no supplied phase clock or exact symbolic belief operator inside it.
The task remains a synthetic algorithmic task.

The runtime contained none of the external Q4/Q5 trained weights. We trained
three fresh smaller checkpoints with width 64, four heads, one block, six graph
nodes, four training loops, and 2000 AdamW steps per seed. Each batch has 64
items and all four loop outputs are supervised. The source architecture and
data generator were copied from the native repository at
`80d2ce8d073c6b634eab56f94c3f24c1a3c3a9e1`, source blob
`8ef8dbb12726fad18a23573bf6228e65a151023d`.

The local package includes an architecture-only snapshot of make_batch,
Block and LoopedChase; the repository integration imports the original file.
Native complete-state tracing reproduces the model's forward cursor and head
outputs exactly. This result is independent of the original Q4/Q5 cohort.
A pretrained language-model experiment has not been executed.

## Interpretation

Write F(h,e) for one native recurrence and G(h) for the native answer head.
An edge-only patch P satisfies G(P(h)) = G(h) exactly because G reads only
the unchanged cursor. A later difference G(F^j(P(h),e)) != G(F^j(h,e))
falsifies sufficiency of the current answer-head output over that intervention
class. The primary same-task test found no discrete witness. The cross-task
calibration found witnesses after introducing conflicting graph memories.

A finite splice can leave the natural state manifold. Such a witness gives
neither a full causal abstraction nor evidence of a hidden objective. Distinct
hidden states can also be irrelevant to this task's future answer. Report
both null results and continuous-output differences without promoting one
successful manipulation into a claim about all recurrent computation.

## Reproduction and Q4/Q5 handoff

From repository root, run the original trained cohort without retraining:

```sh
python -m research_gym.scripts.loopscope_readout_silent \
  --checkpoint-dir experiments/recurrence_monitorability_v0/loopscope_toy_q5/ckpt \
  --out experiments/recurrence_monitorability_v0/readout_silent_q5 \
  --items 512 --loops 32 --horizon 4 --max-pairs 256 --device cuda
```

The directory must contain safe-loadable `*.final.pt` files with native `cfg`
and `model` dictionaries and `fixed_depth=True`. Missing weights cause an
explicit error. Existing weights are never modified. The script validates the
native replay and logs source/checkpoint hashes before reporting any outcome.

Reproduce the fresh CPU pilot:

```sh
python -m research_gym.scripts.loopscope_readout_train_cpu --seed 101 \
  --out experiments/recurrence_monitorability_v0/readout_silent_cpu/checkpoints
python -m research_gym.scripts.loopscope_readout_silent \
  --checkpoint experiments/recurrence_monitorability_v0/readout_silent_cpu/checkpoints/cpu_s101.final.pt \
  --out experiments/recurrence_monitorability_v0/readout_silent_cpu/results/s101
python -m research_gym.scripts.loopscope_readout_calibration \
  --checkpoint experiments/recurrence_monitorability_v0/readout_silent_cpu/checkpoints/cpu_s101.final.pt \
  --out experiments/recurrence_monitorability_v0/readout_silent_cpu/results/s101
python -m research_gym.scripts.verify_readout_silent \
  experiments/recurrence_monitorability_v0/readout_silent_cpu/results
```

Repeat for seeds 102 and 103. Local PyTorch 2.10.0+cpu and NumPy 2.3.5,
float32, one CPU thread per fit; three training fits were run concurrently.
Training wall times per fit were 35.7 to 36.2 seconds in this runtime.
Nine unit tests pass in standard and optimized Python. An independent NumPy
checker recomputes both experiments' counts and task labels from saved arrays.
The downloadable research package retains all three trained checkpoints and
full raw arrays. `PUBLIC_RESULTS.json` binds them to SHA-256 digests. The NumPy
audit validates stored predictions; a full optimization replication uses the
commands above. Source provenance is explicit about the local architecture
snapshot.

## Methodological background

Activation-patching interpretation and metric choices follow Zhang and Nanda,
*Towards Best Practices of Activation Patching in Language Models* (2024),
arXiv:2309.16042, and Heimersheim and Nanda, *How to use and interpret
activation patching* (2024), arXiv:2404.15255. The distinction between a finite
intervention result and a full causal abstraction follows Geiger et al.,
*Causal Abstractions of Neural Networks* (2021), arXiv:2106.02997.
