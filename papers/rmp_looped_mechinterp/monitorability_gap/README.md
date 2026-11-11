# RMP-MG-01: fixed-target probes and readout sensitivity

This experiment targets `MoralityLabAI/LoopedTransformers` on `rmp-v0-phase2` and uses the existing semantic-echo native adapter. The mathematical definitions are in [theory.md](theory.md). The [protocol](protocol.md) separates the fitted-observer study from local causal effects and semantic validation.

## Validation recorded in revision 2

The local run passed 81 tests and ten constructed controls. Native-forward execution passed state/logit parity using selected definitions extracted from upstream commit `80d2ce8d073c6b634eab56f94c3f24c1a3c3a9e1`. The dependency snapshot covers `make_batch`, `Block`, `LoopedChase`, and the semantic-echo native execution/loader functions. It is a validation fixture; it is excluded from the repository patch so the committed code imports the existing repository modules.

Cursor-only and all-token observations were tested on 12 item/visit cases. A float32 repeat checks numerical precision. The fixed-target pipeline completed 16 studies on an untrained eight-node actor with 96 prompts divided among 12 structural groups. These are smoke measurements. Evaluation on trained Q4/Q5 checkpoints remains pending their external weights. The native task head is the measured observation channel; natural-language CoT monitoring has its own pending experiment.

The checked-in [validation summary](../../../experiments/recurrence_monitorability_v0/rmp_mg01_validation/summary.json) records source hashes and measurements. Full local receipts, selected probe weights, and the dependency snapshot accompany the release bundle. The lightweight workflow exercises the real repository imports.

## Run tests and constructed controls

```bash
python -m unittest discover -s tests -p 'test_monitorability_gap.py' -v
python -m unittest discover -s tests -p 'test_bounded_probes.py' -v
python -m research_gym.scripts.monitorability_gap_controls --out /tmp/rmp_mg_controls
```

The mathematical controls use NumPy; autograd and native execution require PyTorch. Probe fitting stays on CPU. Actor capture supports CPU or CUDA.

## Native causal smoke

```bash
python -m research_gym.scripts.loopscope_monitorability_gap \
  --smoke --device cpu --dtype float64 --items 4 \
  --visits 1 2 4 --horizon 4 --out /tmp/rmp_mg_native
```

Repeat with `--observation all_token_logits`. For float32, set `--rank-rtol 1e-5 --rank-atol 1e-7`. Every intervention records the realized perturbation norm after dtype conversion.

## Fixed-target probe smoke

```bash
python -m research_gym.scripts.loopscope_bounded_probes \
  --smoke --smoke-nodes 8 --device cpu --dtype float64 \
  --group-counts 6 3 3 --items-per-group 8 \
  --steps 40 --batch-size 64 --shuffle-control \
  --save-features --bootstrap 500 --out /tmp/rmp_mg_probes
```

This command fits affine and small MLP observers for fixed targets at 8 and 16 hops. It compares separately fitted visit decoders with a shared decoder. Selection receipts and weights are saved before test scoring. Reusing an existing output directory raises an error.

For a checkpoint run, replace `--smoke --smoke-nodes 8` with `--checkpoint PATH_TO_CHECKPOINT.final.pt`; use the desired actor device and preserve an explicit fitting budget. The fixed-depth checkpoint contract requires `cfg` and `model` dictionaries. The initial scientific runs should retain Q4/Q5 provenance and their development-set status.

## Split and inference scope

Each split owns complete permutation cycle types. Random starts, edge orders, and renamed graphs cannot cross splits. This tests structural generalization of the observer on the selected families. It does not audit which structures occurred during actor training.

The held-out score is a measured difference in log loss for a fixed-budget observer. The paired bootstrap resamples structural groups and remains conditional on the chosen actor. One test group has no reported interval. Finite samples and fitting error remain part of the result.
