# MOLT-GOBLIN-01: strict-prefix recurrence pilot

Development-only, inspired by [Mixture-of-Loops Transformer](https://arxiv.org/abs/2611.34923).
This is an independent tiny PyTorch construction, not a source-compatible MoLT or U-MoLT reproduction.
Companion DataGoblin/HybridTRMLDT/BPD/Compactification-Theory branches use
`research/molt-goblin-01-20261126`.

## First executable comparison

One tied block, fixed zero null slot and strict-prefix attention over previous-pass K/V.
Prefix arm restarts its query from the embedding every pass. Residual arm feeds back its own
previous state as query. One-pass control shares the same parameterization but less compute.
Prefix and residual arms pair initialization, tokens, batch shapes, updates, block visits and
optimizer. Use same width/loops and identical training data; report one-pass compute separately.

```sh
python -m pytest -q experiments/molt_goblin_v1/test_pilot.py
python -m experiments.molt_goblin_v1.pilot --steps 64 --loops 4 --threads 2 --seed 31
```

Run within an external operator hard-cap wrapper; CUDA is opt-in. No downloads.
The cumulative mod-7 prefix sum is a competence smoke, not a proof of reasoning.
Report train token count, block visits (not exact FLOPs), timings, CUDA allocation and
token/whole-sequence accuracy at lengths 8/16 and increasing loops. Verify parallel vs
topological parity with tolerance, because batched floating point order can differ.
The sequential fixture recomputes projected prefix memory; it is not optimized decoding.

## Next locked study (proposal)

Before training at scale, calibrate a competence floor and paired variance on independent
development roots. Add matched untied depth and parameter-matched baselines, full causal
attention with same-position recurrence, upstream U-MoLT reproduction, random matched retention,
LDT membrane-gated and BPD component-gated variants. Separate equal-token, equal-estimated-FLOP,
equal measured-time and equal verifier-call analyses. At least eight independent model seeds
for development variance only; determine confirmation count with EvalMath before selecting
a heldout architecture. Do not reuse historical sealed sets. Capture all arms and failures.

Port to LBF (retained candidates under actual Lean verifier calls), modular-recurrence/pointer tasks
in recurrence monitorability and typed lattice chain tasks. Use canonical task generators and
family-disjoint roots; avoid learning label artifacts. Measure verified solution yield, closure,
false holds, recovery/corruption, causal matched-sham effects, exact authorization decisions and
length/depth extrapolation. A synthetic prefix-sum smoke is insufficient for promotion.

CPU/RAM: run fixed threads under 2/4/8/16 GiB total memory caps; cache frozen calibration
component maps/receipts only with checkpoint and graph version tags. Same-graph training K/V
may retain gradients; cross-update cache reuse requires recomputation or a separately labeled
stale-cache experiment. Record RSS, hit rate, byte traffic, activation checkpointing and runtime.
Use fresh-cache and random eviction controls. Profile RTX 3050 and 5080 independently.

Mechanistic evidence: consult EvalMath mechinterp-power skill and FSI-01; fit BPD components on
training roots, lock minimum meaningful causal correctness delta, matched physical dose and
future-output Jacobian control, report paired recovery/corruption. No confirmation release or
training claim from source-only changes.
