# LS-NEURON-02: MLP-dependent recurrent Boolean pruning

**Development only; FSI-01 REDESIGN_OR_PILOT_ONLY.** Follow-up to [LS-NEURON-01 #357](https://github.com/MoralityLabAI/LoopedTransformers/pull/357). No original NEC04 worlds, held-out TRM Sudoku cases, or other sealed readouts accessed. New synthetic task inspired by [NEC04A](../neuron_communities/NEC04A_PROTOCOL.md) and the [PEF07 native Looper](../../research/pef07/pef07.py); these are fresh proxy models rather than original native TRM checkpoints. Consulted the [EvalMath mechanistic preflight](https://github.com/MoralityLabAI/EvalMath/tree/main/skills/evalmath-mechinterp-power) and [FSI-01](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/finite_sample_inference.md).

## Method

Four sequential visits, each with two focal bits and one nuisance bit. The prefix target is XOR of all focal bits seen so far. All 16 nuisance contexts contain all 256 focal masks; per seed, allocate 8 train, 4 calibration and 4 development contexts (2048/1024/1024 histories). This is a small exact enumerated task within each context, with context-level dependence and only three checkpoint seeds (11, 23, 37). The same seed's models share data partition and minibatch order. Training uses all prefix binary cross-entropies, AdamW, LR .003, batch 128, gradient norm clip 1. PyTorch 2.10.0+cpu, deterministic single-thread execution. No paid compute.

The affine-bypass model has 16D state, linear residual recurrence on state and input, plus GELU MLP scaled by .2. If the whole MLP is removed, all outputs are affine functions of the original bits; conditional on the other bits, an affine threshold cannot classify all four XOR corners (at most 3/4). Thus on complete balanced context blocks no affine classifier can exceed 75% final accuracy. The attention variant uses the memory-token portion of a PEF07-inspired two-token 4-head residual block with learned attention, norms and .2 branch gain. Attention can itself solve parity. For same-width and approximate-budget untied arms only MLP banks vary by visit; the full_budget arm also unties the bypass or attention/norms. In attention mode full_budget is a misleading internal label: its 4677 parameters exceed the tied 3345, even with width 1.

Frozen post-training masks rank absolute activation times BCE gradient on calibration contexts, accumulated over all visits using each bank (global), separately per visit (local), or random masks (3 repeats). Count unique channels and active channels per application separately. Zero-channel mask retains output bias; whole-MLP intervention removes bias as well. All correctness is computed without post-pruning optimization. Checkpoints from dense training and separate narrow training are preserved. No actual sparse dispatch or low-bit kernels measured.

## Results

57 trained checkpoints: 12 affine at 600 updates, 12 affine at 2400 updates (independent deterministic rerun, not continuation), 12 attention at 800 updates, 6 dense-function-matched fork checkpoints, 15 narrow-width checkpoints. Each table entry is equal-seed mean final correctness percentage.

| Architecture | Affine parameters | 600 dense | 2400 dense | 2400 MLP removed |
|---|---:|---:|---:|---:|
| Tied width 64 | 2673 | 100 | 100 | 50.00 |
| MLP untied width 64 | 9633 | 51.24 | 100 | 49.90 |
| MLP untied width 16 | 2721 | 50.07 | 100 | 49.80 |
| Full untied width 8 | 2529 | 49.90 | 49.77 | 51.07 |

The full untied affine model fails the dense competency gate. Do not interpret its small dense-relative pruning loss as robustness.

### Pruning after 2400 updates (global salience)

| Retained channels per application | Tied 64 | MLP untied 64 | MLP untied 16 |
|---:|---:|---:|---:|
| 1 | 50.16 | 49.84 | 51.30 |
| 3 | 49.51 | 49.97 | 51.66 |
| 8 | 51.66 | 55.37 | 77.83 |
| 16 | 54.13 | 69.95 | 100 |
| 24 | 57.39 | 88.02 | n/a |
| 32 | 88.05 | 96.97 | n/a |
| 48 | 100 | 100 | n/a |
| 64 | 100 | 100 | n/a |

At equal active width 32, tied trails same-width untied by 8.92pp. At 48 unique channels, tied gets 100% while the 16-wide untied arm with 12 active per visit gets 92.45%; the tied model uses 4x as many active channel applications. Minimum tested count reaching >=95% dense-relative correctness in seed order: tied 48/48/32, untied same width 32/32/32 (128 unique each), approximate-budget untied 8/12/16 (32/48/64 unique). This is a descriptive grid threshold rather than a global sparse-subset optimum. Accuracy can be nonmonotone with mask count.

At 600 updates, tied width 64 has 99.77% at 32 retained and 49.67% at 3 retained. Global salience outperforms random at moderate width; at 32 retained the 600-update tied arm is 99.77% vs 58.16% random and 86.56% local. At 2400 updates, tied global vs random at 32 is 88.05 vs 60.04. Training history matters.

### Exact-function fork

Copy each 600-update tied checkpoint into four independently trainable MLP banks with shared affine bypass, check exact calibration logits (max difference zero), then fine-tune tied and untied children for 600 identical minibatches, LR .001 with new AdamW optimizers. All six children retain 100% dense. At 32 active, tied 83.63% vs untied 68.91%; at 48 active, 100% vs 97.98%. This reverses the independently initialized 2400-update comparison. At 3 retained, 50.23 vs 49.97. Initialization and training path are important confounds; no general architecture ordering established.

### PEF07-inspired attention proxy (800 updates)

All 12 dense checkpoints = 100%. Complete MLP removal: tied 56.22%, same-width untied 47.56%, width-16 untied 69.89%, fully untied width-1 93.95%. Tied at 3 retained = 49.22%; same-width untied = 46.78%; tied at 32 retained = 82.03%, untied = 97.82%. Width-16 untied at 12 retained = 100%. Attention bypass remains a serious alternative explanation, especially for narrow/full untied arms. Tied global salience at 32 underperforms random (82.03 vs 89.67). No native TRM claim.

### Train narrow from scratch (1200 updates)

| Tied width | Params | Seed 11 | Seed 23 | Seed 37 | Mean |
|---:|---:|---:|---:|---:|---:|
| 2 | 441 | 93.16 | 75.00 | 60.16 | 76.11 |
| 4 | 513 | 93.75 | 99.80 | 100 | 97.85 |
| 8 | 657 | 97.07 | 92.97 | 100 | 96.68 |
| 16 | 945 | 100 | 100 | 100 | 100 |
| 32 | 1521 | 100 | 98.44 | 100 | 99.48 |

Width 4 is 93.75% fewer MLP channels and 80.81% fewer total parameters than the 2673-parameter dense tied model, and ~48pp above the same-width mask on the 2400-step dense model. Separate optimization budget; this does not prove a four-neuron subnetwork already exists in the dense model.

### Width, horizon, precision grid

288 evaluations, including 2/4/6/8 visits and 32/8/4/2-bit nominal weight precision, three seeds, six widths. Widths below 64 are from the narrow 1200-update runs; width 64 is the 600-update tied baseline. Symmetric per-tensor fake quantization of all parameters including biases (2-bit has three levels, not binary). Activations remain float; no packing, memory or speed claims. At 2 and 4 visits, complete focal masks within development nuisance contexts; at 6/8 visits, 4 nuisance contexts and 512 randomly sampled focal masks per context shared across that seed's conditions. Target is parity of all bits at each evaluated horizon. These are repeated measurements, not independent checkpoint replications.

| Width | FP32 T4 | FP32 T6 | FP32 T8 | 8b T4 | 8b T8 | 4b T4 | 4b T8 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 76.11 | 69.69 | 66.31 | 76.04 | 65.64 | 65.49 | 55.34 |
| 4 | 97.85 | 86.47 | 63.75 | 97.85 | 64.29 | 78.48 | 54.44 |
| 8 | 96.68 | 88.51 | 65.04 | 96.09 | 62.81 | 78.91 | 52.08 |
| 16 | 100 | 94.08 | 86.33 | 100 | 86.30 | 87.21 | 60.76 |
| 32 | 99.48 | 96.60 | 82.39 | 99.48 | 83.22 | 95.51 | 66.93 |
| 64 | 100 | 98.80 | 96.42 | 100 | 96.16 | 99.25 | 82.71 |

At nominal 2-bit, final T4 scores 45.96–53.94% across widths. Narrow models show shorter stable computation horizons and reduced tolerance to weight quantization. Budget, architecture and optimizer history differ; monotone scaling law not established.

## Evidence gate

Independent architecture unit: seed/checkpoint group, n=3 per contrast. Each scored development condition has 1024 decisions per checkpoint at T4, 3072 across seeds, clustered by four nuisance contexts per seed. Training seed and partition seed are coupled; the 256 focal assignments within one context are exhaustive and correlated, not independent replicates. All comparisons and follow-on fork/width/precision sweeps are development diagnostics with adaptive exposure. Primary next proposed endpoint: difference in dense-relative correctness at equal active width and equal unique storage (separately), at least 5pp beyond the best qualified matched control; require dense >=95% and >=10pp headroom above complete MLP removal. No reliable matched physical sham floor or prospective power estimate: random masks are only count-matched and repeated three times. Freeze hypotheses, model cohorts and masks, collect >=59 independent sham worlds if using a nonparametric 95/95 maximum tolerance bound, run EvalMath preflight and FSI-01 on independent seed/world units. Neither negative results nor sign counts establish equivalence. No neuron semantic stability conclusion from this run.

Next: compare exact causal single-neuron/greedy backward salience and learned sparse masks, with frozen calibration selection and norm-matched controls; measure recovery after equal-budget fine-tuning and transfer on original native MLP-dependent checkpoints only after checking their independent headroom. Separately train genuinely narrow transformer blocks and recurrent length curricula, then measure actual low-bit kernels. Keep the original NEC04 and Sudoku protected populations sealed. The result supports a compact *learned* recurrent implementation on the toy XOR task and weakens the general claim of extreme post-training sparsity.

## Reproduction and receipts

From this directory:
```sh
python pilot.py --mode affine --updates 600 --seeds 3 --out affine
python pilot.py --mode affine --updates 2400 --seeds 3 --out affine_long
python pilot.py --mode attention --updates 800 --seeds 3 --out attention
python fork.py
python narrow.py
python precision.py
python audit.py
```

First affine run used pilot_v1.py, identical to committed pilot.py except the attention-only typo self.proj(...) rather than self.proj[k](...). The initial attention attempt failed immediately; corrected source was used for all 12 completed attention runs and long affine runs. No effect on affine behavior. The archive preserves original source, failed-attempt repair provenance (v1 source), final logs, all 57 weight checkpoints and raw per-case correctness receipts, and SHA256 manifest. Audit independently checks scalar XOR and disjoint context ownership, 57 checkpoint SHA256 hashes, and all 114 dense/complete-removal correctness vector replays. It does not independently retrain or replay every mask or quantization row.

Receipt SHA256:
- affine/results.json: 258a6bb8c932e46457b791002844e9cff376486d5f84099c49a69aef3f25b49f
- affine_long/results.json: 64bfe6c9e910d2f1241c01dbc7945707326f354979c66db9403aa1a2aee6bd05
- attention/results.json: d0846f1221883b888ad16f030ba457379540a052c46098c22a0239a7d4d61ef7
- fork/results.json: 19dbc5b25d2a892ecb641c87050e8e9129e5b046d1e3cdf5589df3842fe3bee5
- narrow/results.json: e1d2f261e126287855b2a142ad9998e8495eeee1d9cc516d433c6810ce124c6d
- precision_results.json: 656c379632d02352bf2546186fa2df822eff0b6907da94d1c2805cf8e97011d1

Raw local archive SHA256: 23f76dad425acc9ab76c258d4c6835b4bf9d8c6d04a22a66784b22141c3bcea8. GitHub contains source and this report, not checkpoint binaries.
