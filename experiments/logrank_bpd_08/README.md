# LOGRANK-BPD-08: staged learned component calibration

Development pilot, no confirmation. Six fresh program roots: 16D 1901/1907/1913, 32D 2003/2011/2017. Fixed random bijective query codes and known literal/parity/AND dictionary as in 07. Fit depth-3 dictionary on 256 training memories; all fitted programs unique, exact on training, and independently equivalent to generating programs. Train hard router and bank using only fitted instructions and training task labels; test all query codes on 1024 disjoint memories. No transfer to unseen query IDs claimed. 500 Adam updates (batch 128, lr .01), width-32 GRU, identical initialization seed and minibatch sequence within root. Fixed three-slot vectorized execution with semantic active length 1..3, one selected component per active read; all slots and dense V projection computed, so active reads do not establish measured FLOP savings. Communication uses fixed q-bit query address plus one answer bit (model consumes code; no wire serialization benchmark). Read bank V has unit scalar writes; this is BPD-inspired training rather than BPDTrainer. 8118 allocated parameters at k16, 9926 at k32, trainable counts differ in frozen-bank arms.

## Reproduce

```sh
cd experiments/logrank_bpd_08
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python pilot.py --k 16 --out receipt16.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python pilot.py --k 32 --out receipt32.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python teacher.py --k 16 --out teacher16.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python teacher.py --k 32 --out teacher32.json
OPENBLAS_NUM_THREADS=1 python audit.py receipt16.json receipt32.json teacher16.json teacher32.json > audit.json
python snap.py teacher16.json teacher32.json > snap.json
```

First five arms formed original development plan; two teacher arms were chosen adaptively after initial rotated-bank failures and reuse these exposed roots. Teacher-joint auxiliary literal loss reads the column selected by the fitted program rather than by the current predicted mask. Teacher-warmup uses 150 updates with instruction and teacher-selected literal losses alone, then 350 updates with task and sparsity loss restored. Same 500 optimizer updates; supervision/compute differ by arm.

| Arm | held-out correct /147456 | exact instructions /144 | certified queries /144 | perfect roots |
|---|---:|---:|---:|---:|
| identity | 147456 | 144 | 144 | 6 |
| rotated_fixed | 85036 | 51 | 0 | 0 |
| rotated_learned | 102814 | 87 | 2 | 0 |
| no_trace | 88200 | 65 | 0 | 0 |
| task_only | 79739 | 0 | 0 | 0 |
| teacher_joint | 104250 | 96 | 1 | 0 |
| teacher_warmup | 147456 | 144 | 102 | 6 |

Independent audit replayed all 1,032,192 saved held-out decisions from extracted routes and stored V, recomputed labels, split disjointness, instruction counts and rational dominance certificates. It checks saved-route inference, not optimizer trajectories or original dictionary search optimality. No router checkpoints are archived; raw receipts retain V, extracted routes, predictions, losses and seed/split information. The certificate uses largest absolute coordinate with exact rational margin m=2 max|v_i|-||v||_1, requires m>||v||_1/65536 for conservative fp32 rounding headroom, and verifies composed program equality on its union support. It certifies the mathematical function of stored coefficients; runtime guarantee requires the stated arithmetic assumptions.

## Signed-coordinate projection

Project each learned teacher-warmup component onto its maximal absolute coordinate with original sign. All six projected banks preserve every held-out prediction (147456/147456) and certify all 144 query functions by exact support-local truth-table equality. Only 102 of the original dense queries were certified by the conservative dominance test; global equivalence to the original dense model on the other 42 is not established. This is an adaptive structure-exploiting compression result.

For k=16 a raw dense FP32 bank occupies 8192 bits versus 80 bits for 16 signed coordinate indices (102.4x bank-payload reduction). At k=32, 32768 versus 192 bits (170.67x). Excludes router parameters, query program table, memory, accumulator and metadata; no whole-model compression claim. Multiple components can select the same coordinate in general. Native BPD PR #166 has one imported learned-bank replay fixture (root 1901, 256 sampled decisions) and scoped CI; native trainer remains unrun.

## Next development gate

Freeze teacher-warmup and projection before new, non-isomorphic query families. Include noisy teacher traces, unseen composition/addresses, larger arity and a fixed identity strong control. Compare curriculum-specific task-only controls, match training resources, and use EvalMath actual root-level sham calibration before native causal or confirmatory claims. No protected cohorts used.
