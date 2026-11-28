# InkyNano Stage 1 local routing gym

The original trainer and the full Stage 1 sweep ran on Windows using an explicitly synthetic RoutingGym fixture. Each arm used 12 training roots and four development roots. This establishes a local training and replay path for verified routing traces. The fixture cannot establish teacher-routing performance, generalization, RTX 3050 CUDA performance, or quality under restricted native routing.

## Teacher archive and workstation

The companion receipt records a 33,104,911-byte routing-only archive, with SHA-256 `16d1930ec23f91e28003bb654011215438116fa15c5b609a4566abf39a02b5b4` and 18 per-file manifest entries. The available source points to Silico experiment `exp_01m863sawber29ms3bjnr4eg89` through an `artifact://` URI. The companion PR carries the packaging code and receipt. It had no attached download, release or Actions run exposing the payload, and the workstation had no copy or Silico connection available. The original archive hash is pinned in `stage1.py extract-pilot` and remains unverified locally against that payload. The synthetic fixture passed its own manifest check. The pinned native pilot reported 58.91% successive-pass top-8 overlap on generated continuations and generated H0 oracle mass of 45.05%, 64.08% and 83.81% at 8, 16 and 32 slots. These are measurements from the companion receipt. [Measured pilot receipt](https://github.com/patrickdugan/SilicoExperiments/blob/e43c01156e82585402f24183d30516c346754444/InkyNano/PILOT_RECEIPT.md)

The workstation had 30.24 GiB free disk and 7.64 GiB available physical RAM at preflight. NVIDIA reported 4,096 MiB GPU capacity, 3,477 MiB used, 486 MiB free and 85 °C during an active rollout. WDDM withheld the compute process name and per-process memory. The gym reused PyTorch 2.10.0+cu126 and NumPy 2.4.3. The first sweep used CPU, one Torch thread and batches of four. It took 41.67 s, with 24.17 s in training across 24 checkpoints. Peak process working set was 591.5 MiB and incremental peak was 574.3 MiB. The run sampled a minimum of 7.32 GiB available host RAM.

After the rollout, NVIDIA reported 3,962 MiB free. The full CUDA repeat took 44.03 s, with 27.27 s in training across 24 checkpoints. Peak incremental process RAM was 1.04 GiB. PyTorch allocated 18.7 MiB and reserved 24 MiB at peak, with a 1.5 GiB allocator cap. A device-wide NVIDIA sample during training showed 115 MiB used. The runner checks memory during trace assembly, fitting and evaluation, writes checkpoints and receipts atomically, and can resume an interrupted run. Only the routing schema fixture and small controllers were loaded.

## Protocol

- Read each arm separately and keep the archived root and family splits. Reconstruct pass-level prompt and completed-prefix features from the native visits and cross-check their collapsed arrays against the supplied NPZ files for teacher inputs. One window is one target segment. The synthetic generator uses 160 generated tokens and 12–24 forced tokens per root and shares five of eight IDs between consecutive loop passes by construction.
- Compare training-only global top-8 frequency, prompt demand, strictly preceding window demand with prompt initialization, and three learned variants. The independent, layer-mesh and pass-mesh controllers each have 17,440 parameters at width 32 and use the same per-root-weighted soft-target loss and 20-epoch training budget. Same-pass messages pool six-layer blocks and their shared state. The pass-mesh also reads the preceding pass representation computed from evidence available at the current window boundary. All variants receive the same pass-indexed inputs. The layer-mesh serves as the parameter-matched inter-pass ablation.
- Each physical layer has a 128-expert powerset lattice with intersection and union. Residency is a deterministic projection onto capacity-constrained nodes with a uniform per-layer expert byte cost and a bounded number of replacements. Pass predictions collapse into one shared per-layer cache. Count cold fills and subsequent admissions. Use all three capacities and four horizons, with 0/2/4/8 replacements where a second window exists. The forced fixture has one window at each horizon, so every forced configuration uses budget zero.
- The exact per-segment hindsight oracle chooses the best cache for each observed segment with unlimited replacements. Budget zero also has an exact fixed-turn oracle. The positive-budget hindsight replay is greedy across segments. Headroom in the receipts uses the exact per-segment oracle. Root mass weights visits within a root and gives each dev root equal weight in the final mean.
- Adaptive refresh uses forecast drift measured by total variation since the previous evaluation boundary and discounts estimated admission gain by normalized entropy. The primary run uses a 0.002 drift floor and charges 0.025 expected mass per 64 MiB of admitted payload. Replay evaluates already scheduled boundaries. Transfer estimates apply one H2D copy per admission at a default uniform 3,981,312 bytes, the raw dense INT4 payload floor. The estimate excludes scale and alignment, full-bank prompt prefill, and other runtime costs. Pass layer-specific estimates with `--expert-bytes` when measured values are available.

## Synthetic development results

Illustrative slice for the generated arm at H32, 16 slots per layer and up to four replacements per layer per boundary. Byte estimates sum over four cold-start development roots.

| Policy | Root-mean retained mass | Admissions | Estimated H2D bytes |
|---|---:|---:|---:|
| Global top-8 frequency | 21.83% | 1,536 | 6,115,295,232 |
| Prompt | 54.96% | 1,536 | 6,115,295,232 |
| Previous window | 72.09% | 2,648 | 10,542,514,176 |
| Independent learned | 72.12% | 1,560 | 6,210,846,720 |
| Layer-mesh, matched control | 72.19% | 1,562 | 6,218,809,344 |
| Pass-mesh | 72.16% | 1,562 | 6,218,809,344 |
| Pass-mesh, adaptive | 71.89% | 1,536 | 6,115,295,232 |
| Hindsight, capped greedy | 86.76% | 2,830 | 11,267,112,960 |
| Hindsight, exact segment oracle | 87.00% | 3,000 | 11,943,936,000 |

The pass-mesh has 14.84 percentage points of oracle headroom in this slice. Across 51 CPU fixture comparisons it beat the layer-mesh in 18, with differences from −0.77 to +0.58 percentage points. At the illustrated setting the adaptive gate saved 26 replacements at a cost of 0.26 percentage points. In a zero-cost sensitivity run with identical weights and the same drift threshold, the gate made 22 replacements and retained 72.18%. The fixture's forced-window controls reduce to fixed residency. The per-root receipts include admissions, transfer estimates, pass-level mass, refresh counts, oracle headroom, split counts and model time/memory.

The existing `gym_train.py` path ran at generated H64, eight slots and two replacements on the same fixture. Its independent controller retained 40.48% and its original mesh retained 40.18%, with 48,704 and 26,720 trainable parameters respectively. Its prompt and cumulative-history controls retained 28.23% and 31.70%. The newer matched controls and strictly previous-window baseline are the primary comparisons.

The unit is the independent fixture root. Windows, passes, capacities, budgets and reruns reuse the four dev roots per arm. The primary descriptive contrast enables a preceding-pass message at the same parameter count and replay limits. The 51 configuration comparisons reuse the development readout, which was unsealed. No scientific effect floor, matched-sham floor or confirmation power estimate was calibrated on this fixture. Treat the results as pipeline diagnostics. A teacher-data comparison needs new independent source families to quantify paired root variation, set a minimum useful gain at a specified transfer budget and fix any confirmation comparison before scoring it. [EvalMath finite-sample guidance](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/finite_sample_inference.md)

## Reproduce locally

From the repository root, use an existing environment with Python, NumPy and PyTorch. The full legacy test suite also imports safetensors. All 25 tests passed. Run the following for the synthetic fixture.

```sh
python -m unittest discover -s InkyNano -p "test_*.py"
python InkyNano/stage1.py make-synthetic work/synthetic-pilot --seed 17
python InkyNano/stage1.py run --synthetic --input work/synthetic-pilot --out work/stage1-run --device cpu --epochs 20 --batch 4 --threads 1 --seed 17
python InkyNano/stage1.py run --synthetic --input work/synthetic-pilot --out work/stage1-cuda --device cuda --epochs 20 --batch 4 --threads 1 --seed 17
```

For the original pilot, acquire the routing-only tarball from the Silico experiment, then run the pinned archive and per-file verifier before training.

```sh
python InkyNano/stage1.py extract-pilot InkyNano-RoutingGym-pilot.tar.gz work/routing-pilot
python InkyNano/stage1.py run --input work/routing-pilot --out work/stage1-teacher --device cpu --epochs 20 --batch 4 --threads 1 --seed 17
```

For the larger commodity-GPU dataset, pass the completed and manifested analysis directory of every capture shard. The runner checks the original demand arrays against the trace, rejects root/family duplicates across shards, and builds CPU-backed memory maps while reading one trace turn at a time. Use `--device cuda` when available GPU memory permits, and use a fresh output directory when changing settings. [Commodity capture and handoff](COMMODITY_GPU.md)

```sh
python InkyNano/stage1.py run --input shard00/analysis shard01/analysis --out work/stage1-commodity --device cpu --epochs 20 --batch 4 --threads 1 --seed 17
```

See the [CPU run](stage1_results/synthetic_seed17/run.json), [CUDA run](stage1_results/synthetic_seed17/cuda/run.json), [sweep summary](stage1_results/synthetic_seed17/summary.csv), [per-root CPU receipts](stage1_results/synthetic_seed17/receipts.json.gz), [per-root CUDA receipts](stage1_results/synthetic_seed17/cuda/receipts.json.gz), [checkpoint and receipt hashes](stage1_results/synthetic_seed17/manifest.json), and [Stage 2 specification](STAGE2.md).
