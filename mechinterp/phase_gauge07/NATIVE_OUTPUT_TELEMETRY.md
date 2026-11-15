# COE-07 A: Native pretrained output-oracle replay

**Evidence type:** reanalysis of existing **pretrained** LoopScope-02 output traces. The source is the earlier `gaoxin492/step_21700` (SHA-256 `932b5e0d...`) native TRM run, with all 256 puzzle target-cell correctness histories saved across 16 ACT steps. This is a **truth-conditioned correctness indicator** and has access to ground truth. It is not a model-observable latent probe, a learned readout, or a new inference run.

Source files and locked Git blob IDs:

- `mechinterp/loopscope02/results/benchmark_bundle.json` `01ec63b4b2a4ae5399f783c6036a7fc018825a87`
- `mechinterp/loopscope02/results/trm_results.json` `3c35daeb71da32f5c9ebbf03da539b8ee718b0ee`
- Both report the same internal benchmark digest `59459ae0bb60b75db5be017e5f41adead391f2abb0b8fb043cd331c0ee043811`.

Read the first T binary correctness observations for each board; assign bit t to outer visit t+1. Within each of the original 64-item Sudoku strata, compare every unordered pair with distinct target digits. Report retained distinctions under ordered words, canonical cyclic-orbit words and popcount-only words; `final` reports only the correctness of the final visit.

## Eight-visit results on full original 64-item strata

| Stratum | Eligible cross-label pairs | Ordered | Cyclic-invariant | Popcount | Final step |
|---|---:|---:|---:|---:|---:|
| S1_naked1 | 1,811 | 0 | 0 | 0 | 0 |
| S2_prop2 | 1,788 | 0 | 0 | 0 | 0 |
| S3_prop3plus | 1,789 | 446 | 446 | 446 | 0 |
| S4_stalled | 1,795 | **1,414** | **1,389** | **1,369** | **738** |

For S4, cyclic structure contains 20 extra distinctions beyond count, while a known starting phase contains 25 more. Every S1/S2 item is correct at all eight steps, collapsing the truth-conditioned histories to the same all-ones code. These results are expected from an oracle that conditions on correctness; they cannot establish that a real quantized latent probe can recover the extra distinctions.

The **preregistered one-board-per-digit** heldout selection gives S2 `0/36` and S4 `28/36` distinctions for all three (ordered, cyclic, count) at T=8. The full S4 stratum exposes the rare phase-invariant distinctions that the small first-nine selection misses. The full-stratum result is **post-hoc diagnostic** and does not satisfy COE-07's preregistered native activation-probe success gate.

At `T=4` on all 64 S4 boards, (ordered,cyclic,count,final) = (1278,1244,1244,802)/1795. At `T=16` the values are (1447,1428,1422,685)/1795, giving 6 cyclic-above-count distinctions and 19 phase-sensitive distinctions. A longer observation horizon can reduce differences in the final correctness state as more puzzles converge, while history retains earlier distinctions.

An illustrative `T=8` pair is S4 item 05 versus item 25, binary histories `11111000` and `01001111`. Both contain five correct steps, and their necklaces differ, so the cyclic-invariant observation distinguishes them even with an unknown starting visit.

## Reproduce

```
python -m mechinterp.phase_gauge07.telemetry_replay --out /tmp/coe07_output_oracle.json
```

`telemetry_replay.py` verifies both source Git blob hashes, the original benchmark digest, the model SHA-256, and the complete 16×256 binary array before computing the result. Because it reads data already committed to this repository, the replay requires neither the checkpoint download nor a GPU. It can run with the Python standard library except that the shared `score.py` import currently needs NumPy. The full native activation extraction in `run_native.py` remains independently gated behind checkpoint download and hosted runner availability.
