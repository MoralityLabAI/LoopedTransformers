# INK-ROUTE-02 coordinate-2027 routing

This follow-up starts at [INK-ROUTE-01 merge f511aff](https://github.com/MoralityLabAI/LoopedTransformers/pull/311). It calibrates one candidate causal endpoint on the native pretrained model. The [promotion protocol](PROTOCOL.md) gives the proposed estimand, meaningful threshold, power and sealed access rule.

## Historical audit

The original eight-root CUDA run had 96 paired rows, eight tensor archives, 16 restored rescues, zero top-token flips and maximum relative reconstruction error of (1.75\times10^{-7}). Recomputed mean pass-3 K zero-minus-sham β₃ was −0.098214. The receipt audit also checked both successful two-root smoke runs and the four failed CPU attempts.

Transformers 5.24.0 calls the same reader 12 times in each of passes 2 and 3. Its input combines completed block 2 and the previous traversal's final layer-17 state. Unweighted RMS normalization precedes scaled dot-product attention. The output has Q, K, V and MoE order and the decoder unbinds it in that order. Archived output-side reconstruction and query-side softmax agree. The pre-hook supplies a private block-3 copy for one visit. The rescue supplies the clean reader output in the ablated lane for the final token. Subsequent readers use the original bank. INK-ROUTE-02 additionally verifies both banks at every read across each traversal, while allowing the model to compute its next pass's state through the native forward. [Transformers source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py)

The source checkpoint is `thinkingmachines/Inkling-2-Mini@b85c3cfa615fc413cc9af3379f491b963f3b586a`. Its full 19,526,019,168-byte shard is SHA-256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`. The runner checks the tokenizer, config, index and full shard before model loading. The packed-expert backend streams selected experts through the installed Transformers converter. Text execution uses native eager attention, the recurrent decoder and logits. The vision tower is unused.

## Development evidence before the broader controls

The earlier local 60-document pass-3 K calibration used a public FineWeb train roster from `CC-MAIN-2024-51`, distinct from the eight historical prompts. This round reuses that exposed roster only for development and retains its frozen coordinate draws. The 2,880 rows include 16 reference shams and four placebos per root with two replays. The mean zero-minus-mean-sham K difference was **−0.027714**, with root SD 0.052089 and 45/60 decreases. The matched-sham 95%/95% floor was **0.220428**, above the 0.10 scientific minimum. Replay discrepancies were zero at the recorded precision. Maximum realized dose disagreement was 0.376%. The pinned EvalMath strict preflight returned **REDESIGN_OR_PILOT_ONLY**. This calibration supports no positive confirmation sample size and the sealed readout remains closed.

## Native smoke

Two fresh development prompts passed on the pinned RTX backend, with 208 rows across both passes and two replays. All eight rescues reproduced the clean message exactly. Worst relative reconstruction error was 1.669e-7 and the largest realized equal-dose discrepancy was 0.220%. Maximum rescue log-probability difference was 1.788e-7 nats. The run peaked at 4.01 GiB reserved CUDA memory and 8.65 GiB sampled process RSS. Available host RAM reached a minimum of 4.77 GiB. Full checkpoint verification and model load took 37.91 s and 18.76 s, respectively. The eight profiled forwards totaled 332.70 s under the shared workload. [Smoke receipt summary](receipts/native-smoke/summary.json)

The repeated full-prefill run finished its first public development root, with 104 paired rows, four tensor archives and no top-token flips. Its four forwards took 569.27 s. The run was stopped after that complete root when shared resource use made the original budget impractical. The interruption receipt records the last validated root, physical GPU headroom, and hashes for the local partial run. [Interruption receipt](receipts/interrupted-full-prefill/interruption.json)

## Pass-3 control extension on archived native banks

The pinned Transformers reader was replayed on CPU using the 60-root CUDA run's verified clean banks and query. The reader matched archived clean outputs within 1.558e-7 relative error. Its 3,120 rows include 120 isolated reader-output rescues, with maximum coefficient reconstruction error of 1.979e-7. This replay measures the immediate routing effect of intervening at the actual reader input. Downstream logits are available from the separate full-prefill receipts.

| Channel | Mean zero minus coordinate sham | 95%/95% sham floor | Roots whose sign depends on sham choice |
|---|---:|---:|---:|
| Q | −0.000268 | 0.006558 | 4/60 |
| K | −0.027714 | 0.220428 | 56/60 |
| V | −0.001249 | 0.006903 | 2/60 |
| MoE | +0.002305 | 0.099230 | 32/60 |

The descriptive root-bootstrap interval for the K difference was [−0.04157, −0.01575]. The mean K shift relative to clean was −0.05187 for zeroing, −0.02488 for the equal-dose random vector, −0.02819 for the equal-dose norm-preserving random vector, and −0.05189 for norm-preserving target removal. The nominal direction survives averaging the reference shams, but the magnitude falls below the mechanistic floor. The strict preflight repeated the development-only disposition. [Replayed reader summary](receipts/archived-pass3-reader/summary.json)

As a separate writer diagnostic, the mean RMS-normalized coordinate at layer 17 was +32.54, +30.36 and −2.43 across passes 1–3. Its mean squared-norm share was 46.0%, 40.1% and 0.76%, respectively. These are observations on the same 60 documents. [Exploratory writer and channel summaries](receipts/previous-pass3-k/pass3-channel-summary.json)

## Both-pass calibration on fresh native captures

The RTX batched native reader run completed 60 public development roots and 6,240 paired rows. It reproduced the clean reader messages exactly and passed all 240 isolated reader-output rescues. Maximum relative reconstruction error was 2.209e-7 and maximum realized equal-dose error was 0.388%. The four clean prefills totaled 252.98 s. Peak CUDA reservation was 3.86 GiB, peak sampled process RSS was 14.88 GiB and minimum sampled available RAM was 8.74 GiB.

| Reader pass | Channel | Mean zero minus coordinate sham | 95%/95% sham floor | Roots whose sign depends on sham choice |
|---|---|---:|---:|---:|
| 2 | Q | −0.000235 | 0.003005 | 4/60 |
| 2 | K | −0.008022 | 0.160749 | 58/60 |
| 2 | V | −0.000422 | 0.001868 | 2/60 |
| 2 | MoE | +0.001824 | 0.010358 | 15/60 |
| 3 | Q | −0.000257 | 0.007230 | 4/60 |
| 3 | K | −0.027918 | 0.218282 | 56/60 |
| 3 | V | −0.001231 | 0.007780 | 2/60 |
| 3 | MoE | +0.002342 | 0.101569 | 32/60 |

All endpoints had a declared resolution of 0.001. The primary descriptive 95% root-bootstrap interval was [−0.04198, −0.01586]. Pass-3 K shifted −0.05229 relative to clean after zeroing, −0.02496 after the random vector, −0.02829 after the norm-preserving random vector, and −0.05231 after norm-preserving target removal. The equal-dose matched-sham floor bounds much larger variation than the observed mean effect. The native batch calibration uses the same development documents as the archived run, with a different prefill batching scheme. [Batched native reader summary](receipts/batched-native-reader/summary.json)

EvalMath returned **REDESIGN_OR_PILOT_ONLY** for the both-pass calibration. The proposed confirmation remains unregistered because the development contrast falls below the matched-sham floor and there is no development-supported planning alternative above that floor. Sealed evaluations remain unopened. [Strict preflight output](receipts/batched-native-reader/preflight-strict.json)

## Development redesign with magnitude and K-query matching

Replayed the pinned native reader on CPU using the verified pass-3 banks and query from the same 60 public development documents. Each document was one statistical unit. The two-root reader check passed before expansion. The sealed partition remained unopened.

The coordinate control moved a non-target activation toward zero by the target's absolute magnitude, with the realized BF16 L2 dose within 1%. This permits a smaller coordinate to cross zero. We matched absolute pre-edit activation and signed K query loading oriented along the direction of ablation. Before measuring the reader outputs, we ranked eligible coordinates by the larger relative mismatch, then the sum of mismatches, then coordinate index. Each root contributed at most four coordinates. The main overlap check required both mismatches to be at most 10%. The 25% and 50% checks describe sensitivity to that caliper.

- **Single-coordinate overlap** was 0/60 roots at 10%, 0/60 at 25%, and 53/60 at 50%. The median best joint mismatch was 45.4%. The nearest-coordinate comparison averaged **−0.00564** for zero-minus-control β₃, with a descriptive 95% root-bootstrap interval of **[−0.01282, +0.00026]**. This comparison uses approximate matches. The strict single-coordinate comparison has no estimate.
- **Projection-matched vectors** kept coordinate 2027 fixed and matched the dose, the clean bank's projection onto the delta, and the K query's projection onto the delta. All 60 roots admitted a solution. The mean zero-minus-vector β₃ gap was **−0.0000181** and the largest absolute gap was **0.000490**. Every root fell within the existing **0.001** β₃ resolution. Maximum realized dose mismatch was 0.329%, maximum bank-projection mismatch was 0.118%, and maximum query-projection mismatch was 0.096%. Relative reader reconstruction error peaked at 2.08e-7.

For a clean bank `x`, K query `q`, and edit `δ`, the K score depends on the edited dot product `q · (x + δ)` and the edited bank norm. The norm is fixed by the dose and `x · δ`. The vector results show that non-target directions can reproduce the observed change in the routing coefficient by matching these quantities. The K reader message still differed from target ablation, with a median L2 difference of 51.93 in reader units. The evidence supports a projection-level account of the development β₃ effect. A matched comparison between individual coordinates remains unresolved. [Raw development receipts and summary](receipts/matched-controls/dev60/)

Run the two-root check and then the 60-root replay from the repository root using the pinned native environment.

```text
python -m research_gym.inkling_routing_02.matched_controls --receipts NATIVE_READER_RUN --public-roster research_gym/inkling_routing_02/receipts/fineweb-development-roster.json --count 2 --out match-smoke
python -m research_gym.inkling_routing_02.matched_controls --receipts NATIVE_READER_RUN --public-roster research_gym/inkling_routing_02/receipts/fineweb-development-roster.json --out match-dev60
```

## End-to-end K-only development check

At the final prompt token of the pass-3 reader, compare the native K message produced by coordinate-2027 ablation with the message produced by an equal-dose projection-matched vector that holds coordinate 2027 fixed. Both edits take place on a copy of the previous-pass block-3 bank. Supply clean Q, V, and MoE messages to the decoder at that position and keep earlier token messages clean. Include a clean duplicate lane and a clean-K rescue for each ablation. The bank checks cover the complete native bank at every later reader in that traversal.

Use two new authored prompt roots and replay each complete prefill twice. Measure total variation (TV) over the complete next-token distribution for the treated pair and each arm versus clean. Record top-token IDs and clean-reference log-probability changes. A root contributes one paired contrast; lanes and repeated forwards measure controls and replay stability. Require a realized dose and projection match within 1%, a β₃ gap at most 0.001, clean-K rescue TV at most 0.0001, and reader reconstruction error below 0.001.

Before any expansion, require TV at least 0.01 (one percentage point of probability mass) and at least ten times the maximum observed replay or rescue TV. Both initial roots must clear that floor in both replays to open the ten additional authored development roots. The threshold is a development criterion for this readout. The sealed evaluation remains closed.

```text
python -m research_gym.inkling_routing_02.k_only --snapshot PINNED_SNAPSHOT --out k-only-smoke
```

The native run writes raw rows, checkpoint and resource receipts, and hashed NPZ archives. Each archive contains the complete next-token log-probability vectors, the reader banks, raw and delivered reader messages, and the input IDs. A successful run writes its summary beside the output directory.

The `--device cpu` fallback uses the same pinned local checkpoint and software and requires 12 GiB of available host RAM before model load. The `--offload-vocabulary` CUDA fallback keeps the embedding lookup and output projection on CPU, uses a 1.85 GiB allocator budget by default and checks for a further 1 GiB of GPU reserve and 8 GiB of host headroom. Use the same placement for every arm and replay in a run. For a qualifying smoke run, pass its directory as `--smoke-receipts` to select the additional development roots and carry forward the smoke TV floor.

### Two-root native result

The successful CUDA retry ran four full prefills on the pinned checkpoint and produced 24 paired rows. Both roots gave a replay-stable distribution change when the treated K message was switched between ablation and projection-matched vector. All conditions retained the clean top token.

| Development root | Ablation vs vector TV | Ablation vs clean TV | Vector vs clean TV | Absolute K β₃ gap |
|---|---:|---:|---:|---:|
| Weather balloon | 0.006181 | 0.007818 | 0.005778 | 0.00007255 |
| Bakery | 0.003374 | 0.002380 | 0.003775 | 0.00011509 |

At these roots the K message swap changed next-token probabilities despite close β₃ matching. The two-root mean pairwise TV was 0.004777. Maximum replay or clean-K rescue TV was 2.86e-8 and maximum reconstruction error was 1.39e-7. Realized dose, bank-projection and query-projection disagreements were at most 0.0152%, 0.0188% and 0.0548%, respectively. The four tensor archives independently reproduce the row-level TV measurements and the K-only delivery checks. Both roots fell below the 0.01 expansion threshold, so the additional development roots remained unrun. Sealed evaluations remained unopened. [Native receipt summary](receipts/k-only-native/summary.json)

### Exact β₃ clamp

Rerun the same two development roots with one common FP32 β₃ for both treated K messages in each prefill. Use the midpoint of the native K weights for the ablation and projection-matched vector, then mix each arm's native normalized banks using that shared coefficient and its complement. Keep the same realized-dose and projection checks. Include both unclamped K messages in the same batch for a placement-matched comparison. The clean replay, clean-K rescues, Q/V/MoE delivery and bank restoration contracts still apply.

The output-side reconstruction independently recovers β₃ for each delivered K message. Require an absolute β₃ error and relative message reconstruction error at most `2e-7`. Report the raw and recovered β₃ gaps separately. Keep the expansion threshold fixed at 0.01 TV and require replay noise low enough to resolve it.

```text
python -m research_gym.inkling_routing_02.k_only --snapshot PINNED_SNAPSHOT --clamp-beta --out k-only-clamp
```

Both development roots retained a measurable next-token difference with shared β₃. The two replays within each placement agreed. The first run used CUDA with CPU vocabulary offload because of GPU headroom. A repeat with the original all-CUDA placement reproduced the effect and agreed with the offload run within `9.0e-8` TV for the clamped pair. The table reports the all-CUDA results.

| Development root | Shared β₃ | Unclamped pair TV, same batch | Clamped pair TV |
|---|---:|---:|---:|
| Weather balloon | 0.93985677 | 0.00175818 | 0.00144736 |
| Bakery | 0.92959690 | 0.00236124 | 0.02520405 |

The largest independently recovered gap between the two delivered β₃ values was `1.22e-8`. The clamped K messages reconstructed within `4.17e-8` relative error. All clean-K rescues passed, with a maximum TV of `6.56e-8` across placements, and every condition retained the clean top token. At the bakery root, clamping changed the delivered ablation K message by 0.00129 in L2 and increased the ablation-versus-vector TV by about tenfold relative to the same-batch unclamped pair. The result measures sensitivity at the shared midpoint. The downstream source of the amplification remains unresolved.

Only the bakery root cleared 0.01, so the two-root expansion rule stayed closed and sealed evaluations remained unopened. Both placements contain 32 paired rows and four hashed local tensor archives each. [All-CUDA receipts](receipts/k-only-clamp-cuda/) · [Vocabulary-offload receipts](receipts/k-only-clamp-offload/)

### Downstream trace and expert swap

The first large bakery divergence appears in the final-token MoE at Hugging Face layer **14** (zero-based) on pass 3. The clamp moved the **vector** lane across the top-eight router cutoff. Expert 66 replaced expert 60. Expert 66 led expert 60 by 0.00005868 in the clamped lane and trailed by 0.00112614 in the unclamped lane. The vector lane's clamp–unclamped MLP output L2 difference rose from **0.2131 at layer 13** to **18.1949 at layer 14**. At layer 16, expert 69 subsequently replaced expert 96. The bakery ablation lane kept its expert sets throughout the trace.

We exchanged only the vector lanes' final-token expert sets at layer 14 and recomputed the routing weights from each recipient's own logits. Restoring the unclamped set in the clamped vector lane reduced the original ablation-versus-vector TV from **0.02520 to 0.01201**, a 52.4% reduction. The restored lane was 0.00788 TV from the original unclamped vector. The reverse swap moved the unclamped vector lane 0.02540 TV from its original output and to within 0.00348 TV of the original clamped vector. The layer-16 expert switch reversed with the layer-14 swap. The remaining 0.01201 TV means the single router intervention only partially recovers the clamped comparison.

The balloon vector had a visible difference at the layer-12 key projection, followed by identical final-token attention outputs and identical expert sets through the tail. Its layer-14 swap was a no-op and its output contrast stayed at 0.00145 TV. All unswapped lanes reproduced their baseline distributions exactly on both roots. The two replays agreed, all conditions retained the clean top token, and both roots remained development-only. Sealed evaluations remained unopened.

```text
python -m research_gym.inkling_routing_02.k_trace --snapshot PINNED_SNAPSHOT --out k-trace
python -m research_gym.inkling_routing_02.k_trace --snapshot PINNED_SNAPSHOT --swap-layer 14 --swap-arm vector --out k-swap
python -m research_gym.inkling_routing_02.k_trace_report --discovery k-trace --swap k-swap --out k-trace-summary.json
```

The discovery and swap receipts preserve raw router scores and stage measurements, paired output rows, checkpoint and resource metadata, and hashes for the local tensor archives. [Discovery receipts](receipts/k-clamp-downstream-trace/) · [Swap comparison and receipts](receipts/k-clamp-downstream-swap/)

### Layer-14 and layer-16 fixed-set factorial

The factorial reads the two expert sets per layer and root from the hashed, unmodified all-CUDA discovery archives. Both discovery replays must agree on the selected experts. In each of four conditions the clamped vector lane receives either the recorded clamped or the recorded unclamped expert set at each router. The emitted weights come from that lane's current logits and the chosen expert indices. An unmodified replay and all four conditions use the same eight-lane batch, exact shared β₃ and two forwards per root. The analysis checks the archived K delivery, the clean and ablation lanes, and the realized expert membership.

The outcome is next-token TV from the clamped ablation lane. At each layer-14 setting, take the signed TV difference between the two layer-16 settings. Their difference measures interaction on this TV scale. Compare restoring both layers with the existing 0.01 TV threshold. The two roots remain development-only.

```text
python -m research_gym.inkling_routing_02.k_factorial --snapshot PINNED_SNAPSHOT --discovery k-trace --out k-expert-factorial
```

| HF layer 14 set | HF layer 16 set | Bakery TV | Balloon TV |
|---|---|---:|---:|
| Recorded clamped | Recorded clamped | 0.02520405 | 0.00144736 |
| Recorded clamped | Recorded unclamped | 0.02117288 | 0.00144736 |
| Recorded unclamped | Recorded clamped | 0.01421022 | 0.00144736 |
| Recorded unclamped | Recorded unclamped | 0.01200900 | 0.00144736 |

Restoring layer 16 lowered bakery TV by **0.00403117** with layer 14 set to clamped and by **0.00220122** with layer 14 set to unclamped. The signed difference-in-differences was **+0.00182995**. The layer-16 restoration had a smaller effect after layer 14 was restored. The **0.01200900** joint restoration remained above **0.01** and reproduced the earlier layer-14-only restoration, in which the downstream router had selected the unclamped set naturally. The balloon expert sets were identical across the two recorded states at both layers and all four outcomes agreed.

The unmodified forwards reproduced the archive exactly, as did all untargeted output lanes. Both replays agreed on all outcomes. The maximum recovered β₃ gap was 1.22e-8, maximum clean rescue TV was 4.56e-8, and all conditions retained the clean top token. The 160 rows cover twenty native forwards across two roots. An offline check of all 32 forced routing calls reconstructed their emitted weights from the archived recipient logits within 1.84e-7 absolute error. Sealed evaluations remained unopened. [Factorial summaries and receipts](receipts/k-expert-factorial/)

## Reproduction

Use Python 3.12.10, PyTorch 2.11.0+cu128, CUDA 12.8, Transformers 5.24.0, Accelerate 1.16.0, NumPy 2.5.5, psutil 7.2.2, safetensors 0.8.0 and tokenizers 0.23.3. The CUDA runner checks these package pins. Before allocation the runner checks host memory, CUDA memory and physical NVIDIA headroom. The per-process allocator is capped at 5 GiB by default. Existing workloads continue running. Use a fresh output directory for every attempt.

```text
python -m unittest discover -s research_gym/inkling_routing_01 -p "test_*.py" -v
python -m unittest discover -s research_gym/inkling_routing_02 -p "test_*.py" -v
python -m research_gym.inkling_routing_02.receipts audit --history PREVIOUS_OUTPUTS --out historical-audit.json
python -m research_gym.inkling_routing_02.native --snapshot PINNED_SNAPSHOT --out smoke
python -m research_gym.inkling_routing_02.receipts analyze --receipts smoke --out smoke-analysis
python -m research_gym.inkling_routing_02.native --snapshot PINNED_SNAPSHOT --public-roster research_gym/inkling_routing_02/receipts/fineweb-development-roster.json --out development
python -m research_gym.inkling_routing_02.receipts analyze --receipts development --public-roster research_gym/inkling_routing_02/receipts/fineweb-development-roster.json --preflight EVALMATH/skills/evalmath-mechinterp-power/scripts/preflight.py --out development-analysis
python -m research_gym.inkling_routing_02.receipts bundle --receipts development --out development-bundle
python -m research_gym.inkling_routing_02.reader_calibration --snapshot PINNED_SNAPSHOT --public-roster research_gym/inkling_routing_02/receipts/fineweb-development-roster.json --batch-size 20 --cuda-budget-gib 4.2 --out reader-development
python -m research_gym.inkling_routing_02.reader_calibration --snapshot PINNED_SNAPSHOT --public-roster research_gym/inkling_routing_02/receipts/fineweb-development-roster.json --archived-pass3 PREVIOUS_OUTPUTS/ink-route-pass3-k-native60 --out archived-pass3-replay
```

The first native command runs two fresh smoke prompts. Estimate no scientific effect from those prompts. Validate both roots before running any public roster shard, and exclude the smoke prompts from all development floor and power estimates. For recovery use `--start OFFSET --count COUNT` with the original roster and identical software and placement. Pass nonoverlapping successful receipt directories to `analyze`. The raw paired rows contain per-forward timing, realized doses, final-token outcomes and tensor summaries. Local NPZ archives preserve the bank, query, reader output, delivered message and writer vectors. Each run emits resource snapshots, checkpoint metadata, stage failures and SHA-256 hashes. The offline CI jobs exercise synthetic contracts. They do not run the checkpoint.

The GitHub bundles contain gzip-compressed raw JSONL, the original run's SHA-256 manifest, and the metadata receipts. Verify decompressed JSONL against the original manifest. Tensor archives are retained locally and listed by hash in the run manifest.

The batched reader mode first validates two public development roots, then captures clean banks for the remaining roots in batches. It runs the same native Transformers reader on the perturbed last-token banks. Analyze either reader mode with the same `receipts analyze` command and roster. Keep these routing-only rows distinct from the full-prefill rows when reporting downstream effects.
