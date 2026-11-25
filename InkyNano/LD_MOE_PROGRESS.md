# LD-MoE continuation: ordered native trace and exact demand-cache replay

## Actual archived evidence audit

The current PROMPT_MATRIX pilot captures only phase-integrated demand vectors, not ordered top-8 selections. Its 8 captured requests are 4 roots (one train and one dev per task, two modes each), each with seven evaluated decode tokens, on a teacher composed from the first 12 experts of each physical layer. Native decode H0 hindsight top-8 mass computed from the archived aggregate vectors (average over the two modes of each root):

| Root | Split | Scoped top-8 mass |
|---|---|---:|
| arc-agi-2-training/6fa7a44f | train | 0.93994344 |
| sudoku/1700000 | train | 0.95486154 |
| arc-agi-2-training/2de01db2 | dev | 0.95181177 |
| sudoku/1700012 | dev | 0.94745972 |

Dev root mean 0.94963575. Within the same 12-expert support, top-12 mass is 1 by construction. Source: prompt_matrix_receipts/pilot/capture/routing.jsonl, composition.json and cases.jsonl at 53fee1d46079c830264fd2c4192be97510428c1d. This is a fixed-set oracle on phase totals, neither temporal hit rate nor original teacher coverage. The full-answer matrix reports 0/8 Sudoku, 0/8 ARC and 0/8 Storyworld correct/valid, including the scoped teacher; high relative mass within a poor scoped teacher is insufficient. No per-token cache metric can be recovered from these phase totals. See PILOT_REPORT.md.

## New ordered trace

The native runtime now has opt-in trace-events/trace-events-ram modes, with one token per forward, context <=1024. It records ordered original graph slots, raw unbiased sigmoid scores, bias-adjusted rankings, selected indices/raw weights, and final scaled mixture weights. Based on pinned llama.cpp a6ac41c757c066941766164439f6876f2deaab04 src/llama-graph.cpp and src/models/inkling2{,-loop}.cpp. The wrapper maps scoped local IDs to original IDs, preserves the score coordinate support, checks complete native visit coverage, top-k and scaled weights, and binds model/runtime identity. Last sampled but unevaluated output token has no router event. This mode deliberately alters prefill batching; use uninstrumented matched runs for timing and verify incremental-vs-batched quality separately. The native C++ changes have not yet been compiled or exercised here.

```sh
python InkyNano/matrix_run.py capture --ordered-events --batch-tokens 1 \
  --cases matrix/cases.jsonl --model "$FULL_BANK_GGUF" --native "$NATIVE" \
  --runtime-revision "$RUNTIME_REV" --context 512 --max-new 16 --limit 2 \
  --threads 4 --rss-limit-bytes 3221225472 --out matrix/ordered-smoke
```

For a scoped debug run provide --published-composition with the exact model receipt. For unrestricted support use the full 128-expert model. Do not force eager loading of full bank. Set --max-new/--limit for bounded smoke only; later capture independent train/dev roots at meaningful budgets. A full bank with actual quality parity remains the gate.

## Exact cache replay

ld_moe_cache.py evaluates native demand-only scheduling with LRU, exponential frequency, previous-pass evidence and farthest-next-use hindsight. For fixed per-layer capacity and uniform bytes within each physical layer, the latter is optimal for minimizing demand admissions under the mandatory co-resident top-k visit rule (exchange the evicted nonrequired expert whose next use is farther away; an exhaustive small-DP test checks 243 sequences). All native active IDs are pinned for each visit; a miss forces a demand admission. No prefetch or CPU bypass is modeled. At k=8 all policies have identical residency after each visit, so eviction learning only has headroom for k>8 or a separate prefetch/CPU execution model. Byte estimates use the published mixed bank's 4,810,752/7,174,656 bytes per expert by physical layer. Include prefill, cold starts and recurrent reuse. These are packed payload estimates, not measured H2D, stalls or disk reads.

```sh
python InkyNano/ld_moe_cache.py --routing matrix/ordered-smoke/routing.jsonl \
  --out matrix/cache-smoke.json --capacities 8 12 16 24 32
```

Requires dev cases in the input; capacities beyond scoped support skipped. Reports per-case and root-macro results. No native ordered capture in this session; new replay tests are synthetic finite checks. Actual full-bank parity and performance, learned forecasts, backend scheduling and substitution still pending.
