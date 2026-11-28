# LD-MoE experiment ledger

Status: registered development proposal, no new results. Supersedes prompt-fixed residency as the primary path; keep previous negative results and receipts intact. Do not consume paid cloud or grant credits for the first two stages.

| ID | Stage | Required control / output |
|---|---|---|
| LDT-MOE-01 | Native exact offload | Unrestricted full-bank runtime vs identical packed precision reference, all 128 router rows, token logits/IDs/weights, memory and stalls |
| LDT-MOE-02 | Native routing, learned cache | Demand LRU and recency, prior pass, temporal/spatial predictor, matched-budget independent vs mesh, exact same selections |
| LDT-MOE-03 | Conditional substitution atlas | Native mixture, random resident substitutes, nearest router score, activation-conditioned output model; local error, weights and uncertainty |
| LDT-MOE-04 | Per-token residual router | ReMoE-style recency control, frozen teacher, zero-correction parity, parameter-matched independent/TRM/mesh, actual discrete transfers and native fallback |
| LDT-MOE-05 | Healing | Router-only vs joint expert/backbone adaptation, same train token and compute budget, untouched evaluation |
| LDT-MOE-06 | Mechanistic cross-pass study | Matched no-pass, shuffled messages, coordinate controls; predictor then causal interventions |

## 01 exact native baseline

Use published full expert bank and corresponding backbone from GGUF_CONVERSION.md, and current native runtime/demand-paging build from PROMPT_MATRIX.md. Freeze original physical ID mapping and same checkpoint. Full GGUF mmap/demand paging or CPU execution is a first reference; a scoped teacher is only a debug control. Check numerical parity with unrestricted reference on shared prefixes at every native visit and final logits, calibrated for kernel precision. Preserve exact native routing even when expert absent. Stop on missing expert, unresolved parity, invalid cache accounting or memory pressure.

On 3050 use batch 1, start short context (512 or lower), available host/GPU limits measured before launch, disk-backed bank, bounded staging. Do not assume whole 16-GiB bank fits 6 GB free host RAM. Native CPU offload may be slow; record it. Use existing 4 GiB physical limit, account for desktop usage, driver/library, KV and scratch. Compare unrestricted exact vs fixed random/prompt subsets using the same quantization, tokenizer, prompts, budget, sampling and scorer. Collect full answer outputs, not just throughput. Preserve separate capped timing pilot. Instrument native router to emit per visit original 128 scores and correction bias, top-k IDs/weights, token/pass/layer, causal sketches and timestamps. Hash model/roster/runtime. Trace runs and uninstrumented throughput runs separate.

## 02 exact scheduling

Use same offline traces for paired cache replay, then native on-policy implementation. Predict only from information available before the prefetch deadline; actual target scores can be used for demand fetch at the visit, not for earlier prefetch. Record target visit, prediction time, requested bytes, copy-completion time, actual hit, wasted bytes, evictions, CPU fallback and stalls. Count all transfers including cold fills. Same expert (l,e) across loop passes; permit distinct unequal-size costs between layers. Compare LRU against fixed budget next-use oracle as a hindsight ceiling for that specified cache model. Fixed capacity 8/16/32 per layer plus measured global byte budgets. Replay is observational; exact execution with same route is the parity gate. A missed prefetch never forces a replacement.

## 03 substitution feasibility

Collect current hidden states and native expert outputs plus candidate outputs (including selected, nonselected near threshold, resident candidates, random). Cover early, loop, late layers and pass IDs. Store compact projections only if their error to full output is calibrated; keep full vectors for a bounded pilot. Use training-only normalization. Group roots/families before splits. Compare fixed native mixture weights with explicit renormalized new mixture as different arms. Estimate error and calibrated upper quantile on new development roots; compare random/score-distance controls. Test on-policy distribution shift. If error cannot support a useful fallback rate, do not promote substitutions; retain exact cache controller.

## 04/05 promotion

Freeze a single primary contrast after 02/03 pilots. For 04 compare learned residual to exact native and simple recency at measured matched cost; start at zero, increase penalty/correction on development only. Report exact-fetch fallback, retained mass, local mixture and final logit errors, task accuracy, throughput and bytes. Use full bank as reference. For 05 first report available independent training data, trainable parameter count, optimizer memory, estimated compute and cost; request explicit approval before paid healing. Avoid same-root tuning/evaluation.

## Statistics and evidence

Unit: independent problem/source family, clustered variants, modes, tokens, passes and depths. Initial 16-root trace pilot and synthetic Stage1 are development evidence only. Provisional development target 128 train/64 dev independent source roots where feasible; final independent confirmation count determined from pilot variance and matched sham floor. No sealed test created here. Keep original ARC public evaluation set closed.

Primary exact-scheduling effect proposed: >=15% reduction in measured expert-weight bytes per accepted decode token at preserved native route parity versus strongest frozen causal baseline; measure stall/latency separately. Any offloaded CPU computation is charged in end-to-end time, and transfer reductions that worsen latency are not a deployment win. Approximate arm proposed task noninferiority margin -5 percentage points paired accuracy relative to exact packed reference; also require >=15% measured transfer reduction. These are scientific thresholds, not claimed results. Before confirmation calibrate resolution, matched shams and power per EvalMath, freeze comparison family, adjust secondary tests, and use family-level paired outcomes with appropriate uncertainty. Noninferiority requires interval bound above the registered negative margin; a nonsignificant difference is insufficient. Report failures and zero-output cases, truncation and exact task verifier. Keep n_units, n_cases, root/task partitions, all arms, observed effects, uncertainty, costs, provenance and decision. No pooling correlated visits into independent samples. Evidence classes: deterministic finite test, numerical parity, observational trace replay, native intervention, powered confirmation.
