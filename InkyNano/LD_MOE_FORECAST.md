# LDT-MOE-02 learned demand-eviction forecast

This is a trainable reference for the exact-route scheduler, not a replacement router. New ordered trace required; aggregated archived receipts are rejected.

A forecast is produced after observing the current visit, to decide which nonrequired resident experts to retain for the next visit of the same physical layer. All active native experts must execute. No prefetch is modeled. An admission is charged for every missing active expert. At k=8 there is no eviction choice; the forecast should exactly match all eviction baselines. At k>8 the eviction forecast can exploit extra slots. Later prediction before a transfer deadline needs an earlier feature boundary and explicit asynchronous runtime.

Features: current raw sigmoid/rank scores and active-ID mask; preceding same-layer visit; historical same-pass (control) or historical expected-next-pass selection; and latest already-observed block-end selections. Score vectors of scoped teachers scatter to original expert IDs; absent IDs are excluded from the training loss. No target visit scores or later-layer state enter features. Four six-layer blocks each have a small encoder/readout, with layer/pass embeddings. Local control receives repeated local history in place of block messages. Layer mesh has causal block messages plus same-pass history. Pass mesh changes the historical message to the most recently observed expected-next-pass state. Layer mesh may already contain earlier-pass evidence; this is a test of the explicit next-pass channel, not all recurrence information. All variants have identical parameter counts and training budget. This uses observed message summaries; a learned recurrent message encoder and per-token router correction remain future work.

The supervised target is the native binary membership of the next same-layer visit, so objective aligns with equal byte demand misses within a layer. Equal positive/negative class weight; train-only root-balanced sampling, no dev fitting. Fixed seed, sample caps, matched capacities, fixed cold-start demand-cache rules. Compare LRU, EMA, previous-pass and hindsight oracle. Report estimated bytes and hits by root and phase. Do not report actual H2D savings without native execution. Scope and teacher hash are checked. No task correctness intervention yet.

```sh
python InkyNano/ld_moe_forecast.py --routing matrix/ordered-traces/routing.jsonl \
  --out matrix/ld-moe-forecast --device cpu --epochs 8 --samples-per-case 128 \
  --capacities 8 12 16 24 32
```

Use separate train/dev root families; modes share roots. Fit only after exact native full-support capture passes the numerical checks. For a quick synthetic test run python -m unittest discover -s InkyNano -p 'test_ld_moe*.py'. This code has no measured teacher-trained result at commit time. Train/dev comparisons remain exploratory until a powered independent confirmation per LD_MOE_EXPERIMENTS.md.
