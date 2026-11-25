# Cloud audit receipts

Run [45564595785](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45564595785), audit commit `491e48abc52ea9804b0e8e6d42f8f2e13b144c77`, passed. Artifact `14999421761` SHA-256 `5a60190d3ac2d634b943724028fe44bde0cdc46cda5150ef7ff9b72f6e6a0823` was downloaded and independently hash-checked. It contains the generated Lean proof source, compile log, per-root classical replay and aggregate archive audit.

| Check | Actual result |
| --- | --- |
| Archived live episodes | 144 inspected, 127959 reported calls summed, 0 recorded mismatches |
| Archived closed paths | 28/28 replayed symbolically |
| Original cert source hashes | 13/13 match after LF→CRLF conversion (the original byte hashes were Windows-specific) |
| Deferred BFS | 24/24 closures, 96/96 archived counter matches |
| Native deferred proof compilation | 24/24 standalone declarations in Lean 4.19.0, zero axioms |
| Mean/median/max classical search | 0.551496 / 0.344065 / 2.567315 seconds |
| Batch compile of all deferred proofs | 7.457720 seconds |
| Amortized search plus batch compile | 0.862235 seconds per solved root (not per-root native latency) |
| Mean proof steps | 18 |
| Graded ground pilot | 100 systems, 4676 probes, 434 beyond-horizon negatives, zero truncated-distance mismatches |

Recorded primary live search closure by cumulative call threshold: 0% at 128, 18.0556% at 256, 38.8889% at 512. Recorded search + init (still excludes final certification): 1.3889% by 5s, 11.1111% by 10s, 30.5556% by 20s and 38.8889% by 30s. All other actually completed live arms have zero closures at their recorded stopping rules.

## Late registered graph-bridge comparison (independent count-mode implementation)

Reimplemented the exact registered geometric and graph bridge score/expansion formulas in the JavaScript sandbox with a cache of invariant pair features. Same 24 archived roots, B=512, depth=24. Geom MM reproduced the reported novel-call means exactly in each length stratum and overall, with 0 closures. Graph bridge also closed 0/24, charged work 512 on all roots; novel discovery calls 307.333 per root (279.25 / 306.50 / 336.25 at L=16/20/24). Source and reproducer: `audit_heuristics.mjs`.

Thus the missing graph-bridge comparison is consistent with the originally reported closure advantage when evaluated retrospectively in count mode. The original live registration deviation remains. This is not a new prospective claim; other learned count-mode arms and the missing live recurrent comparator are still incomplete.

H2/H3 are still confounded by stopping policy and verifier placement. The exact final-only congruence proof experiment establishes an available deferred baseline. Timing across Linux and the original Windows host is indicative only; compare all controllers on the same hardware before quoting speedup.

The original 127959 tactic calls were summed from archived records, not re-executed. Original live checkpoint outputs have not been independently reproduced here.
