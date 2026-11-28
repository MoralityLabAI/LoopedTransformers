# MOLT-07: bounded observer maps for MMS

Development fixture following [MOLT-06](OBSERVABILITY_RESULTS.md). Native BPD
and model-training effects remain separate. No sealed evaluation. Use
an operator-provided hard cap for local research workloads.

## Sketch

At fixed cut r, chart E and observer O, retain the top-k right singular
vectors V_k and encode a displacement as c=V_k^T d. Decode as V_k c.
The exact first-order tail bound is ||O(d-V_k c)|| <= sigma_(k+1)||d-V_k c||.
The code checks this for every stack projection and measures the actual
nonlinear future-readout error to the full displacement. A finite-dose
rank-five sketch need not preserve correctness even at numerical rank five.

Compare full-stack directions, terminal-only directions, random orthonormal
directions and full restoration, across ranks 1..5 and doses 0.25,0.5,1.
Terminal ranks above its two observable directions include arbitrary null
completion directions; treat those as terminal-SVD controls, not as
identified terminal features. Record coefficient bytes, all paired outcomes,
actual errors, and the nonlinear difference after subtracting the linear term.
The model and clean source are privileged; this is a reconstruction test.

## Cache

The immutable map key binds checkpoint, frame, epoch, full fixed input,
full reached state, observer code/contract, cut, position, support and
maximum rank. Maps store a float64 basis, singular spectrum and binary header.
The exact byte charge includes the 32-byte digest key. At d=16,k=5 the
map requires 812 retained serialized bytes. Coefficients cost 40 bytes
rather than 128 per displacement, before record metadata. Amortization
must include the map, and distinct reached states may need distinct maps.

The cache has deterministic LRU eviction and clears on checkpoint or epoch
changes. A changed frame/state/observer/support makes the key different.
The host owns these bindings; a hash does not prove the observer is correct.
Decode validates shape, finiteness, orthogonality and spectrum ordering.
Map storage caps are not RAM caps: temporary SVD work, tensors, allocator
overhead, report rows and model weights are excluded. The process-wide
lifetime peak RSS is reported where supported, without per-arm attribution
or process-tree enforcement.

The fixed cache trace uses two epochs. Each epoch visits all roots forward
and then backward. Version changes leave numeric inputs unchanged to test
conservative invalidation, not actual premise repair. Compare 0,4,16,64 KiB
with identical request order and one CPU thread. No-cache is direct
construction without serialization; cache misses construct the same map.
Every replay output and verified decision must match the no-cache rank-five
reference. Count builds, hits, invalidations, evictions, payload high water,
wall and CPU time. Timings include hashing, construction or decoding,
projection and suffix execution; training and the fixture preparation are
outside the timed trace. Cap order is fixed; small-run timing comparisons
are descriptive and not a hardware speedup claim.

Four new model seeds (89,97,101,103), 512 updates, 16 length-8 and 16
length-12 shared development roots; crossed model/world dependence.
No selection using protected roots. Do not use the fixture as evidence of
population-level performance or native BPD efficacy. Confirmation would
require EvalMath planning, a scientific error/effect floor, independent
root populations and fresh frozen readout. No paid GPU allocation.

    python -m pytest -q tests/test_observable_memory.py
    python -m experiments.molt_goblin_v1.mms_observer --out /tmp/molt07.json

The workflow stores raw results in a short-lived Actions artifact.
