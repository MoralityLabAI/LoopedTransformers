# OLR-06E: shared native rewrite frontier results

**Status:** completed native engineering; registered neural-scheduling headroom gate FAIL.
**Registration:** [06E](OLR_06E_REGISTRATION.md).
**Executed commit:** 040a58aee13cc48864a4feb2733e747112e641b0.
**Native workflow:** [45321437290](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45321437290), success.
**Artifact:** [14888520606](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45321437290/artifacts/14888520606).

All four shared-frontier/prefix-budget protocol tests and the native workflow passed. Twenty-four fresh equation systems, 96 target obligations, 144 frozen bounded trials and 24 unrestricted references. The 252 older alpha lineages were excluded. There were 280 distinct live leaf-path replays and 280 independent leaf certificates. The standalone conjunction compiler accepted 80 distinct conjunction declarations, giving 360 axiom-free declarations total. These are not 80 independent roots. All 24 unrestricted references closed.

The proposal ledger merges public rewrite graph components and recovers bounded paths. It does not assign native proof authority to an unverified edge. Successful leaf proofs are compiled independently, then reused in explicit conjunction terms.

## Complete conjunctions

| Policy | DEV 128 | DEV 256 | DEV 512 | Transfer 128 | Transfer 256 | Transfer 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Independent | 0/12 | 7/12 | 11/12 | 1/12 | 1/12 | 6/12 |
| Cached independent | 1/12 | 8/12 | 12/12 | 1/12 | 2/12 | 7/12 |
| Shared reverse BFS | 0/12 | 1/12 | 2/12 | 1/12 | 1/12 | 2/12 |
| Balanced frontier ledger | 5/12 | 9/12 | 12/12 | 2/12 | 4/12 | 7/12 |
| Demand frontier ledger | 6/12 | 9/12 | 12/12 | 2/12 | 4/12 | 8/12 |
| Shared PDB | 9/12 | 12/12 | 12/12 | 7/12 | 10/12 | 11/12 |

At the primary 256-concrete-expansion cap, the strongest registered same-information policy completes 22/24 (91.67%). This exceeds the allowed 80% closure threshold. The registered headroom gate is false. Do not fit a neural frontier scheduler under this endpoint.

## Resource accounting

| Policy | DEV mean concrete expansions | Transfer mean concrete expansions | DEV mean planner ms | Transfer mean planner ms |
| --- | ---: | ---: | ---: | ---: |
| Independent | 276.17 | 419.25 | 3.87 | 6.55 |
| Cached independent | 231.67 | 371.08 | 3.44 | 6.32 |
| Reverse | 474.75 | 474.50 | 6.29 | 7.12 |
| Balanced | 164.42 | 348.17 | 9.38 | 26.78 |
| Demand | 170.42 | 343.25 | 9.59 | 25.80 |
| Shared PDB | 73.17 | 158.83 | 6.94 | 13.19 |
| Unrestricted independent reference | 276.33 | 540.17 | 4.16 | 9.01 |

Rows average actual stopping effort (including capped misses), not exclusively completed roots. Timings are hosted planner timings excluding native certification, not a matched latency claim. The shared PDB includes mean setup of 459.17 and 642.50 abstract expansions. With the illustrative combined budget of 512 abstract+concrete expansions, PDB closes only 6/12 DEV and 0/12 transfer. That is prefix accounting on completed paths, not an independent run with interleaved compute. Native leaf execution/replay took 35.05 s in total; campaign time 41.99 s.

The useful engineering outcome is shared exact graph and proof reuse. Concrete work is reduced, yet naive component merging and repeated graph traversal cost time. An exact adjacency cache alone improves reuse with low overhead. The PDB has strong concrete-expansion performance but expensive up-front abstraction. This suggests a different resource question, registered separately in 06F: allocate total computation between partially constructed abstractions and concrete search. No neural promotion or revision of the failed 06E gate.

## Artifact custody

The downloaded archive's SHA256 is 040dea72c9220007342556bac02e17ab5b6392f4fa6f02ee2cdaad40e49019fd. Independently verified every file hash in both manifests (3 registry + 8 evaluation entries).
- Registry manifest: dd50ca18548b334c8b943b0122ece1305c3d223f7ff4f5f96b79c1f8d74e83e8
- Evaluation manifest: 51e31b239847b784b78131b8fe3ae6a16c0bfe21321c40cb8645a6a3e222b748
