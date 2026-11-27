# CPR-01: coded permutation routing

Source motivation: Yann Tal, [Sorting in o(N log N) Time on a Multitape Turing Machine](https://eccc.weizmann.ac.il/report/2026/331/), revision 1, 2026-11-25. This gym deliberately studies separate communication models. It does not implement or validate Tal's tape-machine algorithm.

## Exact finite domain

A line holds one record per site. Each round permits any matching of adjacent swaps. Each record has a unique destination tag, and the payload is opaque to the policy. A swap charges two width-bit payload transfers plus two ceil(log2 n)-bit destination tags. Each round counts even if empty; comparison counts are recorded separately. This is an abstract model, not a latency model or a count of tape steps. Metadata computation, local reads and synchronization are not included in transmitted-bit totals.

BFS over the full permutation graph (all nonempty adjacent matchings, including nonmonotone moves) gives minimum parallel rounds from every state to identity, since moves are involutions. We enumerate every permutation of size 2 through 7. Odd-even transposition and left-to-right greedy inversion matching must exactly reconstruct all payloads and meet the inversion-count lower bound on adjacent swaps. Each adjacent swap changes inversion count by one; an inversion-only algorithm that reaches identity attains that bound. Stable duplicate-key preprocessing is checked on 243 ternary key lists of length five. The rank assignment is externally supplied and its cost is not included.

## Coding fixture

Separate topology: two endpoints holding independent w-bit messages exchange via a relay. With two w-bit unicast uplinks and a shared broadcast downlink, XOR broadcast costs w bits versus 2w uncoded downlink bits; total transmitted bits 3w versus 4w. If downlinks are point-to-point, both arms cost 4w total. The fixture exhausts all message pairs for widths 1..6 and toggles one bit of recipient side information. Each receiver must recover the other message, requiring at least w bits of shared broadcast when its own message is known. No free cross-topology comparison is made.

## Learned pilot

A six-weight antisymmetric logistic comparator learns from binary rank pairs below 16 (three deterministic training seeds, 24,000 proposals per seed). It is reused across odd-even rounds and can see only the two destination tags of its scheduled edge. A hard swap primitive moves payload and tags exactly. No global attention, learned compression, BPD decomposition or native transformer weights are involved in this arm. All fit and evaluation seeds are fixed in source. Evaluation is on 64 independently generated permutations per seed and size 8, 16, 32, 64; these are opened exploratory roots, not sealed confirmation. At 32/64 the extra binary features never appeared in training. The exact integer comparator is a structural ceiling; it supplies no evidence of learned routing beyond the local comparator. Keep incorrect and unsolved cases. Payload and metadata traffic are charged for both arms; learned computation, tag preprocessing and comparison traffic are separate costs. Raw outcomes include the permutations and paired trajectories' aggregate costs.

Evidence gate: exact finite results for the enumerations, exploratory observational only for trained arms. Unit for later population claims: independent source permutation family and independently trained checkpoint, not loop depth or pair comparisons. No mechanistic causal claim, no calibrated sham floor, no power claim. The numerical comparison family here is descriptive, with 12 seed-size cells; no p-values. Scientific target for follow-up: at least 5 percentage points of exact reconstruction improvement over a parameter- and communication-matched control under a fixed bit-flip intervention, with sham-calibrated floor and independent heldout families following EvalMath FSI-01. Do not reuse pilot roots as sealed data. A native tied attention block with local two-token masking and a genuinely rank-matched BPD adapter is a subsequent arm. Include metadata, codebook, route-control and read/write traffic for any claim of general communication savings.

## Replay

Node 22, no dependencies: `node research/coded_permutation_routing/cpr01.js summary.json raw.json`. Compare summary against committed receipt. CI uploads raw paired outcomes. No GPU is required.
