# LSA-02M: native residual rewrite ordering, developmental result

2026-11-24. Native engineering gates PASS; registered developmental gate FAIL. [PR #338](https://github.com/MoralityLabAI/LoopedTransformers/pull/338), evaluated commit 7202d3c29b9caffb8874eacbfd45bc6bbb471845. [Native CPU run 45285605889](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45285605889), [artifact 14872625569](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45285605889/artifacts/14872625569). Three compatibility workflows passed. Numbers transcribed from the actual run manifest.

The 64 original training systems yielded 482 adjacent reference pairs and 512 additional off-path pairs (256 groups), supported by 760 additional native Lean 4.19.0 proofs, replayed and independently compiled. There were 128 shortest-path Hamming detour states and 66 local Hamming minima in the selected groups; maximum shortest-path escape depth was 11. These are sampled training-state statistics, not prevalence estimates. All matched fit duplication checks passed. The new 32 development systems were lineage-disjoint from the earlier 02K and 02L rosters. Evaluation independently certified 205 distinct paths. Original ordinary/OOD contents stayed sealed.

| Search arm | 16 expansions | 64 | 256 | 1024 |
|---|---:|---:|---:|---:|
| Hamming greedy | 10/32 | 26/32 | 29/32 | 32/32 |
| Pair-projection greedy | 11/32 | 28/32 | 32/32 | 32/32 |
| Local greedy | 10/32 | 22/32 | 26/32 | 30/32 |
| Pair-projection A* | 8/32 | 13/32 | 23/32 | 27/32 |
| Bidirectional BFS | 7/32 | 14/32 | 24/32 | 32/32 |
| Four-update residual | 30/64 | 54/64 | 59/64 | 62/64 |
| Single-update residual | 34/64 | 55/64 | 59/64 | 60/64 |
| Reference-only residual | 27/64 | 55/64 | 61/64 | 61/64 |
| Shuffled residual | 22/64 | 43/64 | 53/64 | 59/64 |

The 64 episodes in each learned arm repeat the same 32 roots across two seeds. At 64 expanded words, the four-update residual exceeds the shuffled residual by 17.1875 pp (seed differences +18.75 and +15.625 pp). It trails pair greedy by 3.125 pp, with root-paired max-public bootstrap interval [-15.625, +9.375] pp. It trails both one-step and reference-only by 1.5625 pp. Gate FAIL: the registered public-control superiority and interval conditions fail. The learned/shuffled signal is developmental, with no sealed confirmation or recurrence/off-path attribution.

At 64 expansions, Hamming greedy uses mean 34.969 expansions, 226.844 edges and 1.111 ms planner wall time. Pair greedy uses 31.656 expansions, 198.563 edges and 2.099 ms (including setup). Four-update residual uses 24.688 expansions, 168.531 edges and 13.435 ms; one-step uses 22.344 expansions, 160.141 edges and 8.682 ms. Thus fewer expansions do not establish a wall-clock advantage. In the hardest 10–11 rewrite band at this cap, pair greedy closes 7/10; learned four-update 12/20, single-update and reference-only each 14/20, sham 6/20, Hamming 4/10. Pair greedy closes all 32 roots by 256 expansions and uses mean 37.875 expansions at that cap. This changes the useful target from raw recurrence to local projection geometry, amortized evaluation, and hard-cohort headroom.

Provenance: new development manifest SHA-256 be142147c008a06e55bb01967a90cf27b27b128984d5f4d59cae147b64bb69b3; certified pairs SHA-256 24ade5f3b9830c2539f60397003a2ec3bc30dd0b39ee1c4d0d7ff59d18f4b4a5; root results SHA-256 e3c3a497d9338f6cabfe795317130fb29657a92460b1f98f2de4b7fb13d17bdc. Exact unrestricted BFS 32/32. No scientific or architecture-promotion eligibility.

Next: establish a fresh difficult-cohort frontier using overlapping triple projections and a cheap additive learned correction to the stronger pair prior, with matched shuffled supervision. A table-compiled local model can avoid recurrent inference at search time. Test the geometry and amortized search cost before spending additional capacity on recurrence.
