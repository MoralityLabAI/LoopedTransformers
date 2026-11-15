# AF-19 local evidence inventory

**Scientific verdict: FAIL. Local independent audit: PASS.** Source commit 6ccae99a2336f725b87889229d72e04acca95116 and frozen PROTOCOL.json SHA-256 2a92d76a6f71a09c3ce674755dc819f93c1a9e9a0957f1111178c804889937d7.

The complete local archive AF19_local_evidence_20261115.zip is 1,708,244 bytes with **79 files**, including source code, research reports, all 16 frozen Torch checkpoints and matching named NumPy exports, full proof-action traces, independent evaluation files, and source-bound audits. The embedded manifest.json independently binds each original member's byte count and SHA-256.

- Archive SHA-256: d057fd834920b9b54a524d5d26b84092d5ce2a71d87a95258ff3faf8e44c3581
- Independent aggregate results.json SHA-256: dfd16656e6fa0d367ab296c0e4472bb74662d945ae34fbb32da394510de7f19d
- Host episodes: 11,648; Torch forward replays: 225; independent BFS cost checks: 328; paired causal/mask replays: 48
- Accepted false Horn proofs: zero
- Local tests: 10 regular and 10 optimized, both passing

This page is a GitHub-hosted inventory, while the binary archive and complete per-member JSON manifest are available through the AF-19 research conversation downloads. **No GitHub-hosted binary release is claimed.** Hosted Python 3.12/Torch 2.5.1 and Lean 4.19.0 verification remain outstanding. AF-19 PR #100 stays draft and unmerged against the frozen AF-18 branch.
