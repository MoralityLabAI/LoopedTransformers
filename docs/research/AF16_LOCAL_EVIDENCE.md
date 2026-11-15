# AF-16 source-pinned local evidence inventory

**Frozen scientific commit:** `bd43ad9765d9a78c32b5ee670968ac9762ede7ca`.

**Scientific verdict: FAIL.** Local independent audit: **PASS**. Separate hosted Python 3.12/Torch 2.5.1/Lean 4.19.0 checks remain blocked before runner execution.

- Exact local archive: `AF16_local_evidence_20261115.zip`, **934,140 bytes**.
- Archive SHA-256: `e2318e623c45bff380800ea6bea63914e0e7f3843825947b71073725ba4dd35c`.
- Deterministic ZIP entry count: **72**. ZIP integrity and per-entry SHA-256 verification passed.
- Embedded and externally provided `manifest.json` contains all **72** source, model and evaluation files with their exact SHA-256 digests. The manifest documents the 71 other entries and does not include itself in its member map.
- Preregistered `PROTOCOL.json` SHA-256: `c5a72cdab59dc4434d25f0b36b31d3a14821eb8b72aa3cd12ec992226424ff8a`.
- Frozen `fresh_manifest.json` SHA-256: `a4e701e1132d2cc19b56c79939ec24587dd3adef352c6ff43f7ed1453a8d59db`.
- Exact `results/aggregate.json` SHA-256: `6dffa6655738ca01c7ec2d48814f2ed8a372228967bf63998c633a2fc1f14e7a`.
- **12 original Torch checkpoints**, **12 matching NumPy named-array exports**, four training receipts and all four independently verified evaluation receipts, across seed roster `17,29,43,61` and three objective arms.
- **10,752** individually scored and host-verified rule-selection episodes; **147** independent frozen-checkpoint forward replays; **zero** accepted false Horn proofs.
- Nine tests passed under ordinary Python and nine under optimized Python (the source/corpus/hash tests are included).
- Frozen experiment dependencies were checked against their GitHub Git blob SHA values: AF-15 ranker/gym, AF-05 graph, AF-02 TypedLoop adapter and original Lean reasoner source all match.

The archive was created locally and is attached in the AF-16 research conversation. It is **not** a permanent GitHub release, and its download link should be preserved separately. The committed inventory provides exact identifiers without claiming GitHub has hosted those binary bytes.

The independent verification checked each original accepted rule against public physical incidence and derived goal truth, recomputed minimal proof budgets with an independent breadth-first oracle, verified each action's raw model argmax and retained proof edge, reaggregated all scored rows and replayed neural logits from Torch weights. The scientific negative result is preserved separately from host soundness and engineering checks.

[AF-16 draft PR #82](https://github.com/MoralityLabAI/LoopedTransformers/pull/82) is stacked on unmerged AF-15 [PR #77](https://github.com/MoralityLabAI/LoopedTransformers/pull/77). The original AF-15 source/result and all prior AF-01–15 experimental history remain unchanged.
