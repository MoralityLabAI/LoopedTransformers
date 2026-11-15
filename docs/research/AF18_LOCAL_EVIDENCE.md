# AF-18 checked local evidence inventory

**Scientific verdict:** FAIL. **Local independent engineering audit:** PASS.  
**Frozen source:** [`909b426f`](https://github.com/MoralityLabAI/LoopedTransformers/commit/909b426f0ff4d6b5c6bd4613e16d18e6858be96d).  
**Protocol SHA-256:** `7b607eb2bf6d63f4332b07102063a25c64fa5f5007be6d1821cf8b602ee59eaf`.  
**Local archive:** `AF18_local_evidence_20261115.zip`, 2,216,578 bytes, 84 files.  
**ZIP SHA-256:** `33748ddb9be1af4cd700a12c6e262d3fdc166a75850999d9d727ee7964556005`.  
**Official hosted status:** AF-18 [GitHub Actions #43716898298](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43716898298) preflight failed before any executed runner step; pinned Torch 2.5.1 and Lean 4.19.0 remain unverified.

The archive is available as a local conversation download. This GitHub file is an evidence inventory; it does not imply a GitHub-hosted binary release.

## Retained evidence

| Metric | Local checked result |
|---|---:|
| Frozen trained models | 20 |
| PyTorch/NumPy model pairs | 20 |
| Exact host proof episodes | 13,312 |
| Independent model forward replays | 297 |
| Independently reconstructed Bellman cost labels | 656 |
| Trained causal-cone probe replays | 60 |
| False accepted Horn proofs | 0 |
| Unit tests, standard and optimized | 8 + 8 passed |

Every per-file SHA-256 is recorded inside the ZIP as `manifest.json`; the standalone `AF18_local_evidence_manifest.json` also contains the outer archive digest. Independent audit records and raw per-action logits are retained for every seed.

### Model snapshots

The source root and original AF-17/16/15/05/02/Lean dependency hashes are recorded inside the immutable training/evaluation receipts. Both model parameter tensors and named NumPy exports are checked for equality on loading. Every heldout rule action is replayed by a host-owned symbolic validator independently reconstructing the original public graph and minimum-budget BFS.

The mathematical support bound is derived separately in [AF18_CAUSAL_CONE.md](AF18_CAUSAL_CONE.md). The main results and gate are in [AF18_LOCAL_RESULTS.md](AF18_LOCAL_RESULTS.md). Previous research results and original AF-01–AF-17 source files remain unchanged.
