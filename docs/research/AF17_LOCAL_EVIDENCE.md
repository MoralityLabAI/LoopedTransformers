# AF-17 local evidence and model provenance

**Research source:** [89adc61cbef49b8879efbef8796e6e74208666b0](https://github.com/MoralityLabAI/LoopedTransformers/commit/89adc61cbef49b8879efbef8796e6e74208666b0). **Scope:** local Python 3.13.5, Torch 2.10.0+cpu, deterministic two-thread training. The frozen protocol is SHA-256 `578f8e8664bf7930d66b5de940d85b0409553f846f3b29636afc055694e5f4aa`.

The [scored AF-17 report](AF17_LOCAL_RESULTS.md) records a preregistered **INCONCLUSIVE** scientific verdict. All four local independent audits returned PASS. This does not certify the outstanding hosted Torch 2.5.1 runtime or Lean 4.19.0 check; [CI #43712445248](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43712445248) failed preflight before a runner began executing steps, including its retry.

## Immutable local archive

- Archive basename: `AF17_local_evidence_20261115.zip`.
- Archive size: **1,290,258 bytes**.
- Archive SHA-256: `f8fa4cff1c3975112dd071510a0c6a32b02672479ac47337bab8acea599f9de3`.
- Per-file manifest SHA-256: `f3a8553fdda108e490d0bcbc3a71c5da2e5abdecbaba7a628b7744bbc63c8a9b`.
- Archive entries: **89**, with ZIP integrity and each entry SHA-256 independently checked after writing. It contains the frozen AF-17 source, posthoc analysis, training logs, model checkpoints, independent verifier receipts, raw host episodes and exact result JSON.
- Independent aggregate JSON SHA-256: `01b98652acc3a7be6bbf5b46baa0d794e77ce1c2b203121644991b917aa1ebe2`.
- Host evidence: **11,648** original episodes and zero accepted false Horn proofs. Sampled evidence: **159** complete neural forward replays, **328** independent Bellman BFS label checks and **48** posthoc causal-identity pairs.

The archive is available as an attachment in the AF-17 conversation; this report and its full checksum manifest are committed on the PR branch. This Markdown report is not a GitHub-hosted binary release.

The posthoc [causal-cone proof and numerical tests](AF17_RECEPTIVE_FIELD.md) are separate from the frozen scientific protocol and retain exact source/checkpoint identities. The new data splits are input-feature hash disjoint from AF-16 train, dev, and prior heldout cases. All 16 Torch checkpoints have their corresponding SHA-matched named-array exports and source receipts in the archive.
