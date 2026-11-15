# PEF-14 | Independent hosted Python 3.12 replication

After local PEF-14 scoring, the GitHub workflow was extended to train all six seeds and run the independent auditor in a fresh GitHub Actions CPU environment. [Run 43722118391](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43722118391) completed **successfully** on November 15, 2026, under CPython 3.12.16, pinned PyTorch 2.10.0+cpu and the same registered 450-step CPU protocol. Its artifact is named `pef14-independent-six-seed-cpu-replication` (artifact ID 14164189151) and contains six binary checkpoints, every training/probe/heldout receipt and an independent audit result.

## Artifact-to-artifact comparison

An independent local comparison between the hosted ZIP and the original PEF-14 local receipt found:

- **All six frozen checkpoint files have byte-identical SHA256 hashes** across the two environments. The per-model parameter tensor hashes agree, with the same 450 optimizer steps and frozen source assignment.
- **All 96 registered heldout conditions have exactly the same integer correct counts** out of 768, and all 96 unquantized post-ablation clean correct counts agree.
- All six seed-level MAEs and registered successes agree to displayed precision. Registered result remains **2/3 coalition seeds** and `criterion_met`, with seed 223 failing its individual criterion.
- Each of the 96 scalar probe forecast predictions (`mixture`, `uniform`, `exact-phase`, `conditional-cross`) agrees in the comparison. The underlying probe ledger files have different byte SHA256s: local `6f7ff6ae664679a742e8f15cb58ea3f63e5f489641ab78decf90c29ad58a5f53`, hosted `a89553b25840cb6d48629cd89749b1a1f0aaede179602b9992e56d85cfc3d074`. Their source/processor metadata and float64 raw arrays therefore remain separately identified, rather than claiming all artifacts are byte-identical.
- The independent auditor was run on the **hosted** artifact in its own runner and again on a local extraction of those hosted bytes. Both checks passed, confirming all six checkpoint digests and all 96 probe-only forecasts.

The local fresh-training ledger SHA256 is `cf60408cd8dbf07083b3146bc4dc7e63e9444d79fabff5877e47a98e55586dd4`; hosted training-ledger SHA256 is `b456a82f0a84b49d61eb8abd0979c9951ba520d0d61c58be135f0a06c97a3324`. The local 200-second process interruption generated completed checkpoints whose lost first/final loss receipts were labeled null on resume; hosted uninterrupted training recorded those losses. The checkpoint bytes were independently verified as identical despite that metadata difference.

## Interpretation

This is a **cross-host and Python-minor-version replication of the same frozen experimental design**, training seeds and context partition. It raises confidence that the measured 2/3 improvement is a reproducible result of the registered code. It does not add six statistically independent seeds or new context partitions. The registered 90% normal intervals remain badly miscalibrated at 20/48 coalition and 25/48 threshold coverage. The mathematical total-variation bound is conditional on an idealized iid context-family law, while the finite source partition sampled contexts without replacement.

Hosted artifact SHA256 and all 96-condition comparison are preserved in the conversation's combined reproducibility archive. The PR remains draft and unmerged for scientific review.
