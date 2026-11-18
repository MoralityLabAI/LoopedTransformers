# LS-SAE13: isolated down-projection CPU thread and batch-shape test

**All four GitHub-hosted CPU jobs passed.** The study isolates the first
differing operation in the earlier ACT5 B1-vs-B2 microtrace. It remains
an exploratory engineering diagnostic on two *known* positions
(LS-SAE12 pool 23 and 40), with no scientific gate.

## Provenance

- [GitHub Actions #44294482193](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44294482193);
  source commit \`0ea7646490952306de8406dc810752db33cfb396\`.
- All jobs: Linux, Python 3.11, PyTorch 2.7.1+cpu, NumPy 2.1.3;
  initial four-thread ACT1..ACT4 model forward; checkpoint SHA256
  \`932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3\`.
- Every job uses the same pinned LS-SAE12 source receipt and four
  numerical backend settings: thread count {1,4} x MKLDNN {off,on}.
- The first \`layers.0.down_proj\` input is captured during ACT5 at
  both B1 and B2 from the same exact ACT4 anchor. The first row of
  both input tensors is bit-identical. Its linear weight has shape
  [512,1536], and the input row has shape [97,1536].
- Each job's JSON includes output SHA256, float32 error statistics,
  original forward-hook output comparison and the float64-accumulation
  reference. A companion NPZ captures exact float32 inputs, weights
  and observed projection outputs, with audited SHA256.
- Independent audits passed in all four jobs.

| Job | CPU | B1/B2 observed discrepancy on 23 and 40 | SHA256 of audited JSON |
|---|---|---|---|
| 0 | AMD EPYC 9V74 | exactly 0 | \`7144e376c99ee8b70465a4fbba09abefaf6474d07c076c8968df582df30d2daa\` |
| 1 | Intel Xeon Platinum 8370C | max 4.76837158203125e-7 | \`84c3318e2639ee55ceb4a1f78e7ff889b912c6595c41a63aeec231ef8c7d0071\` |
| 2 | AMD EPYC 9V74 | exactly 0 | \`7144e376c99ee8b70465a4fbba09abefaf6474d07c076c8968df582df30d2daa\` |
| 3 | AMD EPYC 9V74 | exactly 0 | \`7144e376c99ee8b70465a4fbba09abefaf6474d07c076c8968df582df30d2daa\` |

All three AMD NPZ fixtures have byte-identical SHA256
\`01639e6744cb7ab0aa5d97cb425f60ed67bb4fcb7e59ae31830d40c0161f4050\`.
The Intel fixture SHA256 is
\`37cecc1b067abe54e4e3b426546de0e41e6f552970cf3552686df540d5fc1ca9\`.
The processor-specific ACT4 hidden states and captured input bytes differ
across CPU types; this is an *intra-environment* identical-input
experiment, not a single byte-identical input fixture run on two
different processors.

## Mechanism on Intel Xeon 8370C

| Num CPU threads | MKLDNN | B1 vs B2 max abs output difference | B1 vs explicitly flattened 2D input |
|---:|---|---:|---:|
| 1 | off | 0 | 0 |
| 1 | on | 0 | 0 |
| 4 | off | 4.76837158203125e-7 | 0 |
| 4 | on | 4.76837158203125e-7 | 0 |

Both positions give these same maximum B1/B2 differences.

The isolated \`torch.nn.functional.linear\` operator reproduces
the first original projection outputs *bit-for-bit* at four threads
for both batch shapes. The input to the linear projection and the
weight are held fixed; even duplicating the B1 first-row input as a
B2 tensor reproduces the difference. The discrepancy is therefore
internal to this float32 linear operation's execution path.

At one thread, B1 and B2 yield bit-identical outputs. The one-thread
result matches the **four-thread B2** original output in both
positions; its B1 output differs from the four-thread B1 original.
Toggling \`torch.backends.mkldnn.enabled\` changes none of these
observations. The 2D view of the B1 input yields the same output
as its B1 3D view. These controls narrow the difference to a
thread- and matrix-shape-dependent reduction/linear kernel path
on this Intel CPU.

The maximum absolute difference between float32 replay and the
float64-accumulation/cast reference is 2.384185791015625e-6
for position 23 and 1.430511474609375e-6 for position 40 on
the Intel job. The FP64 reference is diagnostic; it does not
establish a formally exact dot-product implementation.

## AMD EPYC 9V74 control

All three AMD jobs reproduce identical receipts. Their B1 and B2
projection outputs are bit-identical at one and four threads,
regardless of the MKLDNN setting, and the direct isolated
\`F.linear\` outputs match the captured native model outputs.
The CPU family is therefore associated with the numerical
batch-shape outcome in this host sampling.

## Consequences and remaining controls

An Intel one-thread down-projection removes the *isolated*
shape-dependent float32 discrepancy. A full-TRM one-thread
ACT5..ACT16 replay is still necessary to determine whether
other matrix operations or attention paths introduce differences,
and whether all 16 ACT predictions become invariant to batch shape.

The exact low-level BLAS kernel/instruction and reduction ordering
have not been identified; PyTorch's MKLDNN toggle is insufficient
to name the responsible backend. Additional tests should pin the
same input/weight fixture across CPU types, record BLAS runtime
configuration, and include a full-model one-thread run.

This does not modify LS-SAE12's registered FAIL or establish any
direction-specific causal efficacy.
