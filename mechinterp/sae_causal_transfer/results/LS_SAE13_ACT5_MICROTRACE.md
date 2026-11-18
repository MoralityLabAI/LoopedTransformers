# LS-SAE13: ACT5 first-divergence localization across hosted CPU types

**Exploratory, independently audited.** Two repetitions of the same
ACT4-anchor, B1-vs-B2 single-ACT trace happened to receive different
hosted processors. They agree on the pretrained checkpoint SHA, source
receipt, Python 3.11.18, NumPy 2.1.3, PyTorch 2.7.1+cpu, Linux kernel and
four threads. Their divergent numerical behavior is reproducible in the
original LS-SAE13 Intel CPU pilot versus the AMD baseline.

## Matched experiment

- Known LS-SAE12 positions 23 and 40; no fresh or held-out source.
- ACT1..4 in B1. Save exact \`z_H,z_L,x_emb\` anchor, repeat it into B1
  and B2, with a clean duplicate filler row in B2. Run one native ACT5
  (3 H cycles, each with 6 L updates and one H update).
- Trace the first row's input and output SHA256 at every projection in
  \`layers.0\` and \`layers.1\`: \`qkv_proj\`, \`o_proj\`,
  \`gate_up_proj\`, \`down_proj\`. There are
  3 x (6 + 1) x 2 x 4 = **168 calls**. Compare them in execution order.
- Rerun each batch with a focused capture of the earliest mismatching
  projection and calculate its input/output float32 discrepancies.
- \`trace_act5.py\` performs the microtrace and writes the full digest
  pair list; CI recomputes receipt shape and SHA256.

## Results

| | Run 44292485927 | Run 44292815767 |
|---|---|---|
| CPU | AMD EPYC 9V45 96-Core | Intel Xeon Platinum 8573C |
| Source commit | \`b0194f9\` | \`738be59\` |
| CI verdict | PASS | PASS |
| First projection input mismatch | none of 168 | index 4 |
| First projection output mismatch | none of 168 | **index 3, layers.0.down_proj** |
| ACT5 hidden states and logits, B1 vs B2 | bit-identical | differ |
| Receipt SHA256 | \`1c76d8af7d5a1bbfa42ba3775d424d8a52256c04acd2d6a3f0990268286f88d0\` | \`cdb462002a8262756e71f6da57ed19288a5efb9155c7aa4db1495797805bba1c\` |

At Intel event index 3, the input to \`layers.0.down_proj\` is **bit-identical**
between the B1 and B2 first rows. The previous projection inputs and
outputs (indices 0, 1 and 2) also match. The output of
\`layers.0.down_proj\` first differs, with the same magnitude on both
known puzzles:

| Position | Down-projection maximum abs B1/B2 difference | Down-projection Frobenius difference | ACT5 maximum abs logit difference |
|---|---:|---:|---:|
| 23 | 4.76837158203125e-7 | 1.1353619221326739e-5 | 9.34600830078125e-5 |
| 40 | 4.76837158203125e-7 | 1.0721741884366905e-5 | 1.201629638671875e-4 |

This is the **first L-level update in the first H cycle of ACT5**;
the differing projection is a \`1536 -> 512\` float32 linear map
on 97 tokens. The next recorded projection input differs as a downstream
consequence. The AMD run showed no first-row input/output differences at
any of the 168 projected operations, and ACT5 logits remained identical.

## Mechanistic interpretation

The first detected numerical difference occurs **inside a linear
projection on identical first-row inputs**. The B1 and B2 tensor shapes
feed that projection into potentially different CPU GEMM kernels or
reduction schedules. Its output changes by roughly float32 rounding
scale, then the recurrent network magnifies the difference. On the Intel
pilot, these ACT5 differences ultimately flip argmax predictions at
ACT9 (position 23) and ACT11 (position 40); see
[LS-SAE13 Linux pilot](LS_SAE13_LINUX_PILOT.md).

The two host CPU models are the strongest observed difference in the
software environment. They are **not a controlled randomization of CPU
features**; the traces do not identify an MKL instruction, oneDNN kernel,
thread schedule or compiler path by name. The B2 microtrace has two clean
rows, whereas the main pilot's B2 rows contain an edit; the pilot's
independent B2+ and B2- clean rows were bit-identical, supporting a
batch-shape rather than row-content explanation in that environment.

## Next controlled test

With exact \`down_proj\` input bytes and weights captured, replay the
isolated \`nn.Linear(1536,512)\` operation at B1 and B2, holding the
first-row input constant. Repeat under one- and four-thread CPU settings,
with and without MKLDNN where available, and with an explicit
float64-accumulation reference. Record full float32 difference maps
and check whether padding B1 to a canonical B2 or B61 eliminates
shape-dependent drift on both processors.

Scientific LS-SAE12 remains FAIL; these findings diagnose numerical
integrity only and do not establish structural-direction efficacy.
