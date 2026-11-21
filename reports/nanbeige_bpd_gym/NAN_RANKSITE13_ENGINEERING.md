# Missing rank/site conditions: complete random pipeline evidence

All six missing native rank/site/depth cells completed the actual eight-step
BPDTrainer recipe on a small random target. An independent CPU artifact audit
verified **48 optimizer steps**, checkpoint shapes/updates, pointwise controller
replay, source order, four-term loss accounting, all diagnostic mode coverage,
all-one closure and unchanged target bindings. These are random integration
results, not additional pretrained VPD/BPD fits.

| Physical module | Rank | Fit depths | Full pipelines | Optimizer steps |
|---|---:|---|---:|---:|
| down_proj | 16 | T1, T2 | 2/2 | 16 |
| o_proj | 8 | T1, T2 | 2/2 | 16 |
| o_proj | 16 | T1, T2 | 2/2 | 16 |

Every cell retained exact native all-one logits/boundaries, a finite nonzero
pre-optimizer factor-gradient gate, positive factor/controller updates, exact
CAL mean recalibration and finite positive-diagonal/permutation gauge checks
stable at 1e-5. Initial component-only and final all-one/component/static/
clean-informed/reached-state diagnostics retain complete coverage. Shared target
weights were unchanged across all six cells; there was no factor/controller
carryover. General rotations and identifiable hard-mask components are unproven.

The random model had 22 layers, hidden64, MLP128, eight query heads/two KV heads,
head dimension16 and vocabulary166144. Its o_proj input width128 differs from
output width64. The .05 CUDA cap held: peak allocated49,825,792 bytes, reserved
75,497,472 bytes; runtime389.969 seconds, one numerical CPU thread/Below Normal.
The registered large BF16 cache job continued unchanged. No pretrained weights
or host gold were opened by this random fixture.

The [plan](../../experiments/nan_ranksite13/ENGINEERING_PLAN.md), six-cell config,
derived recipe and tests were pushed at `7d8411b` before execution. Historical
FIT07/RECOMP09 sources were not edited. Full evidence is in the
[pipeline receipt](../../experiments/nan_ranksite13/receipts/random_pipeline_v1.json)
and [independent audit](../../experiments/nan_ranksite13/receipts/random_pipeline_independent_v1.json).
Raw checkpoints, replies and per-record diagnostics remain local.

## o_proj requires the following MLP, with finite precision measured

The new [bridge](../../mechinterp/nanbeige_bpd_gym/o_proj_bridge.py) explicitly
maps a physical o_proj output write through its native residual addition,
pre-MLP RMSNorm and MLP to residual_after_layer, before final loop normalization.
It binds actual pre-attention residual/output tensors to an independently
captured pre-MLP input, preserves the full source prefix GEMM shape and edits
only the last token. It never reconstructs state by subtracting rounded outputs
or substitutes a same-width identity map for the MLP.

Three CPU tests passed, including an independent float64 analytic RMSNorm/SwiGLU
Jacobian, full-prefix shape preservation, immutable parameters and rejection of
bad coordinates/live hooks/trainable or quantized modules. The separate native
random proof was registered/pushed at `f16b955` before execution. Both native
passes and all 16 fixed directions, three doses and two signs were retained:
**192/192 physical full-forward injections exactly matched the local remainder**.
Both clean remainders were byte exact. Peak CUDA allocation17,427,968 bytes;
runtime54.735 seconds. [All comparisons and diagnostics](../../mechinterp/nanbeige_bpd_gym/receipts/o_proj_bridge_random_v1.json).

Exact local finite-write execution does not imply that a BF16 autograd
derivative accurately predicts small rounded writes. The following table
retains all 16 directions per cell; these are correlated measurements on one
random checkpoint, without confidence intervals or equivalence thresholds.

| Native loop | Relative o_proj output L2 dose | Median centered-write/JVP relative error | Maximum error |
|---|---:|---:|---:|
| 1 | .001 | 1.8040 | 3.0452 |
| 1 | .01 | .4375 | .4931 |
| 1 | .1 | .0600 | .0690 |
| 2 | .001 | 3.0292 | 12.4187 |
| 2 | .01 | 2.1007 | 2.4898 |
| 2 | .1 | .3141 | .3664 |

Requested doses scale the last-token o_proj output norm; the receipt separately
retains actual rounded input and post-addition displacements. The derivative
uses native-dtype tangents and is a surrogate through rounding. Its relative
difference from an identity bridge was .02693/.00394 at loops1/2, while finite
write errors above are materially larger. No derivative/finite-write acceptance
threshold was tuned after seeing these results. A real o_proj/SAE comparison
must preserve actual finite native writes/doses and coordinate checks alongside
derivative geometry; derivative cosines alone are insufficient.

The additional production post-fit fixture passed all six cells and preserved
**5760 signed finite writes**. An independent CPU audit checked every archived
BF16 addition, actual norm, control binding, all72 feature-response groups and
36 derivative groups per cell, retaining all128 features. Its random dictionary
was only an engineering coordinate fixture. This does not add pretrained scores.
The [post-fit receipt](../../experiments/nan_ranksite13/receipts/postfit_random_v1.json)
and [all-cell audit](../../experiments/nan_ranksite13/receipts/postfit_independent_v1.json)
retain exact provenance. Native MLP outputs/JVPs are source-bound measurements,
not an independently replayed3B MLP in the CPU audit.

The real six-cell pretrained extension is now frozen under source-manifest SHA
`27b3e5e636abcdf9217f88a5f19999c46b56d9723c76e6f5263c781e93239ddc`.
Its [protocol](../../experiments/nan_ranksite13/PROTOCOL.md), worker, independent
audits, cached-suite and actual finite-write SAE probes are bound before
inference. The initial [progress report](NAN_RANKSITE13_PROGRESS.md) contains
no new model scores. The finite queue waits for the actual KV12 supervisor/model
to finish before loading the first cell. BF16 eligibility remains an exposed
screen; NF4's failed fitting screen stays closed. No donor transfer is enabled.

Unit: one random target/seed per engineering fixture; cells, metadata roots,
tokens, loops and directions/doses are repeated measures. Scientific causal
floor, matched-sham floor and causal power are uncalibrated. Planned/actual
formal comparisons zero. Verdict: actual framework/coordinate engineering,
without pretrained component quality, feature mediation or causal transfer.
The original 168-case test stays sealed and original algebraic gates remain
failed. The full research goal remains active.
