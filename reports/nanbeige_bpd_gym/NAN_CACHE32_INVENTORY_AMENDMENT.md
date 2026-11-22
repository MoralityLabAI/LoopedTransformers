# CACHE32 Windows host-launcher admission amendment

The real Windows venv startup fixture reproduced an admission false positive:
the host's own launcher matched the old foreign-host scan. The
correction recognizes only the calling host's current-venv parent, with exact
arguments and creation-bound identity. The actual active GPU worker remained
blocked in the fixture. No host module, model, Torch or NumPy executed.

Only the unstarted CACHE32 waiter was retired after authenticated identity,
absent worker permits/runtimes and an empty artifact root were verified. Its
original source seal49…aeddd and runtime are preserved. Original NF4, its
supervisor, BF31 revision2 and the math process continue unchanged.

Seventeen stdlib contracts passed under a128-MiB guard. Refreezing then caught
a Python-object versus serialized-JSON key mismatch in bookkeeping; the saved
and rebuilt canonical plan digests are both
`9314fe764d49a4e2276c3898a0ed720d5edb5c960c417fa03bc29c3dfc314877`.
Original plan bytes are unchanged. The failed unpublished freeze is retained.
A subsequent local contract attempt declined5-GiB RAM admission before tests;
that attempt is retained too. Current-source validation passed remote CPU CI
without reducing the local memory reserve. All24 workflows passed at
`a6143a9ac3a08d1d6912dcb5c8872b3fa4b71333`, including17 stdlib contracts,
two independent NumPy host contracts and17 small-random-model preflight
captures. [Push CI](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44970882903).

Source revision2 is frozen before inference with seal
`19688498218fd00d3d9f1aed619316597eab2ef5d3f226be9524810a2e7d541c`:
33 own files,692 model bindings,27 host bindings and the identical plan.
The original ENGINEERING receipt remains byte-exact against its archived
sources. The real Windows fixture used18.098 MiB maximum sampled private RAM
and retained5.017 GiB available; the earlier local17-contract run used22.875
MiB maximum sampled private RAM and retained4.986 GiB available. No local
Torch/NumPy or numerical fixture rerun is implied; current numerical software
validation ran remotely. These are sampled resource values.

The replacement hidden Below Normal waiter authenticated the publication at
`6b19ec88ae80e14dfe33e27d3842dcef42b0ccad`, held all four original predecessor
handles, and used10.66 MiB private commitment. Its actual PID/creation and the
original model/control handles were independently checked. Both cached
backend runtime records remain absent. No cached pretrained inference has run.
Original numerical source, full1,147-job schedule, all3,962 trajectory slots,
assets, dose/mask equations and scientific criteria remain unchanged. Formal
comparisons0; original168-case test sealed; continuous goal active.

[Full amendment proof and preserved failures](../../experiments/nan_cache32/INVENTORY_AMENDMENT.json) ·
[Original source freeze](../../experiments/nan_cache32/SOURCE_MANIFEST_before_inventory.json) ·
[Current full plan](NAN_CACHE32_PLAN.md)
