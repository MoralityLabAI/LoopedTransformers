# NAN-XREF-24: separate xref19 auditor repair

Original frozen worker/auditor sources and original failed attempts are preserved. This separate host-only audit adds the three decoder transposes required by the stored[3072,128] PyTorch Linear layout. No model call, fitting or new grade.

| Backend | Separate repair | Depth | Cell | Vectors checked | Geometry cells checked |
|---|---|---:|---|---:|---:|
| nf4 | failed_repaired_host_audit | — | pending/missing | — | — |
| bf16 | failed_repaired_host_audit | — | pending/missing | — | — |

All available vectors, all128 features/top4, fixed CAL-only dictionaries, initial/random/SVD/Haar controls, root strata and undefined correlations remain in the complete JSON. This engineering repair does not change an inference criterion, replay a worker, reconstruct a missing grade or turn an old failure into a success.

XREF19 numerical inputs and original failures were already exposed before this separate repair. XREF22 inputs remain subject to its frozen admission rules. The same unchanged independent reconstruction, controller, root averaging and physical geometry checks must pass after the mechanically reversible orientation fix. Tolerances are unchanged.

One serialized numerical CPU thread at Below Normal priority protects the math process. The XREF22 repair waits for the original XREF22 and both STORY23 lanes to finish. No causal/sham floor or power is calibrated, zero formal comparisons, no transfer/equivalence claim. The168-case test remains sealed and the continuous goal remains active.

Source seal: `636ac4474bd20c90eca8b5737e85a89f805d1b6a949f94ff60a2281ee5bc2275`.

[Complete aggregate](nan_xref24_xref19_progress.json) · [Repair protocol](../../experiments/nan_xref24/PROTOCOL.md) · [Original report](NAN_XREF19_PROGRESS.md)
