# NAN-XREF-25: separate xref19 auditor repair

Original frozen worker/auditor sources and original failed attempts are preserved. This separate host-only audit adds the three decoder transposes and one explicit FP64 direction cast required by the stored[3072,128] layout and producer float64 geometry arithmetic. No model call, fitting or new grade.

| Backend | Separate repair | Depth | Cell | Vectors checked | Geometry cells checked |
|---|---|---:|---|---:|---:|
| nf4 | passed | 1 | completed | 1706 | 18 |
| nf4 | passed | 2 | completed | 1706 | 18 |
| bf16 | passed | 1 | completed | 1706 | 18 |
| bf16 | passed | 2 | completed | 1706 | 18 |

All available vectors, all128 features/top4, fixed CAL-only dictionaries, initial/random/SVD/Haar controls, root strata and undefined correlations remain in the complete JSON. This engineering repair does not change an inference criterion, replay a worker, reconstruct a missing grade or turn an old failure into a success.

XREF19 numerical inputs and original failures were already exposed before this separate repair. XREF22 inputs remain subject to its frozen admission rules. The same unchanged independent reconstruction, controller, root averaging and physical geometry checks must pass after the mechanically reversible orientation fix. Tolerances are unchanged.

The original orientation-only XREF24 NF4/BF16 strict geometry failures remain failed. Its future XREF22 stage is unrun under the prospectively published supervision amendment. Only its authenticated waiting supervisor is replaced; no model or numerical worker is stopped.

One serialized numerical CPU thread at Below Normal priority protects the math process. The XREF22 repair waits for the original XREF22 and both STORY23 lanes to finish. No causal/sham floor or power is calibrated, zero formal comparisons, no transfer/equivalence claim. The168-case test remains sealed and the continuous goal remains active.

Source seal: `7d9ac0ff9d809900fc967283c3ec8b24fd4fefa9da410d54ec4227ab2c3a1975`.

[Complete aggregate](nan_xref25_xref19_progress.json) · [Repair protocol](../../experiments/nan_xref25/PROTOCOL.md) · [Original report](NAN_XREF19_PROGRESS.md)
