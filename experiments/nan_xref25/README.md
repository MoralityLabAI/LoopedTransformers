# Separate FP64 geometry-auditor repair

[Protocol](PROTOCOL.md): keep all original19/22 and orientation-only24 failures.
Add exactly one float64 cast to the physical-direction selection after24's
three decoder transposes. Match the actual producer's dtype without changing
the original strict tolerances. No model/worker replay or new grade.

[XREF19 report](../../reports/nanbeige_bpd_gym/NAN_XREF25_XREF19_PROGRESS.md) ·
[XREF22 report](../../reports/nanbeige_bpd_gym/NAN_XREF25_XREF22_PROGRESS.md).
