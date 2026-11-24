# Separate decoder-orientation audit repair

Read the [protocol](PROTOCOL.md). Original XREF19/XREF22 sources, attempts and
failures remain unchanged. This distinct host-only registration adds exactly
three decoder transposes to each frozen independent auditor and redirects its
final receipt into a separate namespace. It performs no model/worker replay.

Software: `python -B -S -m unittest discover -s experiments/nan_xref24/tests -v`.
The fixtures evaluate the exact changed expressions with rectangular decoder
matrices and prove the full AST transformation is reversible.

[XREF19 separate audit report](../../reports/nanbeige_bpd_gym/NAN_XREF24_XREF19_PROGRESS.md)
and [XREF22 separate audit report](../../reports/nanbeige_bpd_gym/NAN_XREF24_XREF22_PROGRESS.md)
retain all complete/missing cells and provenance.
