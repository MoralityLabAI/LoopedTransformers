# Not Enough Deuterium to Detonate the Oceans

A sourced, revised research-paper edition, 20 November 2026.

Contents:

- `Not_Enough_Deuterium_to_Detonate_the_Oceans.pdf`: journal-style paper.
- `Not_Enough_Deuterium_to_Detonate_the_Oceans.md`: editable Pandoc Markdown manuscript.
- `references.bib`: 19 independently named references with linked scholarly or archival sources.
- `REPRODUCIBILITY.md`: audit and provenance guide with pinned GitHub and Hugging Face references.
- `routing_by_root.pdf`, `.png`, and `routing_root_values.json`: plot and its transcribed eight-root source data.
- `verify_routing.py`: root-level and aggregate arithmetic validator, with optional comparison to original pinned JSON.
- `make_figure.py`, `preamble.tex`: reproducible figure and PDF layout sources.

The build uses XeLaTeX. To regenerate the PDF, follow section 1 of `REPRODUCIBILITY.md`.

The revised paper is a reading of existing experiments. It does not add model inference or enlarge the evaluated cohort. The decomposition code is Goodfire upstream `1f531b3` carried unmodified into MoralityLabAI fork commit `2df0092`; review amendments distinguish historical head-SVD/row grafts from VPD, expose root-level heterogeneity, and credit third-party checkpoint authors. The archived DT05 overlap diagnostic remains invalid for unordered set membership.

The 20 November reviewer update makes the historical search denominator explicit, lists the native-fit schedule and seed-211 effect, and identifies a conflicting DT05 raw-versus-dense teacher measurement whose origin is unresolved. The default routing figure and competent-subset sums consistently use `dense_teacher.by_depth["8"]`. The verifier checks the discrepancy when given the pinned JSON.
