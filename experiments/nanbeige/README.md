# Nanbeige research folder

This folder keeps the NAN-COMP-03 research note next to exact-byte archived source and result receipts, and stages the follow-up NAN-COMP-04 late-evidence reciprocal gym.

## Existing work and evidence

- [NAN-COMP-03 research note](NAN_COMP03_RESEARCH_NOTE.md): retrospective low-degree shortcut, R class-score bias sensitivity, and the proposed delayed-evidence test. The original registered COMP-03 failure remains unchanged.
- [Archived original source/result files](archive/PROVENANCE.json): nine copied data files with Git blob SHA-1 identities identical to their source files in ../nan_comp03; includes public source, host labels, scores, pair grades, transcripts, and audit receipts. Both original BF16 and NF4 per-case model records continue to live in ../nan_comp03/receipts/bf16 and ../nan_comp03/receipts/nf4, with original raw artifact index retained here.
- Earlier experiments in ../nan_bpd01 and ../nan_comp02 retain their complete source-provenance chains; their results are not rewritten by this consolidation.

## Next experiment

See [NAN-COMP-04](nan_comp04/README.md) for the late-evidence coefficient task, fixed source populations, fresh CAL/DEV generator, exact validation, two-loop BF16/NF4 model runner and separate host-only grader. This folder is deliberately a new umbrella, keeping the existing experiment paths and history untouched.

The original 168-case ARG-03 held-out test stays sealed. The NAN-COMP-04 stage has not run the 3B checkpoint or BPD transfer.
