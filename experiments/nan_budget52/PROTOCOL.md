# NAN-BUDGET-52

Thinking Machines Lab reported that its variable-loop Inkling checkpoint improved on
serial problems with more passes and attained higher accuracy at the same reasoning
token budgets on some tasks. This companion to STREAM41 tests the corresponding
questions for bounded Nanbeige answers, using its registered one- and two-pass runs.
The fixed-depth Inkling 2 Mini and its separately trained variable-loop research
checkpoint are different models. [Inkling 2 Mini study](https://thinkingmachines.ai/news/inkling-2-mini/#monitorability-of-looped-transformers)

Use all 240 existing same-prompt one-to-two-pass pairs per precision from the
READOUT34 roster. Retain the 8- and 96-event histories, actual/belief/nested-belief
queries, compute/copy/erased controls, and CAL/DEV split. The fixed panels are

- Output token caps of 8, 16, 32, 48, 64, and 96 at both depths.
- Decode phase-token visit ceilings of 16, 32, 64, and 94 at both depths. The first
  output token comes from prefill. For depth `d` and ceiling `v`, the cap is
  `1 + floor(v / d)`. Prefill is outside this ceiling. Totals include both
  prefill and decode as unpadded logical work and omit physical padding, head
  work, wall time, and other runtime costs.

Analyze the complete panel descriptively. Report known correct/incorrect,
unknown, repair, regression, ties, both-world root outcomes, and work totals
on the same verified pairs. Report how many canonical replies fit each cap.
That count describes the pinned tokenizer's canonical encoding, without
assuming it is the shortest possible encoding or using it to exclude cases.

The pinned-tokenizer preflight found that all 48 canonical final-only replies
fit eight tokens. No canonical packet fit 48 tokens, and all fit 64. The 94-visit
decode ceiling allows 95 output tokens at one pass and 48 at two passes, so
the canonical packet is available at only one depth in that panel. Read packet
comparisons at the shared 64- and 96-token caps for canonical support at both
depths. The preflight checked the full roster before either backend had a
verified STREAM41 closure.

Only read bytes covered by the STREAM41 terminal and independent stop audit.
All caps at a given depth apply uniformly to both rows of each two-row
batch. Up to the lower cap the original greedy prefix and batch occupancy
are the same. Under the first-stop rules an earlier prefix cannot already
be a complete answer. If a verified completion extends past the lower cap,
score it using the existing incomplete-answer rule. Missing originals
remain unknown. This replay does zero new pretrained calls.

For compute and copy also compare final-only correctness with packet final-room
correctness for the same world at each depth and output cap. Within the packet,
report how often a correct final room also comes with all correct checkpoints
and the exact oracle path. This checks emitted evidence and assigns no claim
about the internal reasoning that produced it.

The independently sampled unit is the generated source root. This panel has
3 CAL and 9 DEV roots, with repeated worlds, formats, depths, budgets and
precisions. It uses zero formal tests, no sealed ARG-03 data, no fitted exit
policy, and no estimated causal effect or matched-sham threshold. Treat it as
a development pilot.

Run from the repo with the pinned Nanbeige Python environment and the original
READOUT34 tokenizer receipt. The scorer writes an aggregate only. Before
STREAM41 closes, the same command verifies source and tokenizer binding and
emits unknowns for missing backend terminals.

```powershell
python -B -m experiments.nan_budget52.verify
python -B -m experiments.nan_budget52.run --snapshot $snapshot `
  --tokenizer-receipt runs/nan_readout34/tokenizer01/tokenizer_receipt.json `
  --output runs/nan_budget52/budget_report.json
```
