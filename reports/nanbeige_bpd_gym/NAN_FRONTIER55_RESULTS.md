# FRONTIER55 dose and full-head tradeoff for Nanbeige route contrasts

The exact BF16 support bound put the minimum nominal dose for the signed
NF4 learned-minus-static contrast between **0.659% and 0.697%** across 32
targets from eight exposed CAL roots. Each bound already allows the full
3% upward tolerance. An edit at 0.50% cannot reach cosine 0.50 with any of
these NF4 targets. The bound leaves feasibility above those doses open.

The table gives all-gate passes for the ROUND54 spacing rule and dense
comparator. Entries are worlds out of eight in the stated CAL panel.

| Nominal residual L2 dose | BF16 fit, spacing / dense | BF16 disjoint check, spacing / dense | NF4 fit, spacing / dense | NF4 disjoint check, spacing / dense |
| --- | ---: | ---: | ---: | ---: |
| 0.25% | 5 / 0 | 3 / 1 | 0 / 0 | 0 / 0 |
| 0.50% | 4 / 0 | 5 / 1 | 0 / 0 | 0 / 0 |
| 0.75% | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 1.00% | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 1.50% | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 2.00% | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

Fit retained the 0.25% spacing recipe before opening check states. BF16's
0.50% spacing recipe qualified five of eight check worlds, while its fit
count was four of eight. At 0.75% and 1.00%, the spacing rule produced
six valid physical groups per BF16 panel at each dose. All failed the
0.125 maximum absolute vocabulary-logit tolerance. Across all recipes,
no NF4 world completed the four-arm physical and contrast gates. The
primary full-head audit therefore has zero edited NF4 groups.

## NF4 active-edit diagnostic

The fit physical results fixed the dense 0.75% and 1.50% recipes for a
separate NF4 active-only head check. At both doses, both active signs
matched the registered radius, retained the contrast, and preserved the
full head in **16/16 CAL worlds** across the two panels. The median
minimum signed cosine with the original learned-minus-static contrast
was 0.998 at both doses.

| Nominal dose | Both active heads passed, fit | Both active heads passed, check | Largest absolute vocabulary-logit shift | Largest absolute bare-label margin shift |
| --- | ---: | ---: | ---: | ---: |
| 0.75% | 8/8 | 8/8 | 0.0625 | 0.03125 |
| 1.50% | 8/8 | 8/8 | 0.125 | 0.03125 |

The parent dense sham exceeded the 0.10 original-contrast leakage
threshold in at least one sign in every world at both doses. Thus the
0.75% active recipe has current-head room under the registered gate, with
the matched-sham comparison still open. A foil synthesized in coordinates
with fine BF16 steps can be qualified at the same dose against the same
full-head and contrast gates. The auxiliary diagnostic streamed 80
states with zero new pretrained decoder forwards. [Active-head protocol](../../experiments/nan_frontier55/ACTIVE_HEAD.md)
· [Active-head aggregate](nan_frontier55_active_head.json)

## Audit coverage

Both exposed CAL panels were reproduced at 0.25% for 192 gate fields
each. The pinned head streamed 166,144 logits per state for 304 unique
clean or eligible edited states. The native greedy winners and top logits
matched exactly, and the largest log-normalizer error was 4.05 × 10⁻⁷.
The audit used 143.8 MiB maximum sampled private memory. Roots are the
independent units. All doses, signs, worlds and precisions repeat within
roots. Zero formal comparisons, no new pretrained decoder forwards and
no sealed ARG-03 data. [Protocol](../../experiments/nan_frontier55/PROTOCOL.md)
· [Aggregate](nan_frontier55_results.json)
