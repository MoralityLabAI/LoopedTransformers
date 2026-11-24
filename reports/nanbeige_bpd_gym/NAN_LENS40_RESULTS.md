# NAN-LENS-40 calibrated phase readout

A CAL-only linear correction reduced restricted final-output KL at six of
seven nonfinal sites on focused COMP14 DEV prompts in both NF4 and staged
BF16. At the end of pass 1, it increased KL in both precisions and recovered
zero of 16 both-world roots. The late gain in correctness therefore persists
under this calibration. The estimator predicts the model’s distribution over
16 bare labels. Task correctness comes from COMP14’s independent oracle.

Inspired by Goodfire’s raw-versus-tuned lens comparison on Inkling 2 Mini,
LENS40 uses a Nanbeige-specific observer at four layers per physical pass.
It shares the same raw pinned head across sites and compares a CAL mean logit
shift, a CAL linear correction, and the same correction with coordinate 2279
omitted from the fitted features. The raw scores stay fixed in the omission
control. Every variant scores the identical prompts.
[Inkling results](https://www.goodfire.com/research/interpreting-inkling-2-mini/#decoding-loops-logit-lens-tuned-lens-coda-lens) ·
[Frozen protocol](../../experiments/nan_lens40/PROTOCOL.md)

## Focused DEV results

These are 32 worlds from 16 previously exposed roots per backend. A both-world
root requires two correct answers. Layers are zero-based.

| Backend | Site | Raw / mean shift / linear KL, nats | Raw / mean shift / linear correct worlds | Raw / mean shift / linear both-world roots |
|---|---|---:|---:|---:|
| NF4 | Pass 1, layer 21 | 1.953 / 2.077 / 2.043 | 2 / 2 / 0 | 0 / 0 / 0 |
| BF16 | Pass 1, layer 21 | 2.500 / 2.585 / 2.582 | 3 / 1 / 2 | 0 / 0 / 0 |
| NF4 | Pass 2, layer 17 | 1.848 / 1.541 / 1.001 | 24 / 27 / 27 | 11 / 11 / 11 |
| BF16 | Pass 2, layer 17 | 1.139 / 1.003 / 0.606 | 29 / 30 / 30 | 14 / 15 / 14 |
| NF4 | Pass 2, layer 21 | 0.000 / — / — | 29 / — / — | 13 / — / — |
| BF16 | Pass 2, layer 21 | 0.000 / — / — | 32 / — / — | 16 / — / — |

At pass 2 layer 17, the linear correction had lower KL than the mean-shift
control in both precisions and tied its oracle-correct world count. Omitting
coordinate 2279 from the fitted features gave KL 0.895 NF4 and 0.551 BF16,
with 28/32 and 29/32 correct, respectively. This is a feature omission in a
readout fit. A causal intervention on the live model would test a different
quantity. The phase marker’s size is insufficient evidence for a task edge.

The archived unrestricted native head scored 3/32 to 29/32 across passes
in NF4, with 26 repairs and zero regressions. BF16 scored 3/32 to 32/32,
with 29 repairs and zero regressions. The 16-label lens and native head
each retain their own correctness counts. All erased DEV twins had exactly
equal restricted score vectors in every reported observer and site.

## Verification and evidence scope

The audit read 144 archived world-arm records in each precision and measured
eight sites per record. Fits used the 32 focused/copy records from the eight
CAL roots only. The archived layer-21 selected logits and native winner
logits replayed exactly at both passes. A second calculation recomputed
class choices, final-output agreement, paired repairs and regressions,
and both-world correctness for 348 reported cells. In the synthetic leakage check, changing the DEV target
logits left predictions bitwise equal, and permuting label columns permuted
the predictions. The first attempt stopped before scoring because the
COMP14 audit represents verdicts as integer 0/1. The replay used the corrected
reader and a new attempt directory.

The audit used at most 117.3 MiB sampled private commitment and 94.7 MiB
sampled working set, with 7.35 GiB minimum sampled available RAM. It ran
alongside RAM28, with zero new pretrained forwards and zero inferential
tests. The exposed COMP14 roots support a development observation. A fresh
confirmation set and matched physical interventions would be needed to
test whether the late readout reflects donor-specific causal transfer.
The sealed ARG-03 test was untouched.
[Aggregate results and provenance](nan_lens40_results.json)
