# NAN-COMP-03 post-hoc audit: reciprocal score margins, answer-prior sensitivity, and low-degree source shortcut

**Evidence class:** deterministic retrospective analysis of already exposed development cases. No new checkpoint inference, model updates, confirmation, or original competence-gate promotion. All registered NAN-COMP-03 results and the sealed 168-case ARG-03 test remain unchanged.

## 1. What source-only classification actually requires

For the 16 committed development worlds, each of the two current demonstrations has coefficients a0=b0=1 and a1 in {-1,+1}. The output operator is exactly identifiable from the first two nonconstant coefficients:

- I: b1=a1, b2=a2.
- S: b1=0, b2=a1.
- N: b1=-a1, b2=a2.
- R: b1=-a1, b2=a1^2-a2.

R and N cannot collide in this integer domain, since 2a2=a1^2=1 would be required. This simple exact source-only decision rule recovers **16/16 worlds** using one demonstration per world. It does not require model inference, hidden-state recurrence, an exact degree-32 future calculation, or a reciprocal power-series recurrence. Confirmed directly from both demonstrations and the frozen host labels by posthoc_shortcut.py.

BF16's 12/16 original success is precisely all I/N/S and zero R; this is compatible with selecting on b1 and failing the b2 discrimination. It does not prove that the neural model implements that specific rule.

## 2. Readout score post-hoc R-bias sensitivity

The original bounded greedy model output starts with the same bold opener in all 16 DEV curriculum cases. The saved class_logprobs at the subsequent answer-token position support a **hypothetical class-score reranking** by adding delta to the R class. This is NOT a direct token-logit intervention, live autoregressive run, prospective readout calibration, or another accepted original model response.

A source-only scan over delta in [-1.00,+3.00] with fixed 0.01 increments gives:

| Readout-score scenario | BF16 correct /16 | BF16 both /8 | BF16 R /4 | NF4 correct /16 | NF4 both /8 | NF4 R /4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No correction, frozen observed outcomes | 12 | 4 | 0 | 10 | 2 | 0 |
| Post-hoc delta=+1.25 | 14 | 6 | 3 | 12 | 4 | 3 |
| Post-hoc delta=+1.50 | 13 | 5 | 4 | 11 | 4 | 4 |

At delta=+1.25, source-erased class-reranked outcomes are 4/16 in each backend, giving BF16 10/16 and NF4 8/16 differences. The original registered source-use threshold was 4/16. BF16 therefore meets all four original **numerical thresholds on re-ranked exposed scores** for delta approximately [1.25,1.49]; NF4 meets them at none of the 401 scanned offsets. These numerical windows were found AFTER seeing outcomes and therefore have no confirmatory status. The original actual bounded-greedy competence results remain BF16 12/16, both 4/8, R 0/4 and NF4 10/16, both 2/8, R 0/4.

An increased R prior can also cause false R choices in correctly classified N examples. The viable BF16 interval is narrow. No global prior adjustment proves that a model computes reciprocals.

## 3. Source-sensitive reciprocal signal remains below the decision boundary

Define M_R(s)= log p(R|s) - max_{o in {I,N,S}} log p(o|s), using the saved answer-token operator-class log probabilities. For each of four R-target donor roots, compare M_R(donor) to M_R(recipient) under an identical input/source family and distinct independently verified output labels.

- BF16 reciprocal curriculum: positive in 4/4 roots, mean +2.531244 class log-prob units.
- NF4 reciprocal curriculum: positive in 4/4 roots, mean +1.999995.
- Within-root outputs erased: exactly zero paired differences, because the prompts/token IDs are identical.

This is **observational source sensitivity**, with four exposed counterfactual pairs and a class-score metric. An internal BPD channel has not been located, intervened on, or shown to carry a verified reciprocal solution. No unconditional p-values are licensed by these data.

## 4. New prospective concept: delayed-evidence reciprocal discrimination (NAN-COMP-04 candidate)

Retain the old failed report. Freeze a separate source-only protocol and all generator seeds before new inference.

To defeat the x1/x2 source shortcut, construct fresh integer-series families:

    f(x) = 1 + a*x^m + c*x^(m+2),  m odd, e.g. 3 or 5, a,c nonzero integers.

When we compare N(f)=f(-x) and R(f)=1/f, their coefficients match through degree 2m-1. At degree 2m, the reciprocal has the extra a^2 term. At degree 2m+2, the reciprocal has an additional 2ac term, relative to its first-order negative. For m>=3, terms of order u^3 begin after degree 2m+2, so this is an exact power-series coefficient identity.

Make source-paired N/R worlds with the same inputs; display the output arrays only through degree D chosen from:

- D=2m-1: N and R are observationally identical on these examples. This is a source-only negative control; all deterministic classifiers have a 50% root-pair case ceiling with equal N/R targets.
- D=2m: the a^2 term becomes visible. This tests whether the model uses the new coefficient to separate the two operators.
- D=2m+2: the 2ac cross term is available and can be scored independently.

Separately ask the model for the coefficient of x^(2m+2) in 1/f, without supplying the reciprocal output. The exact value is 2ac on the specified family. Score this independently from operator classification so a generic R answer bias cannot pass the coefficient-computation endpoint.

Include other I/N/S roots with matched token lengths and source families; test two-loop native recurrence against single-pass, equal compute, output-erased, early-coefficient-only, prompt-level rule and fixed CAL-only R class bias controls. Require initial source and label independence by generator root, with complete root-pair accounting.

## 5. Scientific gate for a later BPD study

An online BPD donor-transfer experiment requires sufficient clean competence on the new task. Use a genuinely new source distribution and predefined clean-correct donor eligibility. If that gate fails, retain the negative result and allow only source-conditioned continuous margin diagnostics in a distinct experimental registration.

The eventual transfer endpoint must include bidirectional N-to-R and R-to-N edits. A constant R bias can help one direction only; donor-specific success in both directions with equal physical perturbation dose is a stronger causal discriminator. Add random-dark, input-level, task-gradient, online/static mask and full-donor controls; require present-readout parity and delayed task effects.

Call EvalMath's FSI-01 and mechanistic-power preflight with root-level independent units. Eight development roots cannot calibrate the 95th-percentile/95%-confidence matched-sham tolerance threshold. Keep the original 168-case test sealed and do not use any exposed COMP-02/03 outcomes for fresh confirmation.

Reproduce the retrospective numbers from the actual committed model receipts:

    python -m experiments.nan_comp03.posthoc_shortcut --out /tmp/comp03-posthoc.json
    python -m unittest discover -s experiments/nan_comp03/tests -p 'test_posthoc_shortcut.py' -v

This new note is methodological follow-up and an experiment proposal, not a replacement for the previously frozen protocol or a completed NAN-COMP-04 registration.
