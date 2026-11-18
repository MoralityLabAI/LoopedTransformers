# LRM-REV-01: delayed evidence-revision memory gym

**Protocol/source-only bridge, November 18, 2026.** Mathematical upstream: [Beyond Bayes BB-CQ-05, merge abc89f73](https://github.com/patrickdugan/beyond-bayes/commit/abc89f73d223f00305eb0416a34fe5aee602494d). Evaluation counterpart: [EvalMath EVAL-REV-01](https://github.com/MoralityLabAI/EvalMath). Native model training, probe calibration and heldout scoring are all **future work**; this bridge generates deterministic public engineering worlds and exact reference answers.

## Recurrent test interface

Twelve known-label reports have old match bits `x`. The host selects a correction set `S` of size `j in {1,3,6}` before generating reports and independently of their values. Corrected slots carry fair provisional bits until replaced with iid regime-dependent accepted reports. Priors for three hidden issuer regimes are `(9/20,7/20,1/5)` with match probabilities `(19/20,13/20,3/20)`.

**Early:** expose `S` before encoding, then provide legitimate non-provisional observations followed by individual accepted replacements. A 4-bit count register (plus public stage) is an exact Bayesian updater. **Delayed:** show all old match bits without `S` until encoding is complete; then disclose the schedule and replacement reports, optionally returning a fixed prefix of `r` old report bits. The model must update its posterior across this information change. The exact target after each reveal is the posterior for the observed information, rather than the posterior of the undisclosed full history.

Source-only fixture tokens are dictionaries with an explicit type, index, and value; code that transforms them into model tokens is a model-specific responsibility. The gym carries private `true_regime` for scoring and the full accepted history for full Bayes. Do not present these to the learner unless the arm is explicitly the oracle control.

## Measurable proposed neural experiment

Train or reuse **fresh** recurrent models under two source-disjoint curricula: an early-schedule curriculum and a delayed schedule curriculum. For each, score the complete curve `r=0..j`; compare genuine teacher-forced forward passes with closed-loop reached-state continuation separately. Primary scientific endpoint for a future source-frozen run: per-independent-issuer-root Brier **approximation cost relative to the optimal reduced-information posterior**, with model-normalized forecasts and a 0.01 Brier science effect floor subject to EvalMath independent-root sham resolution. Record explicit full-history and early-count Bayes controls and the stale native model. The registered baseline must see the same information.

For causal mediation, patch a native hidden state after the correction phase using a donor from the same regime but different named-slot history. Match projection rank, physical dose and the full future output Jacobian response where possible. Include wrong-donor, mask-gauge, equal-dose random and zero interventions, and report task-grounded correctness/net Brier change. The number of independent model seeds, the split between training/calibration/development/confirmation, and any family-adjusted statistical threshold must be committed in a **new** protocol before loading model weights or protected roots. Previous LOOPSCOPE-09 12-depth FAIL, LOOPSCOPE-12 original cohorts, SAE and ARG-03 protected tests stay untouched.

## Resource reporting

Keep model weights, hidden precision and layer count separate from runtime live memory. Count the 12 initial old bits, schedule identity, individual replacement messages, verifier archive, provenance proofs, public stage and any persistent output cache. If `S` arrives after compression and all future corrections must be answered exactly without an archive, 12 bits of initial state are necessary in this finite abstraction. If `S` arrives before compression, valid-count state suffices. A 3-bit verified old-count message can recover final count for six corrections, but the external archive still owns the complete old history.

## Source-only acceptance

The deterministic source module and tests require only Python's standard library; no GPU or model checkpoint. They verify exact posterior normalization, full history/verified-count parity, correct posterior changes across reveal stages, balanced synthetic semantics and false old-bit injection detection. They are engineering preflight, **not** an observed neural mechanism.
