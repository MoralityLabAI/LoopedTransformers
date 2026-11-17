# ROC-05B: physically matched source-rule and hidden-state causal interventions

**Prospectively frozen for causal measurements in configuration commit ef18aef009e72c0b0e75de9331d6135db7977d98 (17 November 2026).** ROC-05A development answer accuracies were available before this protocol, so ROC-05B is a development-family mechanism diagnostic. No ROC-05B causal interventions were performed when the configuration was committed.

## Model/source lock

Use exactly the twelve original ROC-05A first-run 1,920-step model checkpoints from hosted training run 44101721734. The configuration contains SHA-256 hashes for all twelve files. Independently run the original model and train/development source hash checks before any analysis. The repeated later CPU training produced different checkpoint bytes for 10/12 files; **ROC-05B uses only the first-run checkpoint bytes**, with no optimization or model replacement. The source inputs are the already exposed 256 development cases and the train-only examples required for observer fitting. ROC-05's 384 reserved table identities remain ungenerated test examples; ROC-02's original source families 6/7 also remain sealed.

## Fixed observer and output horizon

The original model reads the early A-rule binding and all B/C distractors while masking out the final visible query event from recurrent memory updates. Form the midpoint after event processing and four of the original eight tied residual steps; after perturbation, perform exactly the remaining four residual steps. Select the final visible query's row from the *original frozen 8×8 table head*. Compare unperturbed staged logits with native model.forward, using a maximum absolute difference of 1e-5.

Construct the primary rank-seven task observer from the **1,024 train-only midpoint states**: top-seven left singular vectors of centered state × centered correct-label one-hot cross-covariance. The eighth supervised singular direction is noncanonical for centered eight-class labels and excluded. A separately trained PCA rank-seven observer is descriptive. All interventions retain frozen original model logits, never a refitted observer output head.

On exposed development shadow cases at fixed indices 0 and 1, measure T = derivative of the four-step suffix, computed-change D=T-I, observer subspace projector P, A=U-transpose D U, B=U-transpose D (I-P), C=(I-P) D U, and H=(R_q T)-transpose (R_q T), where R_q selects the native head row for the fixed query. Report all raw Frobenius norms and full-D-normalized ratios and the fraction of future-logit Gram captured. Use exactly 128 seeded rank-matched random QR subspaces per state and six one-sided add-one rank tests with Holm correction, preserving ROC-01 Amendment C's effect thresholds and abstentions. These are local diagnostics with nonexclusive labels.

## Source change and equal-radius interventions

For each original exposed development record with true class y and final query index q, swap the initial A binding table outputs at q and (q+4)%8, preserving all intermediate events and the final query. The independent symbolic interpreter must return a different new target yprime. Compare natural-dose full legal source input edits with hidden interventions at radii 0.05, 0.10 and **primary 0.20** in L2 of the same original midpoint hidden state.

The matched hidden directions are source-edit difference projected onto the supervised rank-seven observer, its dark complement, train-only PCA rank-seven observed subspace, and the full input-induced change, together with one SHA-keyed random rank-seven observed direction and one seeded 128-dimensional isotropic direction. Every valid hidden direction is normalized to the registered radius; invalid projections are explicit abstentions.

For the equal-cost input comparator, continuously interpolate only the initial A binding features between the original and legal edited inputs. On a fixed 64-cell interpolation grid, locate the first crossing of the target midpoint L2 radius, then perform 24 bisection iterations. Intermediate feature vectors are off the legal discrete symbolic task manifold and carry no independently valid rule label. If the registered radius is unreachable, record the case as ineligible. The same original midpoint, remaining four loops, and original frozen table-head/query route apply to every physical arm.

The primary signed output endpoint is the change in new-target versus old-target margin under the original model: gain = (post_logit[new]-post_logit[old])-(base_logit[new]-base_logit[old]). The legal counterfactual oracle establishes the intended new target, and correctness/argmax changes are reported separately from margin changes.

## Primary gate, limits and provenance

The primary high-512 contrast compares the per-seed average rank-seven observed source-steering gain with the equal-radius softened-input gain at radius 0.20, using only cases with both routes eligible. Require at least 80% eligible cases for **each** of the six original high-512 model seeds, otherwise record ABSTAIN. A positive mechanism diagnostic additionally requires five of six paired seed differences positive, mean difference positive and one-sided exact 64-sign sign-flip p at most 0.05. The original low-four models, other radii and source-nuisance B/C controls remain descriptive.

This is a predefined development-family causal check, conducted after the ROC-05A training accuracy outcomes were available. A positive result cannot establish independent new-rule transfer, a five-type mechanism identification or a closed dynamical quotient. Publish every per-case arm/validity result, six-seed effect vector, source/checkpoint SHA, computed-change null card, and the full independent audit. No model updates, ROC-05 reserved examples or original ROC-02 family-6/7 examples are permitted.
