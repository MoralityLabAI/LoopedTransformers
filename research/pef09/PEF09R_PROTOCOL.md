# PEF-09R | Conditional recovery after ungated global-axis ablation

Registered on 2026-11-15 **after** reading the PEF-09 heldout ablation results, which showed PEF-09's original pair-selectivity criterion failed and its unconditioned rank-one ablation collapsed native accuracy to the majority-class baseline. This is a clearly distinct follow-up with new decisions frozen before its heldout scores. It cannot make PEF-09's preregistered criterion positive.

## Frozen materials

Use the same six PEF-07 450-update checkpoint files, verified by individual bytes and weight SHA256 against the original ledger; same context-disjoint 160/48/48 split. Freeze and verify the pre-existing PEF-09 selection ledger as an input, including every pair direction, global PC, center, and task/seed/visit identity. The PEF-09 selection ledger SHA256 must be `f206d88f5f2ea9703edc385d47bcf6fd6cd11a6cb4a3ece6edeae230dd734157` for the local preregistered PEF-09 run. The selection ledger uses only PEF-07 probe contexts; the same 768 probe examples fit all new readouts. The 768 heldout examples are reserved solely for final evaluation.

## Readout recovery question

The original native h12 head predicts the constant majority class after PEF-09's t=8 and t=12 rank-one intervention, while frozen h14 typically also loses predictive performance. Does a linear readout learned from *post-ablation* probe states recover useful task signal in the orthogonal complement?

Use the exact PEF-07 `fit_ridge` with standardized features, intercept, penalty 1.0 and threshold 0.5. For each seed and task, at each visit t in {8,12}, consider the target pair (0,1) rank-one direction, target pair (2,3), and global-PC direction obtained from the **frozen PEF-09 probe selection ledger**. Apply the already-defined ungated affine projection to all 768 probe states and all 768 heldout states, then replay every remaining natural bit to h12 and the original continuation bits (0,1) to h14. Fit a new ridge for h12 and a separate new ridge for h14 on the ablated **probe only** states. Score the unmodified native h12 head and the newly fitted h12 and h14 ridge decoders on the ablated heldout. Also measure the native h12 baseline and the already fitted h14 PEF-09 baseline. Every probe fit precedes heldout scoring; the same two fixed ridge penalties apply to all experiments.

As a sanity comparison, also fit h12 and h14 ridges on untouched natural probe states, then evaluate natural heldout. Preserve the original task labels and context partitions throughout.

## Rank sweep and geometry

For each model, calculate the full PCA spectrum of the centered natural h12 probe state [768,32], its total energy, top-1 variance fraction, top-2 fraction, top-4 fraction and effective rank `(tr Σ)^2/tr(Σ²)`. Also compute absolute probe cosine similarity among the two pair directions and global PC at h8 and h12, and the PEF-09 random-control-to-target projection-energy ratio range. These are probe-only descriptive diagnostics.

Rank sweep: use the same h12 probe PCA basis, and remove its top k directions for k={1,2,4,8} from *every* natural h12 probe/heldout state with one identical ungated affine projector. Fit the same ridge to the resulting probe states and evaluate heldout h12. No rank, penalty, or readout architecture is selected using heldout results. The rank sweep measures accessibility in subspaces, **not** whether an existing h12 head can use them without re-training.

## Registered decision

For each coalition seed, require BOTH pair-derived directions at **visit 8** to yield >=576/768 heldout accuracy under the newly fitted **h12 residual ridge**. If at least 2 of 3 coalition seeds meet both, label `residual_linear_recovery_criterion_met`; otherwise label `criterion_not_met`. Report every threshold arm, h14 recovery, frozen native result, global PC control, rank sweep, original unablated ridge and all seed-specific score counts. This recovery test is conditional on PEF-09's already observed global bottleneck.

## Interpretation

A successful newly trained ridge readout after frozen ablation demonstrates that label information remains accessible to a linear downstream decoder for this task, even though the originally trained head fails. It does not imply the model's own frozen head would recover, that the network has a unique coalition-specific circuit, or that retraining a classifier is free of cost. The same four-bit teacher is shared across contexts, so the heldout source generalization is across contexts rather than unseen Boolean operators. A negative result means this particular ridge does not recover on these frozen features and does not certify information-theoretic erasure.
