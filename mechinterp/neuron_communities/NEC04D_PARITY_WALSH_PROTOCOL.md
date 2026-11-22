# NEC-04D: read-only fifth-order parity spectrum in original native Looper logits

**Scope:** retrospectively inspect only the original NEC04C 32D/12-loop native source and owned calibration-world validation logits. No model inference, training, SAE or causal scoring. Parent [issue #300](https://github.com/MoralityLabAI/LoopedTransformers/issues/300), original completed [NEC-04C #296](https://github.com/MoralityLabAI/LoopedTransformers/issues/296), [merged PR #298](https://github.com/MoralityLabAI/LoopedTransformers/pull/298) and [hosted run 44993859839](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44993859839).

## Frozen historical input and purpose

NEC04C showed that the fixed original 240-step source-parity curriculum plus 660 full-task source updates changed independent calibration-root macro parity validation accuracy by −0.077 percentage points, with only two of six roots improving. The scientific target of this new analysis is **the final-logit parity signal** that can be present below an argmax decision boundary.

Use exactly original NEC04A CALIBRATION root indices **6,7,8,9,10,11**, exactly each previously audited source panel of 80 complete 32-mask context groups and originally owned validation panel of 24 complete context groups, exactly original two model arms (vanilla, parity_curriculum), and exactly original two native checkpoint budgets **240 and 900**. Source oracle artifact ID **14742886436** and original root experiment artifact IDs **14742124547,14743106877,14743221555,14742897690,14743391255,14742099591** belong to completed original run 44993859839.

Before deriving any new spectral measurement, the analyzer verifies the old NEC04A 64-root seed registration and complete source SHA256 against the original NEC04C pretraining ledger, exact code hashes and original training Git commit b6e7a9a2e2b530e738967aa15d37c265643e98b4. It independently reconstructs the original **complete source and owned validation 12-bit history** and scalar Boolean program label, checks all six previously independent native replay receipts, and byte-hashes all **24 actual original checkpoint PT files** against their prevalidation manifests. No new model forward passes occur.

## Exact signal and metrics

For a complete 32-mask focal context with frozen outer-MUX selector=1, let m(x)=logit_1(x)−logit_0(x), where x is five original focal bits. Define for each subset S of focal bit positions

    Walsh(S) = (1/32) sum_{x in {0,1}^5} m(x) (-1)^(popcount(x AND S)).

The target five-focal XOR parity for each original context has per-context sign from its two original context XOR bits. Independently recover the original composed target sign t(x)=2*label(x)−1, and calculate the **target-aligned parity margin** mean_x m(x)*t(x). Its magnitude must equal |Walsh(31)|, and Parseval's identity must reconstruct the original centered focal logit variance.

Record for every original parity context: correct/32, positive class and predicted-positive counts, majority-prediction fraction, mean margin, margin RMS, margin standard deviation over focal masks, target-aligned margin, normalized target-aligned margin to RMS, degree-five energy fraction Walsh(31)^2 / sum_{S != 0} Walsh(S)^2, zero-variance flag, plus Walsh power by degree 1..5. Undefined fractions for zero nonconstant energy are null, with zero-variance counts retained. A perfect parity oracle has **100% of nonconstant energy at degree five**; a constant predictor has zero centered spectral energy; pure lower-degree functions have zero degree-five power. These are mathematical reference checks, not observed model scores.

Derive one root-level mean over its own context panels for each model arm and checkpoint, then macro-average **six independent calibration-world roots equally**. Preserve every original source and owned-validation group and original logits as hash-verified parents. Pooled case totals are descriptive only; correlated masks/contexts cannot substitute for the independent world-root sample unit.

This retrospective final-output measurement does not establish the contents of hidden recurrent representations, the ability of a different optimizer/architecture to learn parity, neuron communities, mediation or source-specific necessity.

## Reproduction and protections

Dedicated workflow .github/workflows/nec04d-parity-walsh.yml uses pure stdlib for the data audit and pytest for tests. It downloads only original NEC04C Actions-run root artifacts and the original pretraining oracle manifest, verifies old independent checkpoints, computes exact Walsh spectra and emits nec04d_original_parity_walsh_audit.json, with original source/checkpoint hashes and root/context-level metrics. It does not open earlier NEC04B or NEC03P world roots.

The 24 *separate* NEC04A developmental WORLD roots, 16 confirmation WORLD roots, 24 reserved context blocks per calibration world, old NEC03Z source groups56–63, NEC03P recruited worlds and original embargo/confirmation cohorts remain untouched. All new model/SAE updates and protected root outcomes remain zero. **FSI-01: REDESIGN_OR_PILOT_ONLY**.
