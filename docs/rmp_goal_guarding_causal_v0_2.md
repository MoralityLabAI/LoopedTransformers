# RMP-GG-02: rank-constrained causal interchange in a learned recurrent belief state

**Protocol frozen on 2026-11-11 before any GG-02 training, rank selection, or held-out evaluation.** This follows the verified and merged [GG-01 supervised positive control](../reports/checkpoint_rmp_gg01_causal_belief_20261111.md), with source model on LoopedTransformers \`main\` at \`dfac2655512cf18be10cf1489973ffbcc0bc0b95\`. The exact Bayesian task and action oracle are from [EvalMath RHT-20–25](https://github.com/MoralityLabAI/EvalMath/pull/29). GG-01's rank-one fitted-belief intervention changed 0/256 counterfactual decisions on each seed despite near-perfect linear belief decodability **even at random initialization**.

## Scientific question

Can a *small causally trained orthogonal subspace* transfer the prefix evidence needed for a receiver-suffix decision, under a locked experiment whose heldout target is the exact counterfactual action? If the rank-one decoder of log posterior odds fails, the operative neural state might store the computation in a higher-dimensional or nonlinear representation. A discovered subspace would be a bounded **task mechanism**, not evidence of spontaneous evaluation awareness or actual gradient monitoring.

The recurrent network is unchanged from GG-01: two-position sequence per evidence visit, native tied \`ExplicitDecoderBlock\`, width 32, four attention heads, evidence token read at each visit, recurrent memory token. Freeze seeds **11 and 17**, 400 AdamW updates, batch 128, GG-01's unchanged finite observation generator and prefix supervision, and keep all final weights regardless of performance. Compare hashes of initial/final weights and retain C0 readouts.

## Data and split

- Train the base GG-01 model anew using the exact already-registered settings. For each seed use *one fixed* base-model checkpoint across all GG-02 interventions. The model has no GG-02 oracle labels during base training.
- Construct three *disjoint* oracle-manifest sets of source/receiver pairs with **6 observations per sequence**, cut at visit 2: calibration **256** pairs, validation **128** pairs, locked reporting **256** pairs. A pair contains one shared prior and reward ratio, receiver bits \(\mathbf r\), donor bits \(\mathbf d\), hybrid bits \((d_1,d_2,r_3,\ldots,r_6)\). Require hybrid's exact action to disagree with both receiver's and donor's exact full-sequence action. Deduplicate complete \((p,R/G,\mathbf r,\mathbf d)\) keys across all sets. The fixed RNG seeds are **31001**, **31002**, and **31003**, with globally deterministic collision avoidance; lock reports until fitting and hyperparameter selection finish.
- Eligibility further requires frozen model correctness on receiver and hybrid. Report both unconditional 256-pair coverage and eligible-only rates. If fewer than 32 locked eligible pairs, verdict \`INELIGIBLE\`.
- All calibration labels are exact independently computed binary Bayesian counterfactual decisions; contexts and suffix bit tokens remain from receiver during every intervention.

## Interventions

For a receiver state \(h_r\), donor-prefix state \(h_d\), and trainable orthonormal \(\mathbb R^{32\times k}\) basis \(U\), form

\[
h_{\rm patch}=h_r+UU^\top(h_d-h_r),\qquad k\in\{2,4,8\}.
\]

At visit 2 of the **frozen** native network, insert \(h_{\rm patch}\) and run the original receiver suffix. Train only \(U\). Use a QR parameterization, Adam LR **0.025**, **120 steps**, batches of **64**, fixed initialization seed \(41000+1000\,s+k\), no weight decay, and minimize \(\mathrm{CE}(f_{\rm patched},y_{\rm hybrid})+0.01\|UU^\top(h_d-h_r)\|_2^2\). No base-model gradients are updated.

- Probe rank-one GG-01 control fitted on **its own distinct calibration set** (GG-01's \`fit_belief_probe\` protocol). Also report the full donor-prefix state replacement and receiver-state sham.
- Calibration-only PCA subspace at each rank: top eigenvectors of donor-minus-receiver covariance; use its projection for a matched rank-\(k\) control.
- Eight unit Gaussian perturbations, seeded 66713, scaled per example to the learned projected displacement's L2 norm, scored on exactly the same receiver states.
- Full-state interchange must match the explicit hybrid continuation from visit 2 onward within absolute and relative tolerance \(3\cdot10^{-5}\). A mismatch stops the experiment.
- Record for each rank: exact hybrid-action hit rate, hybrid-target logit margin lift from receiver, mean patch norm, 99th percentile and maximum patched-to-clean norm ratio, mean cosine of patched state to nearest clean validation state, and share passing the *calibration-derived* 99th-percentile clean norm ceiling. Model correctness on hybrid and receiver is an *eligibility* condition, never counted as a patch success itself.

## Selection and acceptance

Rank \(k\) is **selected solely on the 128-pair validation set** by highest hybrid hit rate, with lower rank breaking ties, then highest hybrid-target mean logit-margin lift. Frozen selected \(k\) is run on the 256-pair locked reporting set. Report every rank's locked results as descriptive, including the chosen rank. If all rank fits are ineffective, report a negative result.

Causal support requires for *both seeds* on the frozen locked set: selected-rank hybrid hit rate \(\ge 0.50\), at least **+0.10** above sham and **+0.05** above each of eight matched-norm random controls, **+0.05** above the same-rank PCA control, at least **32** eligible pairs and at least **95%** of patches inside the calibration-derived clean-state norm ceiling. An accurate decoder, raw margin change, or full-state identity by itself cannot pass the gate.

An additional counterfactual latent-state sanity test uses an incorrect donor whose prefix matches the receiver. Its state delta is zero, making the patch exactly sham; the implementation asserts this identity before training.

## Calibration policy, provenance, and limits

Report C0, final GG-01 checkpoint hash, exact panel-identifiers, base-model evaluation accuracy, neural training duration, source revision, all rank training curves, and separate calibration/validation/locked endpoints. Training and validation can be repeated for implementation debugging. Any source change after reading locked results must be identified as an exploratory follow-up and cannot replace the primary locked run.

The small 6-bit channel and finite prior/reward grid mean these are **heldout source/donor pairs**, not unseen real-world skills or text distribution shifts. Cross-validation is across new combinations on the same task grammar. No attestor or real optimizer state is observed, no actual reward hacking is induced, and no conclusions about hidden Mythos-like strategic awareness are supported. A separate future GG-03 must randomize real post-training updates, training disclosures, and independently sealed gold targets.
