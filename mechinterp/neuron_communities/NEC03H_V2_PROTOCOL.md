# NEC-03H-v2 | Locked follow-up on untouched development worlds dev-008..015

**Registered on 2026-11-21 after the opened-root 900-step calibration and before any v2 runs on dev-008..015.** Scientific stage: new independent development roots inside the existing NEC-03G Boolean grammar. This supersedes NEC-03G's *development method* and keeps its previously executed eight-root ITT zero and all historical hashes unchanged.

The earlier [NEC-03G negative result](NEC03G_RESULTS.md) trained 16 new native checkpoints in eight distinct Boolean/position worlds and admitted 0/8 roots under its original simultaneous composed/additive/source-community/physical-dose gates. A subsequent, separately documented [NEC-03H opened-root calibration](NEC03H_CALIBRATION.md) reinitialized all 16 original models from their initial training seeds and trained 900 steps under the same optimizer/data stream. Clean models reaching >=90% rose from **12/16** to **15/16**. A source-only fallback supplied sites for four extra models; one independently trained additive teacher remained without a source site. Calibration used only the previously opened dev-000..007 worlds and cannot count as confirmation.

## Population and inviolable source splits

The original 128-world registry remains fixed, SHA256 `8816304a7db020bd0ad016acca7d9195dfd4633e3316a17bd8e4b8e912fd25b4`. **New v2 measured development worlds are exactly dev-008 to dev-015.** Each has its own four-input essential Boolean truth function, four distinct focal-position indices, and a newly seeded native model. The additive control has the same 16-state class prior through a binary 1,2,4,8 weighted-threshold teacher. One such world with its independently trained composed/additive checkpoints is the unit of statistical analysis.

Each world SHA256-sorts all 256 possible contexts of the other eight bit positions into 160 training, 48 development evaluation and **48 reserved**. Every context contains 16 related focal-bit counterfactuals. Neither the 48 reserved groups nor any 64 confirmation-world checkpoints/outcomes may be opened by v2. All 64 confirmation-world specs remain committed by SHA256 hash `670ea6e1bb4fe86ab9691d0af4800d5623fe64e489ca7555561a80c9739f1f39`. The confirmation workflow does not exist.

## Fixed native model, source selection and TopK policy

Train **exactly 900 AdamW CPU updates** for both the composed and additive teacher, freshly initialized from each world's original fixed seeds. Optimizer: LR 0.003, no weight decay, batch 128, gradient clip norm 1.0, the NEC-03G deterministic per-seed sample-index stream. A shortened training smoke may run only in already-opened worlds and is excluded from the new-root power study. The 900-step models receive new immutable byte hashes and cannot be substituted for the 450-step checkpoints.

Source-only native selection uses the first 80 complete parameter-training context quartets. It computes top-two memory-token MLP GELU activation × logit-gradient attribution on each of 12 recurrent visits, removes >=90%-frequency hubs, and applies seeded Louvain to source-only positive-phi co-attribution edges. Choose the highest-ranked qualifying graph community of size 2–12 with at most eight visit:neuron occurrences per visit. When no graph community qualifies, fall back to **two source-frequency-ranked non-hub native neurons at each of the third and fourth chronological focal-input positions**, with source top-attribution frequency >=2/80. The fallback is trained per model on its own source activations and is flagged separately from graph discovery. If four sites cannot be selected, the source candidate remains absent and the composed primary is scored zero under ITT.

For each model that has a source candidate, fit a 128-feature **TopK=16** SAE to its native 64-dimensional GELU memory-token activations using the first 80 parameter-training context quartets, with exactly **480 optimizer updates**. A separate source-only 80-context cohort chooses a rank-order of 128 SAE features for each recurrent visit via activation × downstream margin-gradient. Freeze checkpoint/decoder/normalization and source-only ranks by SHA256 before any development-context readout.

At evaluation time, route the strongest source-ranked *currently active* TopK codes to the candidate's native-neuron visit sites, with the same per-visit component count. The alternative picks equally many least-ranked active codes. Candidate identities are input-conditioned while ranking, codebook and budget remain fixed. For the additional native sham, select source-frequency-matched hub-disjoint native neurons at the same visits. Every comparison has the same hubs-only recurrent GELU baseline, direct frozen `mlp_down` L2 output dose equalized to the **smaller attainable donor dose**, and no amplification, with raw per-input gain/dose and all 16-case predictions retained.

## V2 primary and auxiliary falsifications

The **primary fixed comparison** is source-ranked active TopK versus equally active lower-ranked TopK on the composed Boolean teacher. Its effect is the signed difference in 768 development-case correctness fractions, grouped as 48 complete eight-bit contexts, and registered *once per independently generated world*. All candidate pairs are measured regardless of outcome.

The composed-teacher primary is eligible only when:

- The composed native head reaches >=90% clean development accuracy.
- A source-only graph or deterministic fallback candidate exists.
- The composed TopK-versus-alternative intervention has at least 99% of per-site/example pairs within 1% direct physical `mlp_down` output L2 dose, >=80% sites with positive shared physical dose and >=25% median retained dose of the original left donor.

A failed composed primary contributes **zero** under ITT, remains in the root denominator, and retains the complete reason ledger. Additive-teacher competency/source availability and native-vs-native sham feasibility are separately measured falsification controls; they **do not invalidate** an otherwise eligible composed primary.

For statistical preflight, the additive teacher's same-dose TopK contrast and the composed native-vs-native matched sham contrast are the two registered matched-null sources. When a control lacks a clean competent head, source site or adequate physical dose, the missing sham is assigned conservative **absolute bound 1.0**, thereby making its null-variation floor uncalibrated or unresolvable. A missing sham is never imputed as zero. This may still force a REDESIGN verdict, which is scientifically appropriate. Raw outcomes are kept even when the controls are underpowered.

Three registered comparison families share familywise alpha **0.05**, with provisional scientific minimum meaningful behavioral gain **2 percentage points**, planning target **4 percentage points**, and one 768-case within-world discrete measurement resolution. Native/SAE topologic or correctness behavior of one source world does not establish independent teacher-world transfer.

## Strict chronology, locking and power

The code/thresholds in this protocol and a separate JSON lock list the exact repository Git blob identities, full parameters, and the original world-registry/confirmation commitments. They must be **committed and pass source-only unit/CI verification before** the first new dev-008..015 training job starts. Every new world must store the literal lock SHA256 in its root receipt, and the independent auditor verifies it.

Once the eight previously unexamined development roots are run, an independent auditor checks all native checkpoints, source-only selection and TopK weights, 16-bit truth-table teacher labels, raw 48×16-context paired outputs and donor doses. A separate pure-standard-library job gathers exactly eight distinct world-level effects and passes them to EvalMath's [mechinterp power preflight](https://github.com/MoralityLabAI/EvalMath/tree/main/skills/evalmath-mechinterp-power) at pinned commit `9708f1edaf1cb25904b88b07c3f23c1899df7cac`.

The statistical 95/95 upper sham quantile requires at least **59 independent sham-world roots** even when using their sample maximum. Eight new development roots cannot calibrate that tail, so the locked v2 runner must report **REDESIGN_OR_PILOT_ONLY** and retain the full signed world-level results even when it finds a positive development effect. The shared Boolean-program grammar and PEF recurrent architecture also limit claims of generalization. Further independent development null-root calibration requires a separately approved allocation of dev-016..063 (48 more roots), and at most 56 distinct v2 development roots would then have been sampled. The original 8 NEC-03G roots belong to a different registered intervention/training procedure; they cannot be pooled casually with v2 shams to reach 59.

**Confirmation hard stop:** all 64 confirmation worlds and every reserved 48-group within-world split remain untouched. There is no readiness claim until sufficient independent-world resolution, a fully frozen hypothesis/correction family and separately authorized confirmation workflow are demonstrated. This project provides a repeatable compact mechanistic instrument; it does not assert a result about language models, self-models or RSI.
