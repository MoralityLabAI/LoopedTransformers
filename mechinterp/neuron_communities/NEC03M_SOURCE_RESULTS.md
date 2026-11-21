# NEC-03M: First new five-input source-only competence and directional-SAE feasibility

**November 21, 2026. Evidence class: source-training-only, in-sample engineering feasibility.** This stage uses only the first four of the 96 registered five-input development worlds from NEC-03L: dev5-000..003. It trains no embargoed or confirmation models and scores no development-evaluation or reserved contexts.

## Frozen registration and provenance

The full source-only training, SAE/angle instrumentation, independent root-auditing code and thresholds were committed together with the CPU workflow at **`1560c7136db16b80f27eabe91ed7e7e30c9c9442`**, before any new native models were trained.

The source worlds and teacher labels are from the independently hashed [NEC-03L registration](NEC03L_REGISTRATION_RESULTS.md), new registry SHA256 **`6777d478f6e988e3f57f8882dfef2e84898f765b7adf7e142a914b4cbdcd1217`**. The original NEC-03G 64-world confirmation-source commitment SHA256 **`670ea6e1bb4fe86ab9691d0af4800d5623fe64e489ca7555561a80c9739f1f39`** remained unchanged.

**[Hosted CPU GitHub Actions run 44894980345](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44894980345) passed completely**, including source protocol and 32-state focal grouping unit tests, four independent two-teacher model/SAE source-only training jobs, independent root-level byte/hash/label/rank/physical-audits, and a pure-stdlib aggregate. Every one of the **eight newly trained 900-update PEF-style 12-tied-visit native models** was reloaded and scored independently on its complete 2,560-history *training-source input set*. Each source-fit TopK16 checkpoint, reconstruction MSE, normalization and attribution rank was independently reconstructed on the original disjoint 40+40 training groups.

Final aggregate: GitHub Actions artifact **`nec03m-four-root-5-input-native-sae-source-feasibility`**, ID **14702694906**, ZIP SHA256 **`e6c6c692c4a9240cf2be17e4e1f442831204145f05834ef24e0a2218829f656d`**. The same run archives all four distinct root-level native model, SAE and raw source geometry files with their individual checkpoint SHA256s.

## Four fresh source-world results

All accuracy values below measure **in-sample classification of already-trained source contexts** and do not estimate generalization. The fixed native competence rule is at least **90%**, the source SAE normalized reconstruction MSE floor is **0.03**, and the source direction null gate includes cosine gap ≤0.25, at least 80% positive matched angular/physical samples and at least 25% median retained left donor dose.

| Five-input world | Composed source correct/2,560 | Additive source correct/2,560 | Composed SAE MSE | Additive SAE MSE | Source site kinds (composed/additive) | Whole root source-ready |
|---|---:|---:|---:|---:|---|---|
| dev5-000 | **2,242 (87.58%)** | 2,553 (99.73%) | 0.008142 | 0.007370 | graph / graph | No: composed below native floor |
| dev5-001 | **2,560 (100%)** | 2,560 (100%) | 0.008821 | 0.005895 | graph / graph | **Yes** |
| dev5-002 | **2,474 (96.64%)** | 2,551 (99.65%) | 0.010150 | 0.007678 | graph / graph | **Yes** |
| dev5-003 | **1,567 (61.21%)** | 2,560 (100%) | 0.017362 | 0.010199 | graph / fallback | No: composed below native floor |

**6/8 native models** met the in-sample 90% threshold. All **8/8** 128-code/TopK16 learned SAE checkpoints were fitted to source groups, reproduced by an independent reconstruction check and ranked on a separate set of training contexts. Every SAE normalized reconstruction MSE was **0.005895–0.017362**, inside the fixed **0.03** engineering floor. All eight source-model arms had valid native GELU source sites, seven graph-selected and one deterministic source fallback.

**All 8/8 source-trained SAE model arms** passed the directional null policy's physical and angular feasibility gates. Relative to the hubs-only clean-chosen margin gradient, per-site/example source angle matching covered between **98.37% and 100%** of the original training-source active-code cases. The independent auditor re-enumerated each possible disjoint same-n control set using all sixteen active 32D decoder contributions, verifying it minimized the registered absolute cosine gap and that physical donor output norms were matched by attenuation.

The study therefore exposes a **source model competence bottleneck** under the fixed 900-step training recipe on some harder five-input truth functions. The TopK16 fitting and angle-matched source-control construction were feasible on all eight model arms, including the composed teachers that failed to learn sufficiently. A source SAE's low reconstruction MSE is not evidence that the *native teacher is competent*.

## Scientific hard stop and next work

This pilot scored **zero new development-evaluation context groups**, **zero within-world reserved contexts**, **zero of the 32 five-input execution-embargoed roots**, and **zero of the original 64 four-input confirmation worlds**. Its 40/40 source split for fitting and ranking is new and requires a separate prospective causal study before drawing any behavioral inference.

**2/4 complete teacher worlds** satisfied all current source-model gates. Expanding this 900-step regimen to 59+ independent new task roots immediately risks substantial task-learning attrition; the original NEC-03H-v2 +0.423pp and NEC-03K +0.16276pp posthoc world-level effects remain well below their old +2pp meaningful-change floor. The matched-sham 95/95 independent-world tail is still uncalibrated and FSI-01 stays **`REDESIGN_OR_PILOT_ONLY`**.

The next *development-only* calibration should rerun these **already trained-source-only first four worlds** using a single longer fixed native optimizer schedule or source-only curriculum, from the original model seeds with complete optimizer reconstruction. It should compare in-sample native competence against the immutable 900-step receipts and preserve the two failed learners. Any subsequent new-world study must freeze the new schedule and source-native/SAE donor policy before opening fresh development-evaluation roots. No confirmation unlock is authorized.
