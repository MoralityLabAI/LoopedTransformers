# NEC-03K: source-frozen angular nulls and retrospective causal replay

**Completed November 21, 2026. Evidence class: source-only geometry calibration and posthoc causal sensitivity on eight previously evaluated NEC-03H-v2 Boolean worlds.** The original PEF-07/NEC-03H-v2 source models, learned TopK16 dictionaries and historical source-ranked/hash-sham results remain unchanged. This stage uses no new model/SAE optimization and opens no reserved/confirmation cohort.

## Frozen algorithm and provenance

The source-only angular null policy is registered in [NEC03K_SOURCE_PROTOCOL.md](NEC03K_SOURCE_PROTOCOL.md) and machine-locked in [NEC03K_SOURCE_LOCK.json](NEC03K_SOURCE_LOCK.json), committed at `1c5e0c950624657df99a34f57be5e59c27083623` **before** any source-calibration outcome. The second-stage retrospective behavioral protocol [NEC03K_DEV_PROTOCOL.md](NEC03K_DEV_PROTOCOL.md) and [NEC03K_DEV_LOCK.json](NEC03K_DEV_LOCK.json) were then committed at `7588b8b640d3a3d328f68e8a1d4e85bdd7951983`, **before** any new angle-matched causal outcome was scored.

The conditional control searches **every disjoint equal-cardinality subset** of the active TopK16 codes not selected by the source-ranked donor, up to C(12,4)=495 candidates per input/visit. The selected control minimizes absolute cosine gap relative to the same frozen hubs-only clean-chosen-class margin gradient in the **32-D post-MLP-down physical basis**. Exactly matching the original n code count and local direct output L2 norm requires no output labels or task success feedback. Both donor vectors are attenuated to their smaller available output norm.

The original 64 confirmation-world commitment SHA256 is **670ea6e1bb4fe86ab9691d0af4800d5623fe64e489ca7555561a80c9739f1f39**. The original 128-world registry digest is **8816304a7db020bd0ad016acca7d9195dfd4633e3316a17bd8e4b8e912fd25b4**.

## Source-training-only angular feasibility

The [successful source calibration Actions run 44892904616](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44892904616) used **the first 16 source-training context groups per old development world** (256 focal configurations), with zero development-evaluation, reserved or confirmation input readouts. Every original native-model/SAE checkpoint byte hash was checked, and an independent pure-stdlib auditor reconstructed the 32-D physical decoder contributions for all 16 active codes and re-enumerated the entire combinatorial control roster on every example.

**8/8 worlds satisfied the precommitted source-null feasibility gate**, requiring at least 80% of site/example pairs to meet absolute cosine gap ≤0.25 with positive matched physical dose, 99% matching within 1% output norm, at least 80% nonzero shared dose, and at least 25% median original source-donor dose retained.

| Old world | Source-site selection | Source angular/physical matching | Median source dose retained |
|---|---|---:|---:|
| dev-008 | Graph | 99.87% | 65.32% |
| dev-009 | Graph | 99.55% | 63.15% |
| dev-010 | Fallback | 100.00% | 42.91% |
| dev-011 | Graph | 98.88% | 64.56% |
| dev-012 | Graph | 99.83% | 55.43% |
| dev-013 | Fallback | 96.88% | 57.72% |
| dev-014 | Graph | 100.00% | 61.55% |
| dev-015 | Graph | 99.61% | 55.20% |

Across the eight worlds, the average source-only site/example angle and physical matching coverage was **99.327%**. Every site/example had positive shared donor dose. The source-calibration independent aggregate is Actions artifact **14703220436**, ZIP SHA256 **d469d9859de81feb351d599828a45d218222b074d7c1cfb5a5da90d92d430e06**, with eight separate source training-case angular trace artifacts on the same run.

## Subsequent causal replay on previously examined development contexts

After the source-only feasibility gate passed and the new causal algorithm was locked, [GitHub Actions run 44893548901](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44893548901) reused the same frozen models on their original **48 previously examined development-evaluation context groups × 16 related focal masks**. Each root had to restore its own pre-existing NEC-03K source feasibility artifact from immutable run 44892904616 before evaluation. The full suite passed its frozen-code tests, eight model and nonlinear-transport replays and the independent pure-standard-library audit. This auditor independently re-enumerated all control subsets, verified exact 16-state teacher labels, historical clean/hubs logits, 32-D direct dose and attenuation gains, source/control final correctness, gradient-margin displacement, nonlinear remainder and full 12-visit memory drift.

The new causal aggregate is Actions artifact **14702553408**, ZIP SHA256 **59d48c36b0c15ff69bd06e7f27a9da15443cdadd28cfaa1f52accd65db3a5542**. Eight original-root artifacts also contain all raw new per-input decoder contribution vectors, selected feature IDs, matched doses, sixteen binary output logits and twelve-visit source/control memory differences.

### Complete eight-world results

The original NEC-03I-B hash-active controls are shown in one column as an earlier reference. Those prior shams had equal local output norm but did **not** match donor angle. The new NEC-03K control is individually reselected conditional on the hubs-only 32-D gradient, and the shared attainable physical norm changes with the available control. Each old teacher world remains a single statistical unit.

| World | Original source gain against four disjoint hash controls /768 | New source gain vs angle-matched control /768 | Angle + physical coverage | Exact nonlinear remainder (mean abs. margin) |
|---|---|---:|---:|---:|
| dev-008 | +7, −7, −9, +5 | **+1** | 99.87% | 0.931 |
| dev-009 | +30, +41, +15, +8 | **+7** | 99.35% | 0.581 |
| dev-010 | +1, −1, −1, 0 | **0** | 99.93% | 0.000 |
| dev-011 | −1, +4, +20, −12 | **+3** | 99.12% | 1.483 |
| dev-012 | 0, +4, −3, +2 | **0** | 99.93% | 0.086 |
| dev-013 | −6, −16, −38, −15 | **+2** | 97.92% | 0.458 |
| dev-014 | −10, +2, −16, −11 | **−3** | 99.85% | 0.189 |
| dev-015 | +1, +4, −1, −3 | **0** | 99.52% | 0.039 |

**All 8/8 control policies remained physically and angularly feasible on the old evaluation contexts.** Seven of the eight worlds passed the original 90% composed-model clean competence rule; dev-010 failed it and contributed ITT zero. The other roots produced signed source-versus-angle-control gains (1,7,3,0,2,−3,0), with the one disabled root (0). The total signed gain was **10** additional correct predictions across eight ×768 cases, for a **descriptive world-level ITT +0.16276 percentage points**.

### Mechanism interpretation

The large negative differences previously seen in the fallback-selected dev-013 task (**−6 to −38** relative to four unconstrained active-feature shams) became **+2** relative to the angularly matched control. The graph-selected dev-009 positive advantage became **+7**, smaller than its prior +8 to +41 range. This is consistent with **directional alignment explaining a substantial part of the previous donor-policy contrast**. It does not prove that local gradient alignment is the unique mediator: the new control is selected on the same input's label-blind hubs-gradient, the active-feature subset and attainable matched physical dose also change, and later nonlinear transport remains substantial. Root 014 retains negative correctness despite meeting the angle/dose controls.

The new root-mean effect (+0.163pp) remains considerably below the original **2pp scientifically meaningful correctness threshold**. The source calibration and old development outcomes are on a repeatedly inspected, compact Boolean-world population. They do not provide independent-world significance, and individual focal masks, source contexts and exhaustive candidate subsets cannot be treated as independent worlds.

## Statistical gate and next world family

**FSI-01 decision: REDESIGN_OR_PILOT_ONLY.** The original NEC-03H-v2 world-mean ITT **+0.423pp** remains unchanged; the NEC-03K +0.163pp is a separately registered *posthoc development* estimand. The fixed 95/95 sham-tail calibration requires **at least 59 eligible independent root worlds**, while these eight already examined worlds are insufficient to calibrate any tail.

To address source-family independence and sample size, a separate **NEC-03L five-input Boolean-world registry** now provides a distinct task-grammar source cohort with 96 prospective development-only roots and 32 execution-embargoed root commitments. All old 64 NEC-03G confirmation worlds, their full-spec commitments and within-world reserved context groups remain untouched. The five-input gym needs its own model-competence and source-native/SAE feasibility pilot before consuming any larger calibration cohort.
