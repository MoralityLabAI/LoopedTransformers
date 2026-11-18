# LOOPSCOPE-10C / LGA-07C: global energy-tangent native donor transfer

**PROSPECTIVE source-only registration, 18 November 2026. COMMIT THIS FILE FIRST before implementation, new source generation, fresh model inference or scoring.** This is a separate experiment from LGA-07B, whose negative results on first test 31060 are frozen. No original 31059 test, any ARG-03-N or AMM/CQ protected cohort, or LGA-07B test may be opened for selection.

## The question and exposed-model evidence class

LGA-07B found exact current-head/energy tangent fibers with 37 dimensions per goal token, yet single-token natural donor patch recovered only 64–68% of donor correct labels among clean-competent pairs and a 0.25-native-update dose never reached the registered 15-point donor-transfer gate. We test whether the lost computation is distributed over the entire 81-token physical state. **Full native donor state**, not a single goal token, is an essential positive control: replacing all h_k with naturally reached h_k^donor before the remaining tied recurrence must reproduce the independent clean donor logits.

Freeze original first LGA-02 teacher+SAE source: bundle SHA256 `d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266`, train manifest `7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d`. Freeze first LGA-05 potential: `24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53`, fit manifest `69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d`. Only trained seeds **103,211,307**. They are historically examined teachers, hence every result is labeled **development only** and cannot serve independent model-root scientific confirmation. Do not train/refit a teacher, SAE or potential, and always use the single frozen LGA-05 E_shared with its original k=0 depth embedding.

## Independent fresh source and clocks

The only prospective fresh generator is **`fixed_data(64,31061)`** using the original Torch source generator and its 81 IID digits 1..9; verify 64 unique strings and exclude exact source matches to all previously GENERATED IID LGA sources: 31041/42/43/44/48/49/50/52/53/54/55/56/57/58 and **31060**, with the previously registered sizes. Seed **31059 is independently reserved and must never be generated for this exclusion check**. For each new base and **eight fixed goals (0,10,20,30,40,50,60,70)** construct exactly one source-legal donor by replacing only predecessor digit `(j-1) % 81` via `d→d%9+1`. Its exact valid target is the right cyclic shift of the altered donor input, hence only goal j has a different target. Retain the base ID and all eight paired donors and 81-token targets. Parent source is the independent unit for within-model bootstrap; eight goals and directions are repeated.

The **only primary intervention depth is native k=2 to final trained readout k=4**. Source-recipient and legal source-donor each reach their original h2 using two true tied blocks. Preserve original clean recipient and donor four-pass correctness. No 12-pass extrapolation belongs to this experiment.

## Physically exact global tangent projection without dense 3888² matrices

State H is [81,48]. Let A=W_head be the exact original [10,48] linear current logit Jacobian per token, and g=∇E_shared(H_recipient) be the full whole-state [81,48] derivative of SUM of the 81 scalar potential outputs.

Construct the head-rowspace projector with a stable rank-revealing FP64 SVD of individually normalized nonzero A rows (rank relative tolerance 1e-9, absolute 1e-12). Define P_H=I−A^+A in the **physical 48D channel metric**; apply at every token. Compute U=H_donor−H_recipient, U_0=U P_H, G_0=g P_H. If ||G_0||_F exceeds max(1e-12, 1e-9 ||g||_F), define U_K=U_0−<U_0,G_0>/||G_0||_F² G_0. Otherwise use U_0 and explicitly record energy-row-degenerate status. For full-rank-10 head and nondegenerate energy row, the theoretical current-head-and-energy null dimension is 81*(48−10)−1=**3077**. Record actual rank, dimension and tiny first-order constraint residuals; enforce source-only numerical checks. The analogous **head-only** direction is U_0.

The primary selected edit is U_K normalized to the **natural whole-state donor dose rho=||U||_F**. It is an off-manifold hybrid at a physical radius equal to the natural entire source-donor state change. Generate **eight source-only deterministic isotropic Gaussian nulls** in the SAME global K using independent seeds `LGA07C|sham|teacher|base_index|goal|sham_index` (one per sham), projected head- and energy-tangent, each normalized to exactly rho. Do not inspect task labels or outcomes during selection.

Frozen genuine native arms:
- `global_energy_tangent`: U_K at rho, primary.
- `same_k_sham_0` ... `same_k_sham_7`: K-tangent equal-rho shams, primary null family.
- `global_head_only`: U_0 at rho, isolates effect of further energy tangent constraint.
- `full_donor_state`: exact natural U at natural rho, clean donor physical positive control.
- `goal_token_only`: natural donor Δ at goal token, other 80 tokens unchanged, natural goal-token dose (historical failure geometry).
- `predecessor_goal_pair`: natural donor Δ at positions ((j−1)%81,j), natural pair dose.
- `global_energy_tangent_quarter`: U_K at 0.25*rho, dose-response diagnostic only.

All selected/sham/head-only normalization signs are fixed from legal donor input states before scoring. If source geometry produces zero valid U_K or zero rho, keep the case intention-to-treat with identity edit for primary and all matched-sham arms, recording a fail admission; full-donor and support ablations still run. Report **finite** current all-token head-logit displacement, E_shared(H+edit)−E_shared(H), and also the original local derivative dot(g,edit), physical norm, exact f32 vector bytes/hash. Do not claim finite invariance from infinitesimal tangent.

## Endpoints and preregistered engineering gate

Every teacher has **64 base×8 goals=512** prospective legal source-goal cases (1,536 total). Score all 14 arms plus the clean recipient baseline by advancing **the real original shared block twice** after physical edits, and record every arm's 10-class goal logits, all 81 final token predicted classes, correct donor target, goal-target CE and clean-reference fixed rival margin. Save full input/donor/target roster and physical edit vectors in compact exact float32 receipts with an independently testable SHA. An ineligible source case contributes the clean recipient prediction to all primary selected/sham arms with a paired difference of zero; do not replace or resample cases.

**Primary case contrast:** correct donor goal under `global_energy_tangent` minus the median of the eight same-K sham binary correctness values. First average across the eight fixed goal slots per independent base ID, then obtain the teacher mean, 2,000 pre-keyed random-with-replacement whole-base bootstrap 95% percentile interval. Teacher engineering gate requires (1) at least 80% clean recipient+donor jointly competent goals, (2) full-donor **whole-state** donor-vs-independent-clean-donor original final logits max absolute difference ≤1e-4 for all sources, (3) at least 95% correct donor goals under the full-donor whole-state control among jointly competent cases, (4) ITT mean selected-minus-median-sham correct donor-target rate ≥+0.15, and (5) bootstrap lower 95% bound strictly positive. **DEVELOPMENT_ENGINEERING_POSITIVE** requires ≥2/3 exposed teacher seeds passing every criterion. Otherwise **DEVELOPMENT_NULL_OR_INELIGIBLE**. Report per-root negativity and all case receipts. This is a developmental engineering gate, not a scientific p-value or independent checkpoint inference.

Secondary, all descriptive: head-only-vs-tangent donor correction, goal-only/pair-only natural-dose controls, quarter-radius selected, finite energy and full-head perturbations, projection fraction ||U_K||/||U||, raw donor displacement and signed target margins. Do not promote a later-selected dose, input family or subspace to the primary endpoint. The state-depth and input family are fixed to prevent rescoring a near-ceiling/failed 12-pass teacher.

## Three-stage evidence release and reproducibility

Stage A source-only pretest: run synthetic CPU tests and excluded generator seed **61061**, with freshly initialized untrained TinyRecursiveModel + quadratic energy. Verify all original teacher/SAE/potential byte SHA and source tensor fingerprints with existing original LGA-02/LGA-05 canonical loaders. Emit source_lock.json with this protocol byte SHA, every implementation/auditor/CI source byte SHA, original artifact SHA and explicit `new_test_generated=false`, and upload it **before** generating seed 31061. Stage A does not touch 31061.

Stage B exact first registered test: only after successfully verifying downloaded Stage A source-lock, open new 31061 once, run the fixed entire source model with all 14 edited arms and complete original receipt. Publish first unmodified results and SHA. Execution jobs should run **only on the dedicated original source-release merge commit** via an immutable message gate, rather than running again on docs/report changes or manual workflow dispatch.

Stage C independent stdlib audit: download FIRST source/test artifact, validate original byte SHAs, exact legal donor construction, all arm binary physical vectors and source hash, dose/first-order constraints, retained complete goal logits and all 81 class readouts, correct donor and CE/margin arithmetic, every original 64-whole-base bootstrap and original development verdict. The auditor must import no Torch and must never rerun original model weights. The test is considered independently verified only once this audit job passes.

## EvalMath and follow-up evidence limits

The original power preflight already determined three frozen/exposed teachers cannot support inference over new model initializations. The **15-point developmental effect** follows the preregistered LGA-07B engineer threshold and is not a claim of an estimated 95th-percentile mechanism-variation floor. A scientific follow-up needs new independently trained model roots, calibrating matched shams with at least 59 separate roots when claiming a distribution-free 95/95 tail bound, root-level power and sealed confirmation. New data must never be reused as fresh after first opening. Test legal donor full-state positive control before attributing failure to energy-tangent mechanics. This is a physical Euclidean state projector, not yet a learned BPD factor singleton, compression certificate or evidence that the internal network computes E.
