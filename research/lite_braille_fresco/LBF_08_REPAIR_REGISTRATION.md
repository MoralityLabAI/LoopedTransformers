# LBF-08: Native model-visited verifier repair and action-set supervision

**Registered:** 2026-11-23 before LBF-08 CPU training or native scoring. **Status:** exploratory, source-only prospective development run. The preceding LBF-05 through LBF-07 calibration cohorts and their negative mechanism gates are exposed development results.

## Research question

Does training on genuine **model-visited Lean proof states**, including failed-tactic histories and accepted nonterminal states, improve whole-proof closure over the existing shortest-reference-path-only learner at matched model and Lean costs? Does retaining every native shortest-continuation tactic outperform a one-positive imitation target on the *same model-visited states*?

Three matched arms:

| Arm | Predecision source | Supervision |
| --- | --- | --- |
| source_only | Original native-compiled shortest-reference-path packets | Full native shortest-action set |
| visited_one | 50% reference path, 50% genuine model-visited states if available | First (public-menu-order) optimal native action on model-visited frames |
| visited_set | Identical packet roster to visited_one | Every equally shortest native action on the same model-visited frames |

All arms receive identical seed-specific parameter initialization, model architecture, 720 FP32 AdamW steps, source-lineage sampling and maximum input/compute allowance. A deterministic draw roster is frozen from the source teacher and visitation ledger before training. Where no positive reached-state continuation is available for a sampled root, use a source frame and count the fallback. Do not silently drop roots from scoring.

### Evidence clocks and authority

1. Source teacher: Lean 4.19 independently compiles 48 train-only theorem roots and shortest reference paths. This source is the same kind as LSA-02B.
2. Frozen behavior checkpoint: the registered **seed-107, width-32, 384-byte, AB, hybrid_rule_intact, depth-2, 360-step** model trained exclusively on those train roots. Freeze its checkpoint SHA-256 before collection.
3. Train-only visitation: run that model with the real Lean 4.19 tactic stepper for eight native tactic attempts per theorem root. Record every attempted tactic's **predecision public packet**, typed candidate menu, exact native accepted/rejected/closed outcome, and state transitions. Re-execute every episode with no model and require an exact state/action replay. Never select a theorem, source candidate or decision based on calibration/held-out outcomes.
4. Source-authoritative action-set labels: after the behavior model finishes the theorem, independently explore its native Lean AND/OR graph using the existing collector. For each model-visited predecision goal, label a public tactic positive only if its native successor has minimum finite proof cost. Multiple equal-cost actions are all positive. Zero means *not a shortest continuation*, not necessarily an immediately invalid tactic. Reachability failures, missing rows and censored outcomes are saved in a separate ledger, not converted into false binary supervision. Frame labels are evaluator/trainer-side only.
5. Optional *teacher-forced recovery*: starting from the model's accepted native tactic prefix, take only native graph shortest continuations to a kernel-complete root proof, then independently compile the **whole** script. A forced recovery is **not autonomous model proof completion**. Distinguish originally rejected tactics, accepted nonterminal transitions, model-autonomous closures, reachable forced recoveries, unrecoverable states, and scripts beyond the model's eight-call allowance. No model-facing future proof receipt is exported.
6. Calibration: independently evaluate the three trained arms on the **same 32 disjoint calibration theorem roots**, retaining all attempts and failures, using exactly eight Lean tactic checks per root, depth two, and independent whole-proof standalone compilation plus model-free replay. Ordinary-test and OOD remain sealed.

### Source split / cost registration

- New G03 suite seed **20261129**, 64 roots per shard; native shortest-source compiler must pass **48/48** train roots. Do not substitute any failed source root.
- Behavior source 48 train roots, seed **107**, 360 CPU training updates at LR .001, Plan B width 32, four attention heads, eight typed slots, two recurrent passes and a 384-byte byte encoder. Use the registered public-only hybrid_rule_intact view; readable PP suffix is filled with inert bytes. Freeze checkpoint bytes before visitation, and record source manifest, train-step logs and checkpoint SHA-256.
- A bounded eight-tactic behavior rollout is performed on each of the 48 roots. Model actions may fail, open subgoals, or kernel-close. A failed attempt remains part of the training observation's history; an accepted nonterminal state remains its real successor. Preserve original episode records, full attempt count, SHA-256 and model-free replay check.
- Per source root, collect **at most 8 model-visited training frames** after the first model decision, each tied to exact model-visited candidate/packet hashes and source native shortest-graph labels. Keep all resulting frames/labels (including censored) in a frozen ledger; fitting consumes only states with a finite witnessed shortest continuation. Require at least **12 independent train roots** with witnessed model-visited states for a meaningfully powered development pilot. If fewer are witnessed, stop at **INSUFFICIENT_REPAIR_SOURCE**, retain all receipts, and do not compensate by cherry-picking roots.
- Train seeds **107 and 109**, one GitHub Actions CPU worker, one deterministic PyTorch thread, 720 steps per arm, recurrent depth two, fixed AB weight-tied block order, same 57,475 parameters, same 48 train families, and no optimizer updates from calibration/ordinary/OOD. All six fits run with identical source-root selection; visited_one and visited_set also have exactly matching model-visited packet selection. Repeat the seed-107 source-only fit and compare named tensors and draw logs for exact same-host determinism.
- The **50% visit exposure** is a nominal attempt schedule. Report actual number of optimizer updates using non-reference model-visited states, distinct source roots reached, positive-candidate set sizes, rejected histories, accepted nonterminals, certified autonomous recoveries, forced oracle-assisted proof completions, and censored cases. Every downstream claim must distinguish training-source native authority from new end-to-end model behavior.

## Outcomes and decision gates

Primary: **root-paired native kernel-certified whole-proof completion at eight tactic calls**, seed-averaged within each independent theorem root, **visited_set minus source_only**. A useful development gain is **+0.05 absolute closure rate**, registered before evaluation. Secondary: **visited_set minus visited_one**, with +0.025 exploratory improvement floor. Report descriptive 2,000-draw root-cluster bootstrap intervals (seed 20261123), wins/losses/ties, action checks, model forwards, candidate menu coverage, effective visit exposure, terminal certificates and all roots with no proof.

Observe model-visited repair opportunity and count recovered proof paths **separately**. Native teacher-forced continuations are training/oracle evidence, never evaluation model success. Full Lean proof credit is granted only after an independent standalone certificate succeeds for the model-selected script.

**Competence gate** remains unchanged: at least 32 eligible calibration roots, >=40% trained-order certified close rate, >=10 points over an exact independently registered candidate-uniform native baseline, plus matched sham/power gate. The exact native uniform-menu baseline is not part of this run; automatic scientific promotion remains blocked. Lack of evidence is reported as negative or pilot-only with all denominators.

A result favoring visited_set over visited_one would support multi-optimal action supervision on model-visited recovery states. A result favoring both over source_only would support task-distribution transfer from teacher-reference paths to actual agent-visited states. A null or negative result, particularly with low visitation diversity, recommends changing data collection/solver capacity before investing GPU funds in richer glyph layouts.

## Falsifiers, controls and exclusions

The following are **hard stop/fail-closed** conditions: altered source/episode/compiler hashes, unknown input schemas, query/test-root leakage, synthetic FakeLean data represented as native, no positive model-visited action witnesses, invalid premise-name syntax, corrupted or mismatched model checkpoints, changed source-theorem lineages, missing independent proof verification, non-deterministic repeated seed-107 fit, and model scoring before source-only registration. A failed data/replay gate stays as a recorded failure, never silently patched by dropping cases.

In the visual fresco the 'message' role denotes the instrumented model's nominal coda slot. A change in logit under zero/roll cross-pass message is a model-side intervention; neither illumination nor a semantic glyph is a Lean certificate. Genuine proof credits are retained separately in the native receipts. The benchmark is development-only because the generated family ancestry outside this suite remains unchecked.

**Outputs:** Markdown source and results reports, native-compiled train teacher, model-visitation JSONL, censored-repair ledger, separately certified forced-recovery receipt, freeze/source/checkpoint hashes, six trained checkpoints and optimizer logs, full calibration episodes and replay checks, matched root comparison and SVG fresco in GitHub Actions artifacts.
