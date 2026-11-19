# DEUTERIUM-NATIVE-02: Does pass-indexed VPD importance improve causal recurrent-weight steering?

**Prospective protocol-only registration, 19 November 2026.** The current branch must be merged or otherwise cryptographically archived **before implementing the adapter, opening any new model training source, or running prospective scoring**. New-source generation is prohibited from routine PR/source-only CI. Once model/adapter/optimizer code and source hashes are frozen, the first test must be a separately gated one-time release with uploaded source and selection artifacts. If implementation requires materially different objective/training semantics, record `BLOCKED_UNMATCHED_OPTIMIZER` and amend prospectively before opening fresh test data. Never change this protocol after observing new results.

## Prior observations and scientific scope

- [Original DEUTERIUM-NATIVE-01 first heldout #44465490737](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44465490737) was scored for three old LOOPSCOPE teachers and independently recovered/audited in [#217](https://github.com/MoralityLabAI/LoopedTransformers/pull/217). Its registered development verdict is `DEVELOPMENT_NO_REGISTERED_ADVANTAGE`, matched random-minus-VPD goal NLL mean −0.00002608 nats, 95% old-root t interval [−0.00028738,+0.00023521].
- [DEUTERIUM-ADAPTER-01 source-only](../adapter_audit/RESULTS_SOURCE_ONLY_20261119.md) used the **actual original fitted VPD CI weights** and proved the pinned stock \`param_decomp.ComponentModel\` caches only the final of four inputs to one repeatedly invoked \`block.chan_mix.2\` path and reuses that final-pass importance mask at every pass. The same frozen learned CI networks assign very different masks to earlier original clean reached states (first-pass mean absolute mismatch 0.2746/0.2679/0.2980 in old teachers).
- These results motivate an isolated *adapter-semantics test* on independently newly initialized teacher roots. The exposed old teachers 103/211/307, data seeds 731055/731057 and earlier LOOPSCOPE source cohorts are **development controls only**. They are excluded from any claim of new-model scientific confirmation.

## Precise primary comparison

On each new original-architecture 44,001-parameter four-pass weight-tied TRM, fit two VPD decompositions to the **same physical shared \`block.chan_mix.2.weight\` (48×128)**, with identical 16 component factors, initialization seeds, fit-data sequence, optimizer steps/lr, loss names/coefficient totals, physical/dose targets and random comparison bank.

1. **Stock-last**, the unchanged \`param_decomp\` implementation from commit `2df0092d5be4d8e4748b3c66ef8325dedef17549`: its component input-cache key is the module path; the final of four calls determines one learned CI mask shared across all four native applications.
2. **Pass-indexed**, a minimal separately audited adapter built on the identical pinned component factor/CI network classes and same weight-tied target. Capture original *clean* pre-weight activation \`a_0,a_1,a_2,a_3\` at the four real calls, use the **same single learned CI network** to produce \`m_k=CI(a_k)\`, and apply \`m_k\` at precisely native application k during the masked four-pass replay. Reset its invocation counter before each source forward; reject stale counter state, shape drift, or different actual call counts. The pass-indexed formulation shares factor parameters and CI network weights across depths. The original recurrent **target weight is unchanged and shared at all four steps**.

Source-anchored importance masks (from the four clean-state inputs) are the PRIMARY pass-indexed intervention-selection policy. Online mask recomputation using mask-altered reached states is exploratory only and cannot replace the primary. Any per-pass reconstruction/importance loss must be normalized so aggregate coefficient, number of token/source sample contributions and effective update count match stock. For identical masks on all four calls, the alternate implementation must reproduce stock masked logits, losses and parameter gradients within registered numeric tolerance before it is scientifically eligible.

### Controlled fitting

- Preserve exact Torch float32 44K model native architecture from original \`mechinterp/loopscope04/gradient_flow.py\`, tied \`ReasoningBlock\` repeated exactly four times, original ten-class head, 81-digit one-position right-cyclic-copy supervision.
- Teacher training: 200 AdamW steps at lr .001, weight decay .0001, original \`final_only\` objective, batches of 16 from 512 original-architecture source strings, independent teacher initialization and train-data roots. Do not select seed, checkpoint or training budget based on new fit outcomes.
- For each teacher, **paired VPD fits** start with the *same initial 16 rank-one factor matrices* and same CI initialization. Run exactly 32 faithfulness warmup + 64 main optimizer updates at constant lr .001, 8 source strings/batch, 2 PGD steps; pin the original \`FaithfulnessLoss\`, \`ImportanceMinimalityLoss\` and \`PGDReconLoss\` coefficients and their per-call normalization. Maintain same target model (frozen), source batches, optimizer type, gradient clipping policy, RNG/mask schedule where applicable. Both arms save complete optimizer loss/gradient/faithfulness traces and all 16 fitted component tensors/CI weights.
- **Engineering STOP GATE:** if one backend cannot represent pass-indexed masks without changing effective fit budget/PGD adversary mask dimensions, coefficient scale, or true native four-pass model output, the result is \`BLOCKED_UNMATCHED_OPTIMIZER\` with a source-only diagnostic. Do not patch the adversary after scoring.

## Newly reserved independent model and data roots

Exactly twelve NEW independently initialized teacher seeds, in fixed order:

\`930101, 930103, 930107, 930109, 930113, 930127, 930131, 930137, 930139, 930149, 930151, 930157\`.

For teacher index i (zero-based 0..11), generate independent source strings with \`fixed_data\` ONLY in the correctly gated stage:

| Split | Generator seed | Count | Permitted use |
|---|---|---:|---|
| Teacher supervised train and VPD fit | \`940000+i\` | 512 | Teacher supervision; then VPD input-only fit; gradient oracle |
| Teacher model competence validation | \`941000+i\` | 128 | Historical competence, no fit/selection |
| Candidate selection/calibration | \`942000+i\` | 256 | 16×2 signed VPD and random candidate selection |
| FIRST sealed scored heldout | \`943000+i\` | 256 | One native scientific evaluation after source lock |

Reject exact sequence overlaps across these newly generated groups, all new roots, and every known previously generated original LOOPSCOPE/Deuterium source. Avoid generating protected 31059,31064,31065 merely to check exclusion. Record little-endian int64 input and right-cyclic target SHA256. The source generator itself uses only registered seeds, with no after-result resampling. The 12 new teacher roots are the units of between-model inference. Repeated token positions and control arms within a string are paired; the nested 256 strings do not increase the between-teacher independent count.

## Fixed physical steering and causal controls

For each fitted arm, candidate \`j\` is the exact physical rank-one \`outer(U[j],V[:,j])\` weight contribution from the fitted components. Each of its 16 matrices is normalized to the **actual** applied FP32 Frobenius dose \`rho=0.01||W_original||_F\`. Search both signs, fixed plus-before-minus tie ordering, over the same eight goal positions \`(0,10,20,30,40,50,60,70)\` on the original 256 calibration sources. Choose the minimum mean goal NLL; freeze all actual matrices, scores, indices, signs, measured rank, physical dose, SHA and calibration source roster before test exposure.

One **shared bank of 16 random Gaussian outer-product matrices** per teacher (both signs, 32 identical calibration comparisons) is used for both learned fit arms and trained-source-only as a rank/dose/search-matched control. An original-train-label supervised negative rank-one task-gradient oracle is descriptive. Preserve literal zero/sham identity, all class logits from original unmodified and every edited native full four-pass shared-weight execution, complete applied-matrix restoration, future goal/off-goal CE, collateral accuracy, and false correct-digit changes. Do not use target labels to select decomposition components except the fixed shared calibration sign/index selector, where both arms are subject to the same rule.

### Primary estimand

For new teacher root i, compute heldout per-string mean goal NLL for stock-last, pass-indexed and the same selected random arm, after actual native four-pass recurrence. Define:

\[
\delta_i =
  \bigl( \mathrm{NLL}_{\mathrm{random}}-\mathrm{NLL}_{\mathrm{pass-indexed}}\bigr)_i
  -\bigl(\mathrm{NLL}_{\mathrm{random}}-\mathrm{NLL}_{\mathrm{stock-last}}\bigr)_i
  =\mathrm{NLL}_{\mathrm{stock-last},i}-\mathrm{NLL}_{\mathrm{pass-indexed},i}.
\]

Positive favors pass-indexed learned components over stock at the **same** frozen teacher and data source. Report each teacher's 2000 preseeded paired whole-source bootstrap (n=256 strings), then one 95% Student-t interval over all twelve independent teacher-root effects (df=11). All twelve original registered roots must remain in the intention-to-treat table, including numerical failures and clean-model incompetence. The primary comparison is eligible only when every source/model/decomposition/replay/candidate/dose gate passes; otherwise report \`BLOCKED_PREREQUISITE\` instead of an effect claim.

**Registered development-positive** requires ≥9/12 strictly positive root effects, full 12-root t lower 95% bound >0, mean improvement ≥0.0002 nats, and no new root losing >2 percentage points of off-goal accuracy relative to its clean counterpart under pass-indexed VPD. All 12 original teachers must have clean heldout full accuracy ≥0.95. If a complete valid study fails this gate, return \`DEVELOPMENT_NO_REGISTERED_ADAPTER_ADVANTAGE\`. These cutoffs are an exploratory engineering resolution target, **not a calibrated high-power generality guarantee**.

## Secondary instrumentation and scientific limits

Record full four-depth native component input/reconstruction KL, per-depth mask overlap and disagreement, mask gradients, CI distribution, true parameter edit norm/rank and native final ten-class logits, model output CE/accuracy, individual pass injection as exploratory, half/double dose as exploratory, and gradient oracle differences. Report performance per actual model root and out-of-source heldout inputs, memory, runtimes, and any unstable adaptivity/PGD behavior. A representation-fidelity improvement without native task improvement is an explicit separate outcome.

Exactly one first source lock must authenticate original trainer source, pinned \`param_decomp\` commit, both stock and pass-indexed implementation bytes, optimizer traces/configuration, model and candidate-bank SHA, before first scientific test. First exposure marker is written **before** generating \`943000+i\` sources. Independent stdlib auditor recomputes all scored logits, NLL, matched dose, control selection, bootstrap, all twelve-root verdicts, and source-root identities with no Torch. A failed first native run consumes the relevant root's heldout seed. Replays are engineering reproduction only.

The original three old VPD teacher roots and already scored 731057 are excluded from any first-test comparison in this protocol. Future out-of-family confirmation would require a separate root-level power/resolution preflight and new reserved test, especially if the mean difference remains near the 10^-4-nat level. No claim about LLM mechanistic reasoning, internal optimizers, or VPD as a universal method follows from this study.
