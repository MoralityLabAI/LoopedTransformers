# LR-CGRT-400: Lean-certified, cross-generation hierarchical reasoner

**Status:** Proposed training campaign, November 5, 2026. No Silico GPU job has been launched by this plan.

## Research objective

Train a roughly **400-million-parameter specialist HRM from scratch** to choose and execute proof actions over Lean 4 proof states. Integrate (a) the verified-dependency routing mechanism demonstrated in the LR-01 pilot; (b) hierarchical, shared-weight fast/slow recurrent computation; and (c) **Cross-Generation Reasoning Tree (CGRT)** sampling of alternative proof actions under controlled contextual variation. The experiment asks whether the combined architecture increases **Lean-kernel-verified theorem completion per unit of compute** on genuinely unseen proof families.

**Naming and provenance:** CGRT is interpreted here as **Cross-Generation Reasoning Tree**, the context-variation and prefix-controlled branching algorithm of Xie et al., *Interpretable Analysis and Reasoning Enhancement for LLMs via Cross-Generation Reasoning Trees* (ICLR 2025). The local design trail separately calls its typed-gate compiler **GCPT** (Gate-Compiled Protocol Transformer). The two labels are distinct. A compatible optional GCPT adapter can compile tactic protocols into typed gates; it does not replace CGRT's branching method. If another private CGRT design is intended, replace the branch-policy adapter after establishing its source before the full run.

## Starting evidence

The LR-01 archive `lean_reasoner_lr01.zip`, intended for `MoralityLabAI/LoopedTransformers/research/lean_reasoner/`, contains a restricted arrow/application prover. A 5,161-parameter graph-attention policy achieved 88.75% same-depth completion at two shared passes and 100.00% at four; depth 5–7 completion was 10.97% and 48.75%, respectively. Two-pass-trained weights achieved 59.03% on the deeper set when given eight inference passes. This is a synthetic dependency-routing task with the compiled graph and exact applicability masks. **Lean kernel validation was blocked** in the sandbox, although a separate Python type checker and 17 tests passed. The model carries no persistent neural state across proof steps; proof terms are preserved in the explicit bank.

Do not transpose those results into 400M capability claims. Preserve LR-01 as a frozen regression experiment, and run its `audit_lean.py` with the pinned Lean version before treating synthetic examples as Lean-certified training input.

## Architecture

Input is a normalized Lean 4 goal state: local context, targets, available lemma signatures, proof-term DAG, typed dependencies, and a bounded action vocabulary. Training may use extracted elaborated proof actions. At deployment, the input has only observable proof state, retrieved declarations and certified proof bank. Teacher proof-path labels are excluded.

Define the shared state

`S_t = (Gamma_t, Goals_t, CertifiedBank_t, Tree_t, Budget_t)`.

The slow/high-level state `zH` estimates proof strategy, likely useful lemmas, branch budget and stop/continue decisions. The fast/low-level state `zL` searches applicable tactics and local term-construction steps. Both operate recurrently using **tied blocks**; embeddings and graph edges are computed once per proof step. The proof executor checks tactic application and the final theorem in Lean's kernel. An invalid or incomplete proof never becomes a certificate.

Illustrative recurrence, with two L updates per H update and two H cycles per decision:

```
x_t = EncodeLean(S_t)
for hcycle in range(2):
    for lcycle in range(2):
        zL = LowShared(zL, x_t, zH, proof_graph_mask)
    zH = HighShared(zH, pool(zL), TreeFeatures(S_t))
a_t = BranchPolicy(zL, zH, legal_actions, context_id)
S_{t+1}, receipt = LeanExecutor(S_t, a_t)
```

The recurrent **visit count** changes computation, not learned parameter count. Run 1/2/4/8 microstep and macrostep variations under equal-compute and equal-parameter controls; do not assume two is optimal. Use a controlled state-update gate, clipped residual scaling and gradient-horizon ablations to mitigate divergence at larger visit counts.

### Proposed parameter allocation

Use a uniform 1,344-dimensional hidden width and 16 attention heads. For SwiGLU blocks with FFN width 3,584, a bias-free attention + MLP block has approximately 21.676M weight parameters. This is a design estimate; the implementation must emit an exact parameter manifest.

| Distinct learned component | Nominal allocation |
| --- | ---: |
| Fast L, 8 transformer blocks | 173.4M |
| Slow H, 5 transformer blocks | 108.4M |
| Proof-graph encoder, 2 blocks | 43.4M |
| Cross-context CGRT branch policy, 1 block | 21.7M |
| 32,768 x 1,344 shared input/action embedding | 44.0M |
| Pointer, value, halt, lemma retrieval, and calibration heads | ~9.1M |
| **Total target** | **~400.0M** |

Set explicit constructor validation: 398M <= number of trainable parameters <= 402M. Tie the language output to the shared embedding if text rendering is supported. The first version selects typed tactics, lemmas and term-building operators via pointer heads rather than learning a broad free-form natural-language decoder. Limit maximum proof graph nodes in the initial curriculum; record truncation rather than silently dropping relevant hypotheses.

## CGRT integration at the proof frontier

The original CGRT constructs a shared-prefix reasoning tree across different contexts, taking alternative next tokens at a given node. This adaptation branches at **semantic proof actions** after Lean elaboration. Its controlled contexts are prompt/tactic demonstrations, proof-plan hints, lemma ordering and alpha-renaming; hold the **same proof state and accepted action prefix** constant while evaluating alternatives. This isolates context effects from prefix differences.

At step `t`:

1. For each predeclared context `c in C` (`|C| = 2–4` in the initial campaign), sample 1–2 candidate next actions from the HRM branch policy.
2. Normalize equivalent actions by their Lean-elaborated effects, preserve provenance (`context_id`, logits, branch parent, cost), and retain up to two semantically distinct actions per expansion.
3. Attempt tactics inside a rollback-capable Lean environment; record successful local transitions and failures. Continue successful branches up to a maximum four active states and a bounded wall-clock/token budget.
4. Train the policy on **all verified successful continuations** with preference for shorter valid proofs and calibrated estimated likelihood of completing a theorem.
5. Keep branch correctness tied to the **completed theorem certificate**, never majority vote alone. Use a value estimate only to schedule exploration.

Pure model sampling, CGRT branch sampling, theorem-aware random branching and exact-search schedules are controls. Match Lean executions, candidate actions and GPU seconds when comparing policies; otherwise the observed gain may come from greater compute.

## Dataset contract

### Sources

- **Synthetic typed curriculum:** LR-01 family, expanded only after the Lean-kernel audit, including implication, conjunction/disjunction, quantifiers, equality and rewriting in distinct difficulty bands.
- **Real Lean curriculum:** pin Lean 4, Mathlib commit, elaborator version and any external proof repositories. Extract theorem declarations, exact proof states, applicable local lemmas, accepted tactic traces and exported certificates using Lean's native interfaces. Check licenses and preserve source attribution.
- **Counterfactual branch corpus:** for each sampled state, create CGRT alternatives under invariant semantic transformations and contexts; label actual Lean execution outcomes, branch completion and compute cost.
- **Difficult negatives:** mismatched hypotheses, wrong lemma instantiation, type errors, circular assumptions, `sorry`, adversarial near-terms, unreachable goals. Keep these in explicit unsuccessful classes, with no false success certificates.

### Dataset targets, dependent on extraction yield

- Stage A: 50k–100k Lean-certified synthetic theorems for structural warmup.
- Stage B: 200k–500k real theorem-rooted training traces and 1M–3M nonduplicate proof-state/action supervision records. Track token count after packing rather than equating a state with a token.
- Stage C: 200k+ verified CGRT branch comparisons, using actual accepted/rejected or completed/failed outcomes. Record generated attempt counts and prover time.

Split by theorem family, dependency graph neighborhood and source repository. Hash alpha-equivalent or syntactically normalized proofs; prevent related theorem families and nearly identical proof traces crossing splits. Freeze complete source-package holdouts before any training. Keep depth-8+, unseen lemmas, alpha-renaming, lemma reorderings, context style, and novel proof motifs as separate evaluation axes.

## Loss and update schedule

Train from random initialization with supervised proof-action learning first. Treat the LR-01 checkpoint as an **external teacher for its synthetic family**, not as a compatible weight initialization for a 400M model.

For current legal action set `A` and acceptable teacher actions `Y`, use LR-01's set-valued loss:

`L_set = logsumexp(scores[A]) - logsumexp(scores[Y])`.

Add theorem-endpoint success, progress/value calibration, correct halt/continue, proof-state transformation consistency and context-invariant scoring. Suggested initial relative coefficients, to be frozen and logged before the first comparison: `1.0 L_set + 0.5 L_branch + 0.25 L_value + 0.15 L_halt + 0.10 L_invariance`. Maintain separate metrics rather than conflating these into a single reward number.

After supervised learning has crossed the predeclared gate, use verifier-selected CGRT branch ranking and then a small capped GRPO/rejection-sampling fine-tune on kernel-completed proofs. End reward: +1 for a certificate, with only modest cost/length regularization. A coherent `fail/abstain` on genuinely unsolved theorems receives calibration credit. Keep positive, negative and corrected outputs separate.

Optimization: BF16 mixed precision; AdamW; Flash/SDPA attention if graph-masking implementation preserves directed links; fixed-shape bucketing; microbatch 1–4; gradient accumulation to a measured effective batch; gradient checkpointing across recurrent cycles; clipped gradient norm; truncation/recompute across long recurrence; checkpoints with optimizer and random state. Compare intermediate gradient horizons and state-update control as measured in newer recursive-reasoning research. All decisions use the same frozen evaluation splits.

## Silico compute plan

**GPU:** Start with one H100 80GB. The 400M BF16 weights are about 0.8 GB. Training optimizer state, master parameters and gradients are several GB; activation and attention-memory costs dominate with long proof graphs and recurrence. Memory must be **measured** with the final model, graph mask, optimizer and planned sequence length before launching an extended job.

**Billing:** Silico RUs and GPU-hour conversion depend on account and plan. Read the current Silico Billing view, quote observed RUs per GPU-hour from a real smoke and set an absolute RU cap. No conversion rate is assumed in this plan. Never authorize auto-retries, additional GPUs or long runs by default.

| Gate | Work | Maximum new compute before review |
| --- | --- | ---: |
| 0 | CPU-only source extraction, Lean audit, unit tests, parameter count, GPU memory estimate | 0 H100h |
| 1 | Single H100 compile/throughput/one-batch train/rollback/restart smoke | **0.33 H100h** |
| 2 | 400M synthetic + real Lean SFT, 2–3 seeds short runs, first learning curves | **8 H100h** |
| 3 | Main supervised curriculum, optimizer and loop-depth controls | **up to 160 H100h** |
| 4 | Verified CGRT branch learning and fine-tuning | **up to 96 H100h** |
| 5 | Frozen holdout evaluation and matched-compute ablations | **up to 48 H100h** |
| **Potential campaign** | **Dependent on approvals and viable measured throughput** | **~312 H100h** |

Only Gate 0 is approved by this document. The Gate 1 smoke is the first proposed GPU action and requires user approval of its measured RU cost. Treat later maxima as independent caps requiring a decision between stages. A full 400M from-scratch run may need a larger budget or reduced data scope; make the continuation decision from observed tokens/sec, proof actions/sec and learning curves.

**Single-GPU first** avoids cluster overhead and fits within H100 80GB if profiling confirms it. If measured memory exceeds 72GiB at the proposed batch and context, use activation checkpointing, shorter packed sequences and smaller microbatches before moving to 2–4 GPUs. Use FSDP only with measured benefit.

Write durable artifacts to `$SILICO_EXPERIMENT_ARTIFACTS_DIR`; every follow-on job must explicitly attach the preceding checkpoint and dataset manifests because Silico on-demand jobs have isolated filesystems. Export model/optimizer state and a small replay fixture after every stage.

## Predeclared evaluations and decision rules

**Primary:** Lean-kernel-certified pass@1, pass@4, and proved-theorems/H100-hour on source/family/depth holdouts. Count full `sorry`-free theorem completion; report tactics accepted, certificate validity, budget exhaustion and abstention separately.

**Core comparisons:** 400M shared HRM + Lean + CGRT; 400M shared HRM + Lean, without CGRT; a 400M flat transformer with matching action vocabulary; HRM with randomized CGRT contexts; and a smaller LR-01-like proof selector. Add an exact search/Lean tactic baseline. Match total training tokens and inference-time proof attempts; also provide equal GPU-FLOP or walltime curves. For architecture attribution, perform a smaller, controlled factorial where data and budgets are held fixed.

**Stop conditions:**

- Kernel rejects any reported success: stop and repair the receipt mechanism.
- Branch score performs no better than a prior or random ranking on fresh heldouts: stop CGRT expansion and retain the negative result.
- Recurrent steps destabilize validation or previously correct proof retention: save the last stable checkpoint and study truncation/step-size changes.
- A stage exceeds its user-approved RU or GPU-minute allowance: stop before another job or retry.

**Full-training go gate:** After Stage 2, require statistically supported improvement in proof completion or correct-next-action rate over a compute-matched plain HRM baseline on withheld proof families, with a reliable positive slope as more data enters. Stage 4 claims CGRT value only if it improves certified theorem completion **at matched proof attempts and GPU seconds**. Cluster confidence intervals by theorem family, with three predeclared seeds on affordable comparisons.

**Target research milestone (hypothesis, not a claimed result):** >5 percentage point absolute improvement in kernel-certified pass@1 or >=15% better valid proofs/H100-hour versus the strongest compute-matched 400M alternative on the unseen-family set; simultaneously track the deeper-than-training proof gap and retained correctness when inference recurrence increases.

## Downstream task: interrogating a larger model

Once it learns formal proof-frontier planning, an additional interface can convert *versioned, explicitly attributed* statements by a larger model into Lean propositions. The HRM can choose the next **clarification** to request based on missing premises or candidate inconsistency certificates. Success is a small, properly scoped Lean contradiction witness or a documented coherent revision; fabricated claims and changed meanings receive no proof credit. This is a separate, later evaluation with its own data and baselines.

## Deliverables and publication

Place the campaign under `MoralityLabAI/LoopedTransformers/research/lean_cgrt_hrm400/`. Preserve the LR-01 source artifact, commit or attach exact dataset/compiler revisions, model code, config, data manifests, unit tests, proof-replay fixtures, checkpoint SHA256, mixed-precision/throughput logs, kernel-audit receipts, and per-seed holdout outcomes. Ask Silico to publish code through a **draft GitHub PR** after its codebase layout is reviewed. Never claim a remote commit or successful GPU training without an actual commit/experiment report.

## Reference material

- LR-01 local artifact: `lean_reasoner_lr01.zip` and `lean_reasoner_report.md` from the prior conversation. Its remote publication was pending.
- Xie et al., Cross-Generation Reasoning Tree: https://openreview.net/pdf?id=ON3QLXrwVb
- HRM original: https://arxiv.org/abs/2506.21734
- HRM-Text implementation notes: https://huggingface.co/docs/transformers/main/model_doc/hrm_text
- Recursive model optimization ablations: https://arxiv.org/abs/2609.39967
- Silico compute and artifact handling: https://docs.goodfire.com/administration/compute/
- Silico data access: https://docs.goodfire.com/data/
