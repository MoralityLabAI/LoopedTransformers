# AF-20: counterfactual STOP calibration, final scored report

**Registered scientific verdict: FAIL.** **Source-pinned hosted engineering: PASS.** All six GitHub Actions jobs in [run 43723332839](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43723332839) passed, including Python 3.12/Torch 2.5.1+cpu training and scoring, independent physical Horn BFS replay, frozen split checks, and the standalone Lean 4.19.0 source-Boolean lemmas. This is a small structured Horn policy experiment, not evidence of independent Lean theorem proving by the model.

- **Frozen experimental code:** [commit e7621de4ef3daf2f8287d5f982027879927b4543](https://github.com/MoralityLabAI/LoopedTransformers/commit/e7621de4ef3daf2f8287d5f982027879927b4543).
- **Registered protocol:** [PROTOCOL.json](../../experiments/activation_funnel_af20/PROTOCOL.json), SHA-256 `4949ff20ba16129c4976c674d4dd7aacf232579e837a338c52ecbd66ca4b583c`.
- **Frozen corpus manifest:** [fresh_manifest.json](../../experiments/activation_funnel_af20/fresh_manifest.json), SHA-256 `669fb795d2ad698a9e7a7330dd17c5aaf36c6bbcc4d0697a3377869a146e3d70`.
- **Official hosted aggregate:** `results.json` SHA-256 `b0a7086e7e81bf4ab671e11c20466f3f28913e11835097adebb0ea508f7379ed`.
- [AF-20 draft PR #102](https://github.com/MoralityLabAI/LoopedTransformers/pull/102), stacked on AF-19 PR #100. Keep draft/unmerged pending evidence review.

## Study design

All models reuse the original source-hash-pinned TypedLoop(16) and AF-19 **ancestor-pooling STOP readout**, each with **6,036 parameters**. At a fixed two- or four-call recurrence depth, `bce` and `pair` arms start from identical weights and receive identical sampled source states, exact fact pairs, 240 optimizer updates, and native-cell forward calls. The paired loss incurs additional backward computation. Original AF-17/19 train/dev splits (928/352) remain unchanged. The prospective heldout corpus has **416 inputs** on new physical layouts 67–71, using the existing Horn logical templates.

The paired training roster contains **304 reachability-flip pairs** and **484 connected, outcome-preserving pairs**. Each pair changes one graph-connected initial fact and presents a nontrivial legal-rule-versus-STOP decision. For outcome-preserving pairs, both examples are reachable and have exactly equal shortest-proof cost. This is a conditional-invariance target rather than a claim about arbitrary connected facts.

Both arms fit shortest-action CE, cost Huber, reachability BCE and STOP BCE, and also receive BCE on both members of the same paired examples. The `pair` arm additionally fits relative reachability/STOP ordering on truth flips and probability consistency on outcome-preserving connected toggles. The host alone decides which Horn inferences are legal and verifies every accepted proof edge.

## Complete preregistered four-seed outcome

All four seeds pass the registered **0.65 train/dev proof-success eligibility** threshold for both arms at both loop counts. The primary comparison is `pair_l4` minus `bce_l4` on the two held-out deep families, with >=0.05 gains, >=0.70 absolute proof success, <=0.02 added early-stop rate, >=0.75 flip-order accuracy and connected-distractor probability change <=0.10 and <= BCE+0.02.

| Seed | Eligible | BCE L4 success: depth4 / depth5 | Pair L4 success: depth4 / depth5 | Pair flip-order accuracy: depth4 / depth5 | Registered seed verdict |
|---:|---|---|---|---|---|
| 17 | Yes | 10.8% / 2.7% | 0.0% / 0.0% | 93.5% / 93.5% | FAIL |
| 29 | Yes | 24.3% / 24.3% | 10.8% / 8.1% | 100.0% / 97.8% | FAIL |
| 43 | Yes | 8.1% / 0.0% | 0.0% / 0.0% | 67.4% / 67.4% | FAIL |
| 61 | Yes | 13.5% / 13.5% | 18.9% / 18.9% | 95.7% / 95.7% | FAIL |

Every eligible seed fails one or more required conditions, so the aggregate **FAIL** is unambiguous. Seed 61 improves proof success **+0.05405** on both depths (0.13514 → 0.18919), and has flip-order accuracy 0.95652, but misses the absolute-success and connected-distractor limits. Seed 29's paired arm falls below its BCE control at both depths. Seed 17's four-call paired model stops before proving a held-out goal; its pair-order score stays above 0.93. Seed 43 is development-eligible in the hosted pinned runtime even though the local Torch 2.10 run did not qualify it.

### Verified pair-response controls

| Seed | Heldout depth4 pair joint ordering | Connected reach probability shift | Connected STOP probability shift |
|---:|---:|---:|---:|
| 17 | 93.5% | 0.048 | 0.045 |
| 29 | 100.0% | 0.193 | 0.224 |
| 43 | 67.4% | 0.027 | 0.006 |
| 61 | 95.7% | 0.370 | 0.372 |

Pair ranking fidelity and rollout proof success measure different capabilities. A policy can order two STOP situations correctly while choosing STOP in both. Some seeds also achieve small connected-invariance errors while failing to solve a substantial number of reachable goals. Seed 61 improves the proof outcome modestly while moving the connected-invariance scores away from the registered target.

## Independent replay and evidence

The original successful hosted run preserves **16 Torch model checkpoints with matching named NumPy exports**, 11,648 complete host episodes, 5,424 evaluated counterfactual pairs, 420 sampled rule-policy forward replays and 192 paired-state forward replays. It independently checked **3,152 exact training pair source labels** against physical Horn rule incidence and shortest-path BFS. Every saved action was validated against current proved atoms and original premise/head edges. **Zero false accepted Horn proofs** occurred.

The [five original GitHub Actions artifacts](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43723332839) are retained byte-for-byte and individually SHA-256 pinned in [AF20_EVIDENCE_MANIFEST.json](AF20_EVIDENCE_MANIFEST.json). That manifest also lists every member's SHA-256, byte length and source commit. The original local Python 3.13/Torch 2.10 run independently audited the same frozen source and found a FAIL verdict, with numerical/eligibility differences explicitly separated from the hosted source-pinned result.

## Post-registration mechanism diagnosis

The [decision-gap analysis](AF20_STOP_DECISION_GAP.md) identifies a mismatch between the training proxy for STOP and the greedy host action rule. The model fits `STOP logit - logsumexp(legal-rule logits)`, whereas STOP executes when `STOP logit > max(legal-rule logits)`. The former gap can be negative even when greedy STOP is selected, especially when several legal rules compete. It also provides only *relative pair ordering*, which does not require a reachable state's absolute STOP score to lie below its best legal rule. These are mathematical distinctions; the observed positive pair ranking alongside widespread early STOP provides empirical motivation to test them in a separately preregistered follow-up.

The natural next experiment is to train **a true greedy decision margin**, conditioned on the legal-rule mask and exact host source labels, with an explicit negative-control comparison and no heldout-driven tuning of this frozen AF-20 protocol. AF-19, AF-18 and all earlier negative results remain unchanged.
