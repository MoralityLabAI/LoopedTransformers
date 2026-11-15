# AF-09 final scored research result | November 15, 2026

**Registered primary scientific verdict: FAIL.** Both independently qualified seeds 17 and 29 fail the frozen deep causal-locality hypothesis. **Engineering verdict: PASS.** All six jobs of [primary CI #43682916363](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43682916363) passed, including the original Boolean-Horn oracle, exact checkpoint source audit, all six trained models, the independent per-seed evaluators and the cross-seed aggregate.

- **Frozen experiment source:** [`643fd9085c4801960a228a16ddbe49a8a58dc750`](https://github.com/MoralityLabAI/LoopedTransformers/commit/643fd9085c4801960a228a16ddbe49a8a58dc750)
- **Original registered protocol:** [PROTOCOL.json](../../experiments/activation_funnel_af09/PROTOCOL.json), SHA-256 `b531cd9d0f19e953e1a2d7d25f5f8f72553e923ba65f8186920fdc76ae17b05d`
- **Immutable original comparison source:** Qualified AF-06 warm-ups, commit `36b83eb83bc116dd4e938213edeba3057ffa9016`, artifacts `14116102941` and `14115887129` from CI `43599882702`, checked against their pinned artifact ZIP SHA-256 digests
- **Independent aggregate JSON SHA-256:** `5cfd2edd726e746dbd8068d84f0c4f64738863e58be7af0a753b69110fcaebc5`
- **Draft PR:** [AF-09 #54](https://github.com/MoralityLabAI/LoopedTransformers/pull/54), stacked on [AF-08 #52](https://github.com/MoralityLabAI/LoopedTransformers/pull/52); keep draft and unmerged.

## The exact mechanism and registered test

AF-08 proved a code-margin transport lemma for the real affine atom decoder and status head. AF-09 differentiates its *true-class sufficient bound*, rather than merely pushing opposite-label prototypes apart. Let `z` and `z'` denote the protected Q atom's **actual carried compressed codes** after a clean and addressed-intervened recurrent step, and `y` its source Horn truth. The exact bound is

\[
m_y(z')\ge m_y(z)-K_y\lVert z'-z\rVert_2,\quad
K_y=\lVert(W_R[y]-W_R[1-y])W_D\rVert_2.
\]

The `semantic_margin` arm adds `0.30 * softplus(0.5 - [m_y(z)-K_y||z'-z||])` and `0.15 * softplus(0.5 - m_y(z'))` at each addressed downstream step. This gives a nonzero training penalty even when AF-08's fixed code-separation hinges were inactive. The shared task/status, patched-goal, and other-Q negative-control losses remain in place.

All arms start with *exactly identical, previously qualified* AF-06 warm-up weights for each seed, each with 58 float coordinates carried over 29 typed tokens (464 decoded working floats). The `status`, `locality`, and `semantic_margin` arms receive 160 fixed extra AdamW steps with matched training examples, donors and native-cell forward calls. The margin arm performs additional arithmetic and backward FLOPs; equal optimizer steps are not a full FLOP-matched comparison.

The source sees only initial facts and graph structure. A deterministic least-Horn fixed point produces the exact unobserved Q0/Q1 truth, downstream goal and donor/recipient composed history. Fit uses depths 2 and 3; eligibility uses development inputs only. Frozen heldout families cover new graph composition, a new physical layout, and unseen depths 4 and 5.

Both seeds and all three continuation arms pass the registered **90% original-goal and exact joint-Q train/development qualification gate**. The width-2 semantic-margin development scores are seed 17: goal `0.964286`, exact Q `1.000000`; seed 29: both `1.000000`.

## Frozen held-depth results

Protected (semantically untouched) Q accuracy after **three downstream native recurrent steps**:

| Seed | Heldout | Locality | Signed margin | Improvement | Certificate-rate improvement | Scientific outcome |
|---:|---|---:|---:|---:|---:|---|
| 17 | Depth 4 | 0.515625 | 0.750000 | +0.234375 | +0.238281 | Fails absolute correctness and negative-control gates |
| 17 | Depth 5 | 0.601563 | 0.500000 | -0.101563 | -0.093750 | Fails signed gain, clean-goal and negative-control gates |
| 29 | Depth 4 | 0.375000 | 0.500000 | +0.125000 | +0.125000 | Fails absolute correctness and negative-control gates |
| 29 | Depth 5 | 0.703125 | 0.750000 | +0.046875 | +0.082031 | Fails >=0.05 Q gain, clean-goal, negative-control and false-goal gates |

The original gate demanded for **each eligible seed at both unseen depths** an improvement of at least 0.05 in protected-Q correctness *and* positive-certificate rate, absolute correctness at least 0.90, irrelevant-patch goal retention at least 0.90, a clean-goal drop no greater than 0.03, and no increase in false-positive goals. Both eligible seeds fail multiple required conditions. The aggregate is unambiguously **FAIL**.

The signed-margin irrelevant-patch goal retention is `0.5625` (seed17 depth4), `0.4375` (seed17 depth5), `0.7500` (seed29 depth4), and `0.53125` (seed29 depth5). The strongest single-depth Q improvements are real scored observations, insufficient for depth-stable causal abstraction.

### Secondary graph-family effects

Seed 29's novel-layout group improves protected-Q from `0.946429` under locality to `0.982143` under semantic-margin training, with clean-goal accuracy `0.959821 → 1.000000`. Seed 17 loses novel-composition clean-goal accuracy `1.000 → 0.750` under the same new objective. These are exploratory secondary observations. Neither changes the primary failed gate.

## Mechanistic diagnosis from retained actual codes

The independent per-step certificate ledger records true-label signed clean and intervened margins, learned decoder/readout sensitivity `K_y`, actual two-float code displacement, computed sufficient lower bound, and the resulting Boolean certificate. The following values are **mean final-step values** across the exact selected-donor roster of each family; they describe arithmetic in the trained model rather than extrapolated theorem-proving performance.

| Seed/depth | Arm | Mean clean true margin | Mean code displacement | Mean sufficient lower bound |
|---|---|---:|---:|---:|
| 17 / 4 | Locality | +0.24764 | 0.05039 | +0.03726 |
| 17 / 4 | Signed margin | +1.70248 | 0.04209 | +1.56663 |
| 17 / 5 | Locality | +1.01434 | 0.15462 | +0.36876 |
| 17 / 5 | Signed margin | **-0.02154** | 0.07325 | **-0.25796** |
| 29 / 4 | Locality | -0.54721 | 0.17578 | -1.24126 |
| 29 / 4 | Signed margin | +0.65268 | 0.06419 | +0.46089 |
| 29 / 5 | Locality | +1.86821 | 0.22593 | +0.97618 |
| 29 / 5 | Signed margin | +0.88619 | 0.04236 | +0.75961 |

These values separate two failure mechanisms. The new objective suppresses protected-code drift and can increase true-label margin on depth4, but at depth5 the clean semantic margin itself may become small or negative. Stabilizing the local numerical code does not guarantee a correct latent subgoal at unseen graph depth. The underlying kernel check and the neural proof-state scores are kept in separate ledgers.

## Independent replay and integrity

GitHub CI used the registered Python 3.12 / Torch 2.5.1 CPU runtime. The preflight passed 9 tests in both Python modes and recompiled **4,176 closed Lean 4.19.0 synthetic Boolean-Horn oracle theorems**. Lean certifies the oracle's closed Boolean outcomes only.

The complete six-model evaluation retained **2,976 clean prediction rows**, **5,952 donor-counterfactual rows**, **9,024 recurrence-stage traces** and **9,024 independently checked affine margin-certificate rows**. For each seed, the evaluator recomputed full source labels, sample provenance, raw logits and metrics, source and Torch/NumPy checkpoint hashes, physical code-patch identities, and sampled exact forward passes. Its separate NumPy verifier reconstructed each stored `K_y` and margin inequality from the saved original decoder/readout weights and actual code arrays. **Zero false-positive mathematical certificates** were reported. Both independent evaluations and aggregate completed successfully.

All AF-01 through AF-08 source experiments and scientific verdicts remain unchanged. This result supports a valid **mechanistically interpretable training signal** in a small TypedLoop, while the registered evidence rejects the claim that this particular loss already gives depth-stable proof-state transport. Further attempts should preregister more stringent graph-depth invariance and clean true-class margins across multi-depth recurrence, with a compute-matched status baseline and untouched-descendant controls.
