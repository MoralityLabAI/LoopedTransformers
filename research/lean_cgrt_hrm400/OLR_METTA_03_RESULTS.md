# OLR-METTA-03: learned spending of native Lean evidence checks

**Date:** 2026-11-23  
**Experiment class:** native Lean + pinned Hyperon, synthetic public testimony, exploratory model development.  
**Source-only registration:** [OLR_METTA_03_REGISTRATION.md](OLR_METTA_03_REGISTRATION.md), commit `2382c6c`, recorded before source and query scoring.  
**First verified run:** [GitHub Actions #45204632661](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45204632661), green; [CPU regression #45204632472](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45204632472), green.  
**Full original evidence artifact:** [OLR-METTA-03 native budget #14834385381](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45204632661/artifacts/14834385381). ZIP SHA-256 `961dc14b551a1a411972f42fbe4e051a87040417b96ce98149c6b4bc755bcaf9`.

## Why this test follows the earlier OLR/MeTTa work

The earlier OLR-01C/01D classifiers ranked visible arithmetic guards and demonstrated native pending-goal rollback. [OLR_METTA_BRIDGE.md](OLR_METTA_BRIDGE.md) confirmed that Hyperon can filter source evidence and propose a bounded atomic repair, while Lean remains the proof authority. This follow-up asks a distinct learning question: **when should a recurrent policy pay a native verifier cost to resolve uncertain evidence?**

The proof language is the *same* four-tactic propositional/arithmetical Lean family. The policy intentionally sees a **restricted** observation that masks the theorem's numerical guard. The verifier always sees the exact statement, and a full-information arithmetic oracle is kept as the ceiling. Source witness claims are synthetic, fallible statements. Hyperon validates their declared lineage/revision/eligibility relationships, **not their actual truth**.

## Native states, legal actions, and cost function

Each root has two alternative implications to R. Each successful derivation applies one implication, proves the arithmetic equality by `rfl`, then discharges pending P and Q using separate native tactic calls. A missed equality causes the native `rfl` to reject the proof attempt.

Each model starts with six synthetic public witness claims, containing a candidate vote, declared strength, source ID, revision and eligibility. A pinned Hyperon 0.2.10 match query offers source-unique, current and eligible claims. The corresponding evaluator independently reconstructs the expected set from metadata and rejects any mismatch. Duplicate source aliases and superseded revisions therefore fail to gain independent evidence credit.

Three actions share a **six-call** cap:

| Policy action | Actual native operation | Outcome cost |
| --- | --- | --- |
| `skip_r0` | Apply r0 and check its guard, then solve P/Q if valid | 4 calls on success, 2 on rejection |
| `skip_r1` | Same for r1 | 4 on success, 2 on rejection |
| `probe_r0` | Speculatively apply r0 and test its guard; restore the saved root, then attempt the informed four-call complete proof | Always 6 on valid registered roots |

The verifier feedback from the speculative probe arrives **after** its two charged native calls. Source episode outcomes are real Lean responses, including complete kernel/axiom checks and independent standalone compilation of the successful full theorem. The restored state is replayed from the original immutable Lean server state ID. The reward is:

```text
utility = 1.0 * independently kernel-certified completion - 0.07 * attempted tactic calls
correct unprobed branch  = 0.72
paid probe, then proof    = 0.58
wrong unprobed branch    = -0.14
```

The direct analytic oracle computes exact numeric guard truth using privileged theorem data and closes every root for 0.72. The **always-probe** policy closes every root for 0.58.

## Source/data split and learned policies

- **Train:** 24 independent arithmetic triplets, each with two paired truth placements (48 worlds). Source operands occupy 17–30.
- **Development:** eight distinct pairs (16 worlds), disjoint in exact (a,b,offset) lineage from the source but using the same operand range.
- **OOD:** eight different pairs (16 worlds), operands 31–44. This is an **operand-range shift**, not a new proof language or Storyworld-domain transfer.
- MeTTa claim templates span high-reliability, weak, conflicting and stale/alias modes, all deliberately synthetic. The model receives the six-slot MeTTa-admitted features and never the private correct branch or the root ID as a numerical feature.
- A tied self-attention/MLP physical block with width 24 is trained at recurrent depth **three** (5,307 parameters) or **one** (same parameters). A direct six-claim MLP baseline has 5,955 parameters and uses all the same model-visible data. Training uses native source action-value vectors, three seeds (17,29,43), 80 AdamW updates and the same minibatch index sequence per seed. The models learn expected reward for skip-r0, skip-r1, or probe-r0 using supervised regression on real native source receipts.
- Deterministic CPU algorithms, single-thread PyTorch 2.5.1+cpu and disabled MKLDNN are required. Source threshold calibration for the independent-source majority heuristic uses TRAIN only.

All query policy choices were serialized **before** any new evaluation-native outcome was executed. There were **96 distinct native evaluation episodes**, covering every registered root and all three available policy actions. All 96 replayed independently. The 32 unique successful query theorem scripts compiled separately with no axioms. The source fitting stage separately compiled **48** successful source theorem scripts.

## Development and OOD scorecard

Mean utility is computed over the same root pairs and three model seeds for learned policies. Analytic reference policies are scored once per root. These are shared-root repeated model evaluations; the root pair, rather than an individual seed-world observation, is the statistical sampling unit.

| Policy | Development utility | Larger-operand utility |
| --- | ---: | ---: |
| Three-pass tied recurrent evidence reader | **0.58417** | **0.62583** |
| One-pass same-parameter attention reader | 0.58458 | 0.60833 |
| Direct full-packet MLP | 0.55958 | **0.64958** |
| Three-pass checkpoint, all rejected claims reintroduced | 0.24042 | 0.24042 |
| Three-pass checkpoint, MeTTa admission bits removed | 0.57250 | 0.61417 |
| Always pay for a native guard probe | 0.58000 | 0.58000 |
| Curated-source majority, TRAIN-calibrated threshold | 0.35500 | 0.44500 |
| Uncurated majority with the same threshold | 0.61500 | 0.63250 |
| Full-information exact arithmetic oracle | **0.72000** | **0.72000** |

The strong full-information oracle and the simpler direct neural reader are essential controls. In the larger-operand panel, direct achieved **47/48** kernel-certified seeded attempts and three-pass recurrence achieved **46/48**. The direct models used 226 native tactic calls across those attempts, versus 228 for the three-pass recurrence. A six-call, always-probe baseline would spend 288 calls on 48 trials.

The three-pass reader obtained **44/48** seeded development proofs and **46/48** larger-operand proofs. On development roots, its mean advantage over the direct reader was +0.02458 utility, with a descriptive eight-root paired bootstrap 95% interval **[-0.03500, +0.09917]**. The larger-operand difference was -0.02375, interval **[-0.05667, +0.00292]**. Both intervals permit zero or opposing effects. On larger operands, the three-pass-minus-one-pass difference was +0.01750, descriptive interval **[+0.00583, +0.02917]**, but its development difference was approximately zero. This is a small public exploratory cohort and a change of direction across splits, not confirmation of a useful recurrent mechanism.

### MeTTa intervention audit

On the *same frozen three-pass checkpoints*, reintroducing the six original uncurated claims, including deliberately stale/duplicated and adversarial testimony, sharply reduced utility to 0.24042 on both panels. This is a task-designed **representation/data-treatment intervention**, and the source modes favor curation by construction. It shows that the trained model's input pathway responds to filtering.

Removing only the Hyperon admission-bit feature had a smaller, variable effect: average utility fell from 0.58417 to 0.57250 in development, and from 0.62583 to 0.61417 on larger operands. Several per-seed results increased when the bit was removed. The uncurated majority heuristic also outperformed the curated-majority heuristic under the source-fitted threshold. Thus these experiments **do not establish a generic advantage from native MeTTa filtering** across policies or model families.

The exact source-lineage and revision checks are still an important *data-custody* rule: two aliases for the same source cannot masquerade as independent corroboration. Hyperon runs the public rule matcher; Python cross-checks its selected IDs, and Lean controls the final proof verdict.

## Custody and provenance

- Pinned Lean 4.19.0 SHA-256 `92c3d35b5bfaa5e0fea413a775d504cf46cd95e1345df61c2274f76779e7e023`.
- Pinned native Hyperon `0.2.10` with one match query per registered evidence packet.
- Source registered manifest SHA-256 `fe2fe8393cbd275854967e4aba3396b3cb3f0758e0a97aec36c6a78f518351d9`.
- Source model weights `models.pt` SHA-256 `e20f257bce9c23253524a40c052ea136f9cfb42c4b91105f836c91d675869125`.
- Source action rewards `source_rewards.jsonl` SHA-256 `834b6712cd6e772e207d13295c39b4b83ff6fe81f10669c53292f0b5e996df6c`.
- Frozen evaluation decisions `frozen_choices.jsonl` SHA-256 `572ae08eb5b8dc93354d9d9d04193e1f623afd6050eefc1abff75854340d776c`.
- Evaluation native events `native_events.jsonl` SHA-256 `348535c49fb107184ff86c5c284c26e4e5ac75ca9d249504475c2164d25f0186`.
- Evaluation standalone axiom-free source SHA-256 `c530c12e5028e2dad3b10fbd045c163749c54ebf02f1b7d29a67df657e27ab3d`.
- Full artifact is available in [GitHub Actions #45204632661](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45204632661/artifacts/14834385381). The folder stores all three registry splits, source optimizer checkpoints/receipts and evaluation decisions/native replays.

## Reproduction

From `research/lean_cgrt_hrm400`, with native Lean 4.19 built, PyTorch CPU installed and `hyperon==0.2.10`:

~~~sh
python -m ppt_gym.metta_budget_campaign freeze --seed 20261129 --out runs/olr-metta-03-registry
python -m ppt_gym.metta_budget_campaign fit --bank runs/olr-metta-03-registry --steps 80 --out runs/olr-metta-03-source
python -m ppt_gym.metta_budget_campaign score --bank runs/olr-metta-03-registry --fit runs/olr-metta-03-source --out runs/olr-metta-03-eval
~~~

The source-only and query directories are create-only. The first CI run passed the source and query certificate gates; a follow-up build adds an explicit same-worker seed-17 three-pass tensor replay gate. Those replay results must be checked before claiming bit-exact reproducibility across runs.

## Research decision

**Keep OLR-METTA-03 as a validated native research gym and a negative/ambiguous learned-recurrence result.** Its three-pass reader responds to synthetic evidence and sometimes saves tool calls, but the strongest direct model earns more utility on shifted inputs, the full-information arithmetic solver remains dominant, and the model/FLOP budgets are not matched.

A follow-up should add genuinely incomplete proof-state semantics, delayed support for shared dependent witnesses, an actual learned STOP/restore controller and native MeTTa permit/lineage constraints whose usefulness cannot be reduced to a synthetically corrupted vote. Continue to enforce source-only training and prospective FSI-01 shams before considering architecture promotion.
