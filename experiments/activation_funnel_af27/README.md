# AF-27: frozen STOP-gate transfer and native rule-gap diagnostics

The original AF-26 [RULE_ONLY result](../../docs/research/AF26_RESULTS.md) showed that replacing VI-2's learned-cost rule ordering with Direct neural rule ranking recovered 157 completed hard proofs under identical original frozen weights, while substituting Direct's STOP decision alone lost 19. AF-27 tests whether **VI-2's native STOP gate adds correct termination on previously unseen Horn grammars** after fixing rule ranking to Direct.

**Protocol:** [PROTOCOL.json](PROTOCOL.json), preregistered at commit `db8d52b` before any AF-27 generator or new model-scoring code. The source-only case generator was checked prospectively and its one detected depth-six root-ambiguity defect was fixed **before any checkpoint run**. [Source preflight CI](../../.github/workflows/activation-funnel-af27-preflight.yml) independently checks the 384 new case structures, two bitset/host BFS oracles, exact 5/6-firing positive costs, negative unreachability and complete public-case hash disjointness from AF-25 and AF-24B.

## New graphs and frozen checkpoints

Two **genuinely new rule-dependency** families, `forked_and` and `or_gate`, each contribute 40 provable and 56 unprovable cases at each of two held-out depths. The full per-depth cohort has **80 positive and 112 negative goals**. Every case begins with exactly four known atoms, as in AF-25. Physical rule/atom permutations derive solely from source-owned IDs on layouts 510–513 and 520–523. Each graph contains two goal-concluding rules.

One original AF-23 **6,036-weight ReadoutPolicy('ancestor',4)** frozen final checkpoint per seed (17, 29, 43, 61, 73, 89, 103, 131) is retrieved from [original CI #43812708916](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43812708916) and checked against its fixed original Torch and NumPy weight digests. **Zero new training updates, checkpoint selection, or model adaptation.** Inference uses the original min-plus scorer and public host-enabled rule masks.

## Inference arms

| Arm | Native policy or rule |
|---|---|
| **DD** | Direct STOP gate, Direct legal-rule ranking |
| **VD (primary)** | VI-2 STOP gate, **Direct** legal-rule ranking |
| **VV** | VI-2 STOP gate, VI-2 legal-rule ranking |
| **ND** | Forced continuation while any legal rule exists, Direct legal-rule ranking |
| **BFS_NODE_MATCHED** | Public host BFS with per-decision expansion cap equal to the number of unique states scored by the neural two-step lookahead at the same current host state |
| **BFS64** | Public BFS capped at 64 host-state expansions per decision; higher-expansion diagnostic |
| **CLOSURE** | Original additive-cost Horn closure controller, without learned weights |
| **SHORTEST** | Host exact shortest-proof oracle, upper-control only |

Every policy is rerolled from its own chosen host states, under the original shortest-proof-length-plus-one firing budget for positive cases and two decisions for negatives. No truth labels, BFS-derived optimal actions or shortest costs enter neural or non-oracular symbolic inference. **The BFS node-matched control matches unique host state evaluations per decision, not arithmetic FLOPs or latency.** Per-rule examinations, model forward nodes and runtime remain separately reported.

## Registered scientific gate

The fixed comparison is **VD versus DD**, with Direct's legal-rule rankings retained throughout. At **both depths in the same original seed**, VD must:

- Correctly choose immediate STOP on **at least seven more of the 112 negative cases** than DD.
- Finish at least **DD's provable count minus four** of 80 positive cases.
- Produce zero false accepted host-verified Horn proofs.

At least **six of eight** fixed original seeds must satisfy all three conditions at both depths for `STOP_TRANSFER_PASS`. A complete source-checked run failing that bar yields `STOP_TRANSFER_FAIL`. Missing seed/audit/source evidence yields `INCONCLUSIVE`. The original AF-26 `RULE_ONLY` and earlier scientific verdicts remain unchanged.

## Registered secondary mechanism diagnostics

For every positive **initial** host state, [gap.py](gap.py) records the native Direct and VI-2 top-two *legal-rule* score margins, top legal rule identities, source STOP-minus-top-rule margins, and whether each source ranker selects an action within the **independently audited exact optimal rule set**. The label is attached **after** selecting actions and never used to determine model predictions.

The saved two-step node ledger is decomposed into successor value, interior STOP cost and minimum continuation cost for each legal root action. A recorded `interior_stop_binds` flag identifies when this internal STOP branch dominates the min-plus score. A scientific distinction follows between a misordered legal rule (value-estimation issue) and a STOP-gate error (termination calibration issue). Report the native margins in separate units, with special counts for near ties below `1e-5`.

Every full per-step neural ledger and physical Horn proof is independently replayed from model-visible tensors, with sampled original checkpoint forwards. Provenance includes original artifact IDs, per-file SHA-256 and source-locked protocol/manifest hashes.

## Source-only checks

```bash
python -m unittest discover -s experiments/activation_funnel_af27/tests -v
python -m experiments.activation_funnel_af27.gym > af27_manifest.json
```

For a scoring run (Python 3.12 and `torch==2.5.1+cpu` with NumPy):

```bash
python -m experiments.activation_funnel_af27.evaluate \
  --seed 17 --original-zip-dir original --out /tmp/af27-seed-17

python -m experiments.activation_funnel_af27.audit \
  --seed 17 --original-zip-dir original --output-dir /tmp/af27-seed-17
```

Use `original/af23-branch-seed-17.zip` from the immutable source pinned to [AF-23 CI #43812708916](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43812708916).

The [AF-27 source-locked hosted workflow](../../.github/workflows/activation-funnel-af27.yml) runs smoke, independent eight-seed proof audits and the registered aggregate. Preserve every negative seed and every failed runner. Keep AF-27 as a draft, unmerged branch for independent review.
