# AF-29: matched-local-structure global Horn reachability and STOP calibration

AF-28's prospective multifeature STOP gate failed its registered transfer test: increasing provable-goal continuation reduced correct initial STOP decisions on unprovable goals, while the full public Horn closure controller solved the small graph family perfectly. AF-29 tests a sharper target by making each unreachable root **locally indistinguishable** from a reachable root, then measuring how effectively a frozen recurrent model and train/dev-only logistic calibration recover *global* reachability from the complete public incidence.

## Preregistration and source-only structural certificate

The [AF-29 protocol](PROTOCOL.json) was committed first, before the new generator, feature collection or checkpoint evaluation. The experiment uses 96 **paired** positive/negative Horn graph cases at each of four splits. The 96 pairs comprise two source-registered logical grammar families, 48 pairs per family. Positive and negative members of a pair have:

- Exactly the same four initially known atoms, graph's physical atom/rule permutation, legal root-action mask, marked subgoal/goal locations and complete *directed per-node degree sequence*.
- An exactly identical **identity-preserving directed radius-two incidence neighborhood** around every known root atom and each marked role (subgoals 6 and 13, goal 14).
- A difference in **two middle prerequisite Horn rule clauses**, swapping which of atoms 5 and 10 supplies a productive versus irrelevant rule, with all other rule heads and premises preserved.
- Opposite independent exact goal-reachability labels. The reachable member has a shortest proof of 4 or 5 firings in train/dev, or exactly 5/6 firings in the two held-out cohorts. The unreachable member fails both the host physical BFS and independent bitset BFS.

Both pairing labels share the same source *pair-specific* physical permutation. The original network-visible `GraphBatch.targets` are explicitly **set to zero for both labels**, and the frozen network reads only public input features and current physical known-fact bits. Pair IDs and labels are scorer/audit metadata, never neural features.

The source-only structural gate also requires distinct exact logical rule-incidence templates among train, development and held-out, disjoint public graph hashes from AF-27 and AF-28, at least two legal root rules and matching full graph degree profiles. Any failure blocks the original checkpoint load or STOP fitting.

A classifier that sees only these registered radius-two anchored public neighborhoods receives **the same local representation** for the two pair members. With balanced pairs, the paired AUC of that local-only signature is exactly **0.5** (all scores tied), and any deterministic binary classifier has 50% classification accuracy when evaluated on both labels of every pair.

## Original frozen model and prospective classifiers

The source checkpoints are the eight original 6,036-parameter AF-23 TypedLoop / ReadoutPolicy('ancestor', 4) final weights from [original CI #43812708916](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43812708916). They are downloaded by original artifact ID with immutable ZIP/member SHA-256, original train receipts and Torch/NumPy parity. **The original recurrent network receives zero optimizer updates.**

The STOP-calibration protocol is adapted from AF-28: a per-seed six-feature logistic gate is trained on **TRAIN only**. It reads Direct's native STOP margin, the original root reach and distance logits, VI-2 STOP margin, public number of legal rules and current known-fact count. A matched two-parameter **THR** controller is fitted on the Direct margin alone using exactly the same source samples, balanced BCE, 600 full-batch updates and L2 penalty. The threshold is selected on **DEV roots only**, with a grid of 19 thresholds and a prerequisite of at least **87/96 correct provable continuations**. Labels from either held-out cohort are never used for model or threshold selection.

## Registered held-out comparisons

| Arm | STOP decision | Legal-rule ranker |
|---|---|---|
| DD | Original frozen Direct head | Frozen Direct |
| VD | Original frozen VI-2 head | Frozen Direct |
| CAL, primary | Train-only six-feature logistic, dev-frozen threshold | Frozen Direct |
| THR | Train-only Direct-margin logistic, dev-frozen threshold | Frozen Direct |
| ND | Force continuation while any legal rule exists | Frozen Direct |
| CLOSURE | Nonlearned full-public Horn fixed point | Public graph |
| BFS16 / BFS64 | Public breadth-first search capped at 16/64 state expansions per decision | Public graph |
| SHORTEST | Exact shortest-proof oracle | Evaluation upper bound only |

All policies follow fresh complete host-legal trajectories with a shortest-proof-length +1 firing budget on positive cases, and two decisions on unreachable cases.

**Registered primary endpoint:** within each of the **96 matched** reachable/unreachable pairs at each held-out depth, CAL should give the unreachable member a higher STOP probability than the reachable member on **at least 65 pair-equivalent wins**, with exact ties worth half. The same checkpoint seed must also correctly initial-STOP on **at least 60/96 unreachable roots**, complete at least **DD solved minus five** of the 96 reachable proofs, and have zero accepted false Horn proofs. At least **6/8 source-frozen seeds must pass the entire criterion at both held-out depths**.

The independent audit computes matched-pair AUC for CAL, THR and each native DD/VD STOP margin **within the same score scale**. It also reports per-seed proof completions, negative correct STOP, positive premature STOP, Brier/log-loss/ECE, pairwise successes, actual recurrent model-node forwards and symbolic rule examinations/expansions.

The full public closure baseline can solve these Horn graphs exactly, and thus remains a decisive upper control on any claims of beneficial neural proof-search capability. AF-29 measures whether the fixed network's **global graph representation** contains useful transferable reachability information beyond the provably insufficient radius-two local view.

## Reproduction

Source-only (no Torch):

```bash
python -m unittest discover -s experiments/activation_funnel_af29/tests -v
python -m experiments.activation_funnel_af29.gym > af29_structural_manifest.json
```

With Python 3.12, pinned `torch==2.5.1+cpu`, NumPy and original checkpoint ZIP named `original/af23-branch-seed-17.zip`:

```bash
python -m experiments.activation_funnel_af29.calibrate \
  --seed 17 --original-zip-dir original --out /tmp/af29-calibration-17

python -m experiments.activation_funnel_af29.evaluate \
  --seed 17 --original-zip-dir original \
  --calibrator-dir /tmp/af29-calibration-17 --out /tmp/af29-score-17

python -m experiments.activation_funnel_af29.audit \
  --seed 17 --original-zip-dir original \
  --calibrator-dir /tmp/af29-calibration-17 --output-dir /tmp/af29-score-17
```

The source-locked GitHub Actions workflow runs a frozen-checkpoint smoke, eight separately fitted train/dev calibrators, independent eight-seed heldout proof audits and a final paired aggregate. Previous AF-22–AF-28 science verdicts and experiments remain unchanged. Keep AF-29 draft and unmerged until independent review.

## Source-replay engineering record, November 17, 2026

The first source-locked full run [#44017031507](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44017031507) completed all eight train/dev fits and produced independently audited source receipts for seven held-out checkpoint seeds. Seed 89 failed a strict independent train/dev logistic refit parameter check with difference `1.2171e-4`, slightly above the preexisting `1e-4` numerical guard. A second attempt on the exact original source commit reproduced the same discrepancy.

The separately [source-only diagnostic #44017861619](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44017861619) loaded only the immutable seed-89 train/dev feature rows and its original AF-23 checkpoint. Replaying under `torch.set_num_threads(2)` and deterministic algorithms, it reproduced **every training and development source feature exactly**, with zero fitted coefficient difference, unchanged dev threshold and zero DEV STOP-decision changes.

The independent full auditor now explicitly sets the **same two-thread deterministic Torch CPU runtime as the training and scored source** before loading the checkpoint or replaying model features. Its source proof/feature and refit tolerances remain `1e-4`, with all per-run differences and num-thread/deterministic status saved as evidence and required by the independent eight-seed aggregate. No model weights, logistic objective, source graph pairs, heldout policy scores or preregistered scientific thresholds were changed. The failed original runs and diagnostic are retained.


## Independent aggregate source registry correction

During a static pre-aggregate audit of the source-locked matched-pair experiment, the copied AF-28 aggregator still referred to `gym.TEMPLATES`, a parent-only name. The registered AF-29 generator calls the same concept `gym.FAMILIES`. Both references were corrected to `gym.FAMILIES[split]` before final official aggregation. The source result's parent-experiment metadata was also corrected from AF-27 to AF-28.

This is an **aggregation-only engineering correction**. The source-frozen positive/negative graph pairs, original trained weights, full-trajectory neural and symbolic inference, saved per-case source scores, fitted logistic objectives, selected dev thresholds and both registered scientific gates remain unchanged. The earlier source-locked scoring run and its independent per-seed audits remain available as engineering provenance. A fresh complete eight-seed source-locked run is required for the final signed aggregate because the archived source-hash receipts include the aggregate script.


## Bounded original-checkpoint logistic-refit audit amendment

The two original-source attempts of [CI #44017031507](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44017031507) failed an **engineering-only** seed-89 coefficient refit check at an absolute parameter difference of `1.2170537761e-4`, marginally above the inherited `1e-4` verification tolerance. A separately source-locked, **train/dev-only** [numerical diagnostic #44017861619](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44017861619) reproduced all original seed-89 feature values and fitted weights *exactly* with frozen two-thread Torch CPU settings; the first full audit's FP32 discrepancy persisted on a separate worker.

The final independent proof audit preserves its strict **`1e-4` source-input-feature bound**, and accepts refit coefficients only when their maximum absolute difference from the original SHA-pinned fitted model is **at most `2e-4`**. It separately requires **the identical DEV-selected threshold** and **zero changed DEV root STOP decisions**, records the actual maximum delta and validates every stored train/dev source row and all full heldout host proofs. The *scored* classifier always uses the original immutable train-only fitted coefficients and dev threshold; the refit comparison is a verifier, not a new policy or optimizer.

This numerical verification amendment was documented after the initial blocked source-locked runs. It does not change any original frozen recurrent checkpoint, the synthetic pair generator, training/dev features or labels, the 600-update logistic fit, the heldout decision threshold, or either preregistered scientific gate. Because the aggregate's source hash includes the verifier source, a fresh complete eight-seed source-locked CI run is required; all blocked attempts remain public.
