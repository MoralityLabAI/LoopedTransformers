# AUX-GYM V2: depth-conditioned algorithmic state

Status: engineering candidate, no trained V2 results. V1 source and all original dataset bytes remain unchanged. Base 8ca81f432b8ee6286c9eb35bbcd4c93277d50a6e.

## Source audit

Masked pointer chase deliberately hides a source on the queried path with probability 1/2 at positive depth; ABSTAIN=1101. V1 held-out majority share is 0.4935 (987/2000), validation majority share 0.50925 (2037/4000, rounded in the index). A constant chosen on calibration must be frozen before held-out evaluation. Per-token macro recall (17 classes) and equally weighted ABSTAIN/node exact are distinct estimands.

V0 modular depth-one operations are all 16*17=272 ordered multiplier/offset pairs. Enumerating the canonical structure hash and sorted region boundaries gives train=144, monitor_train=73, validation=16, monitor_calibration=14, locked=20, ood_stress=5. The 2448 distinct possible depth-one train examples follow from 17 starts. The validation region can contain 272 inputs (16*17); the shipped shard includes 211. V0's empty staircase tuple causes random.choice to fail; its modular solver rejects empty lists, so merely adding staircase_depths=(0,) is insufficient. V2 creates all 17 training-only zero-op identities using an explicit Solution. Empty structure belongs to the train region. It is not an ordinary V0 training pool and must never be compared as a byte-identical reconstruction.

V1 README claims distinct groups for all rows; the index says modular validation has 3489 groups for 3711 examples and Sudoku validation 3941 for 4000. V1 repeated-final-answer deep supervision is not algorithmic trace supervision. The published checkpoints' raw prediction files are outside git. Re-score only verified matching prediction receipts.

## Run commands

From the repo root:

    python -m unittest discover -s experiments/recurrence_monitorability_v0/aux_gym_v2_20261126 -p 'test_*.py'
    python -m research_gym.interp.aux_gym_v2 audit --root experiments/recurrence_monitorability_v0/aux_gym_20261126/datasets --pools --out /tmp/aux-v1-audit.json
    python -m research_gym.interp.aux_gym_v2 build --out /tmp/aux-v2 --train-quota 64 --eval-quota 8
    python -m research_gym.interp.aux_gym_v2 pilot --data /tmp/aux-v2 --out /tmp/aux-v2-smoke --family modular_recurrence --arm LOOP-T --mode symbolic_trace --seed 1 --steps 8 --batch 8
    python -m research_gym.interp.aux_gym_v2 rescore --shard SHARD.jsonl --predfile PREDICTIONS.jsonl --calibration CALIBRATION.jsonl --out /tmp/rescore.json

Audit verifies published file bytes, optionally decompressed payload hashes, and all six families' histograms. It independently checks two families' semantic answers. Full regeneration of V1 pools remains separate: run prepare_rmp_train_pool at the original sizes in a new directory; compare ordered tokens, targets, depths and hashes. Torch serialization may differ across environments; logical equality and byte equality are separate receipts. Skip unsupported V0 modular staircase and record failure explicitly.

V2 samples exactly equal token quotas within each depth, keeps structure regions disjoint across V2 splits, uses unique fingerprints, and fails on any shortfall. Default per-depth quotas are deliberately small for an engineering pilot. Modular training uses only individually familiar operations; locked_familiar uses new sequences of familiar ops and locked_novel uses exactly one op absent from training, with disjoint structure groups. Masked has one locked split. The uniform-token distribution assigns ABSTAIN 1/17, so report separately from the natural V1 prevalence and binary-outcome macro exact. Fresh V2 generation uses a new seed but locked labels are visible on disk; independence from any prior V1 evaluation or training is not certified. Freeze receipts, withhold locked files from training processes, and deduplicate against prior exposed shards before calling it a sealed test.

## Registered development comparison

Both existing model architectures at 8 visits, same width, batch size and fixed optimizer steps, seeds 1,2,3,4,5,6,7,8. No first-crossing stop rule. Four modes from scratch: final, repeated_final, symbolic_trace and shuffled_trace. Shuffled trace rolls labels within the batch and is a preliminary negative control; confirm with a within-depth shuffle across independent examples and matched label support. Auxiliary heads use the native tied lens; final CE is present in every arm. Symbolic supervision uses visit i+1 state, holding terminal state after task completion, with final visit reserved for final CE. The 0-depth identity examples are intentionally oversampled by the depth-balanced pilot sampler; add an identity-free arm before interpreting a difference. Validate training traces do not enter model inputs.

Pilot code runs fixed steps and saves development predictions/checkpoint with hashes. It does not read locked shards, implement a curriculum, estimate FLOPs or resume. Keep optimizer steps matched within architecture and measure actual FLOPs, including auxiliary losses, across modes. Use a separate matched-parameter comparison if wanted. Development budget: 1000 steps for smoke comparison, then freeze larger budget before confirmatory runs. Lock root-level paired effects with uncertainty; examples are not independent model roots. Report exact, macro recall, node accuracy, abstain precision/recall, class support, depth-conditioned accuracy, and constant comparator chosen on development. On locked masked data report the natural-distribution and uniform-token evaluations separately; no extrapolation claim from a majority-only win. Use an independent evaluator and fixed test-time budget. Primary OOD gate proposal: root-paired gain of at least 0.05 macro recall over repeated-final, lower 95% paired root CI above zero, masked node exact at least 0.25, and positive gain in each of depths 9-12. Register separately before running locked evaluation. Modular familiar/novel results are separate; report chance 1/17 and trained-majority.

## Next gated implementation: BPD / MMS

Only after symbolic trace supervision shows a credible development gain:
* Fit shared BPD state factors on monitor-training traces, align component identity on a distinct development split; never use locked labels.
* Compare ordinary symbolic loss, BPD component reconstruction, trace-aligned intervention loss, random matched-rank sham, and state-label permutation. Log factor drift, component rank, reconstruction residual and causal trace recovery.
* At each visit predict next symbolic state; patch one factor from a matched counterfactual whose true next state differs. Use mask_move, edge_rewire, op_change siblings and matched unchanged-target controls; recompute the entire trace, preserve region disjointness, and score both predictions and intervention specificity.
* MMS 0/4/16/64 MiB RAM tiers: persistent maps keyed by family/depth/visit/factor, hard byte cap, measured RSS, fixed update FLOPs and verifier-call budgets. Compare random/cache-size-only controls; log effective resident bytes and eviction/lookup counts. No extra visits or batches hidden in the memory arm.
* TRM adapter and tied-loop extended-visits sweep are separate experiments. FF-U 8 visits versus LOOP-T 8 visits is the primary compute-controlled architecture comparison. Extended tied visits are a distinct test-time compute arm.
* Implement these adapters, locked evaluator and counterfactual datasets before launching confirmation; they are specified here, not implemented in this patch.
