# Recovered-record audit

`records.py` reads the authenticated extraction and writes deterministic JSON and CSV. It never imports recovered Python or unpickles recovered checkpoints. The archive-member SHA-256 values in `results/retrospective/audit.json` bind every material input, while the accepted-graft export retains exact JSONL line numbers and the random comparator used by the historical rule.

```bash
python experiments/deuterium/records.py \
  --records-root /path/to/authenticated/extraction \
  --out-dir experiments/deuterium/results/retrospective
python -m unittest discover -s experiments/deuterium/tests -p 'test_records.py' -v
```

## Replication count and acceptance

The June 1 paper replication contains 1,305,171 graft rows, including 131,376 `vdp_or_class_row` candidate rows. The 110,731 stable candidate rows and 16 accepted rows agree with the summary. The summary's `best_grafts` list holds 12 displayed examples. The archived source truncates that list to `top_grafts`, whose default is 12; the run command leaves this argument at its default, although its precise executing commit remains unauthenticated.

| Seed | Full accepted count | Summary preview |
|---|---:|---:|
| 20260601 | 3 | 2 |
| 20260602 | 4 | 2 |
| 20260603 | 1 | 1 |
| 20260604 | 0 | 0 |
| 20260605 | 8 | 7 |
| 20260606 | 0 | 0 |

The four omitted examples appear on sweep lines 4578, 4721, 5751 and 5801. There are 14 distinct seed/source/target/component combinations among the 16 accepted rows because two combinations recur at multiple scales. Nine rows use `head_svd` and seven use `head_row`. Fourteen target `repair_step`, while two target `candidate_verify`.

Every accepted row ties its best guarded random comparator on all task metrics. The five-element historical ranking tuple is `(maximum preservation/logic gain, -false_commit_delta, -false_reject_delta, accuracy_delta, -mean_abs_logit_delta)`. Each accepted row wins through the last element, which rewards smaller output perturbation. This supports a retrospective perturbation-efficiency observation under the selection rule. The count supplies no task-metric superiority over random. Trials were selected from a large sweep on the same scoring rows used to calculate acceptance, and no multiplicity correction is recorded.

Random directions are generated at matched rank and Frobenius norm for each mapped source component, then evaluated at each scale. Acceptance pools guarded random rows over every source, component and scale for the same seed and target organelle. The exported best comparator can therefore come from a different parameter dose. Eight random draws per source-component mapping do not mean the acceptance threshold uses only eight rows.

The source's guardrails permit an accuracy decrease of 0.05 and a false-reject increase of 0.05, although the accepted rows themselves have zero changes in those guardrail error rates. Their accuracy improves by one correct decision in 193 or 194 scored rows. The arena uses SVD/head-row directions; its historical `vdp_or_class_row` label cannot authenticate learned adversarial VPD components.

The earlier May 31 full-arena summary separately reports 23 accepted rows, 389,746 records, 93 organelles and 54,295 stable grafts. That sweep is absent from the recovery package. `MISSING_FILES.md` says it was not requested during collection. Its historical summary remains available, and the current audit supplies no independent recount of that earlier experiment.

## ARC and Wordle identities

The four ARC cards have 128 rows drawn from 62 trajectory IDs. All repeated IDs have identical score arrays and target labels. The historical file-fold validation deduplicated training and held-out files separately. It left 27, 28, 30 and 26 repeated training IDs in the respective 32-row held-out cards for seeds 23, 37, 101 and 151. Their truly unseen portions therefore have 5, 4, 2 and 6 rows. The previously selected `D over A, margin <= 1, penalty = 1` condition makes zero changes in correctness on each unseen portion. Across all 62 cached examples it improves 44 correct decisions to 48, with four rescues and zero damage. Both calculations describe the recovered cache; the unseen-row analysis was performed after the historical condition had been selected.

Wordle's observation identity requires `(trajectory_id, step)`. Four cards have 76 scored rows, 72 distinct observation pairs and 71 trajectory IDs. The four-row discovery card is contained in the eight-row discovery card. The two later 32-row cards have no shared trajectory IDs with the other cards. The seed-37 card contains `w_808` at steps 0 and 6, with different valid target actions. Treating trajectory ID alone as observation identity would collapse those two observations incorrectly.

Replaying the archived `post_initial_gate` and anti-`[crane]` penalty reproduces every saved per-row decision. Card scores are 2/4 to 4/4, 4/8 to 8/8, 11/32 to 17/32, and 9/32 to 17/32. The unweighted mean card gain is 0.359375; pooling all recorded rows gives 20/76, or 0.263158. Keeping the first occurrence of each observation pair gives 24/72 to 42/72, a gain of 0.25 with 18 rescues and zero damage. The archival candidate sets and scores can be replayed, while missing source puzzle provenance limits stronger claims about independent episodes or full-game success.

## Governance and adapters

The governance report records target/nontarget accuracy of 0.6500/0.6775 before the dirty patch, 1.0000/0.2975 after it, and 1.0000/0.0100 after decomposition-informed cleanup. Its exact invocation, per-row identities, dirty patch and projection coefficients were not recovered. Available source trains and scores the patch on the same target slice, then uses both scored slices to choose the cleanup. Individual component projections are subtracted without accounting for a nonorthogonal Gram matrix. The rank-4 dirty patch and top-8 SVD-removal defaults erase the whole patch; the historical invocation remains unknown. The associated precision model is a feedforward TF-IDF classifier.

Adapter result JSONs preserve improvements for the earlier static SVD arm and negative outcomes for learned-component arms. The learned-component 12-step run reports bucket accuracy 0.500000, compared with 0.515625 for baseline and random. The validation-selected run leaves every arm at 0.515625 and increases loss from 1.001876 to 1.004116. Its zero-step variant reduces loss to 0.999113 while accuracy remains unchanged. Selected validation-scored coefficients start at signed nonzero values, while random coefficients start at zero. Equal component counts thus leave initialization and dose unequal.

The archived adapter dataset contains 13,086 training rows and 3,270 evaluation rows with disjoint row IDs. Exact text overlaps cover 666 evaluation rows. Splitting literal text before `\nMODEL_ACTION:` identifies task context: 3,117 evaluation rows share a task prefix with training, leaving 153 evaluation rows from 51 unseen task prefixes. This operational identity test requires no semantic judgment. The result files supply counts, while exact historical subset IDs and commands remain missing. Bucket accuracy measures the critic's reward-bucket prediction rather than direct solution of the logic problem.

## Recovered roundout tables

The exported `roundout_*.csv` files reproduce 37 metric rows, five policy rows, ten experiment rows and six historical claim rows. They retain the original values and labels as historical reported collation. The current manuscript must distinguish these author judgments and deterministic estimates from independently recounted sweep rows, cached-score replay and a newly executed prospective comparison.
