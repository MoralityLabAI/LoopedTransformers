# COG-03: recurrent routing over checked finite proof ledgers

The run trains a 24-wide transformer cell and reuses its weights across four passes.
Each proposed rule application is checked against the current established facts.
The tasks are synthetic Horn proof DAGs exported by DataGoblin COG-02.

## Measured run

| Split | Arm | Completed roots / trials | Mean verifier calls | Relevant facts per call |
| --- | --- | ---: | ---: | ---: |
| calibration | looped | 128/128 | 7.461 | 1.0000 |
| calibration | one_pass | 128/128 | 8.469 | 0.8792 |
| calibration | reset_history | 124/128 | 7.727 | 0.9171 |
| calibration | reset_inner | 127/128 | 10.383 | 0.7156 |
| calibration | shuffled_history | 123/128 | 8.531 | 0.8425 |
| calibration | random | 18/128 | 23.227 | 0.1396 |
| calibration | ready_first | 126/128 | 13.891 | 0.5489 |
| calibration | goal_ancestry | 128/128 | 7.656 | 1.0000 |
| ordinary_test | looped | 128/128 | 7.453 | 0.9979 |
| ordinary_test | one_pass | 128/128 | 8.375 | 0.8862 |
| ordinary_test | reset_history | 123/128 | 7.938 | 0.8947 |
| ordinary_test | reset_inner | 128/128 | 10.281 | 0.7173 |
| ordinary_test | shuffled_history | 123/128 | 9.047 | 0.7945 |
| ordinary_test | random | 16/128 | 23.414 | 0.1395 |
| ordinary_test | ready_first | 126/128 | 13.578 | 0.5593 |
| ordinary_test | goal_ancestry | 128/128 | 7.609 | 1.0000 |
| ood_test | looped | 125/128 | 15.055 | 0.9549 |
| ood_test | one_pass | 122/128 | 16.188 | 0.9112 |
| ood_test | reset_history | 87/128 | 17.102 | 0.6830 |
| ood_test | reset_inner | 101/128 | 17.703 | 0.7820 |
| ood_test | shuffled_history | 70/128 | 18.930 | 0.5803 |
| ood_test | random | 0/128 | 24.000 | 0.0833 |
| ood_test | ready_first | 50/128 | 21.016 | 0.5799 |
| ood_test | goal_ancestry | 128/128 | 15.031 | 1.0000 |

Seeds reuse the same root worlds. The trial denominator includes those repeated measurements.
Each split contains 64 distinct roots, with one presentation per root.

## Training

| Seed | Inner passes | Parameters | Updates | First-20 loss | Last-20 loss | Teacher checker calls |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 4 | 5811 | 180 | 1.2992 | 0.0275 | 38588 |
| 17 | 1 | 5811 | 180 | 1.5932 | 0.2832 | 38588 |
| 29 | 4 | 5811 | 180 | 1.3848 | 0.0198 | 41274 |
| 29 | 1 | 5811 | 180 | 1.7339 | 0.1604 | 41274 |

The fixed final checkpoint supplies each evaluation. Training receives only train roots.
The recurrent model and the one-pass model share the architecture and number of optimizer updates.
Their inner-cell compute differs by depth. Every policy receives the same maximum verifier calls.
A shared public rule prevents retrying a rejected action until the ledger changes; conclusions already known are skipped.
The policy sees current facts, public rules and previous checker feedback. Source IDs, split names and teacher labels stay outside its input.

## Controls and evidence

`reset_history` clears carried state at each verifier decision. `reset_inner` resets state before every inner pass while paying for four passes.
`shuffled_history` rolls the same carried vectors across rule positions. `one_pass` is fitted independently at one inner pass.
The symbolic `goal_ancestry` policy supplies a structural ceiling for this finite task family.
The independent Python checker establishes validity in the declared Horn calculus. DataGoblin's separately reported Lean replay covers its exported certificates.
Evidence class: public development pilot. The architecture was frozen before this run; a calibrated causal power gate remains open.
These measurements concern routing in generated propositional tasks. They provide a training and replay interface for later native Lean corpus experiments.

## Reproduce

```bash
python -m research.cog03_ledger_router.run --corpus research/cog03_ledger_router/data/tasks.jsonl.gz --output /tmp/cog03-replay
python -m unittest discover -s research/cog03_ledger_router/tests -v
```

`registration.json` freezes roots and arms. `episodes.jsonl` retains completed and unfinished roots; `attempts.jsonl.gz` records every checker call.
`training.json` counts teacher label checks separately from evaluation calls. Checkpoints and optimizer traces support a full replay.
