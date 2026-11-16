# AF-23: STOP calibration and branching-proof controls

This source-registered follow-up responds to the [corrected AF-22 report](../../docs/research/AF22_RESULTS.md), whose **scientific verdict is FAIL**. AF-22's fixed first-legal control solved 75% and 80% of provable held-out cases at depths 4 and 5, respectively; an AF-23 capability claim must clear that baseline. The original AF-22 weights, protocol, artifacts, and source are frozen.

**Protocol registration:** [PROTOCOL.json](PROTOCOL.json), committed first on this new branch before creating this gym or evaluating AF-23 held-out outcomes.

## Track A: frozen-checkpoint STOP ablations

- **Controls:** original Direct, VI-1, VI-2, fixed first-legal, and shortest-proof oracle from immutable AF-22 Actions archives.
- **Interventions:** root-tied reachability, no-interior-STOP, their combination.
- **Oracle separation:** root-tied reachability uses only the model's own reach logit at the root. It is then reused across all nonterminal descendants of that receding-horizon state; goal membership remains an exact host terminal boundary.
- **Full rollout:** recompute the original model's state neighbourhood at *each actual new state* with the original checkpoint. Evaluate the intervention and execute the chosen legal rule. Cached original trajectories are useful only for first-action audits.
- **Verification:** SHA-256-check original ZIP and member bytes; load AF-22's source-bound checkpoint; audit every new rule firing from independent physical incidence, recompute every min-plus decision and replay sampled genuine Torch forwards.
- **Primary hypothesis:** root-tied/no-interior combined must beat fixed by five percentage points on both held depths and correctly STOP on at least 90% of unprovable cases in at least six of eight seeds. All original failed baselines remain in the score table.

With original artifact ZIPs downloaded as `af22-seed-17.zip` through `af22-seed-131.zip` into a directory:

```bash
python -m experiments.activation_funnel_af23.stop_eval \
  --zip-dir /path/to/archives --seed 17 --out-dir /tmp/af23-s17
python -m experiments.activation_funnel_af23.stop_audit \
  --zip-dir /path/to/archives --seed 17 --out-dir /tmp/af23-s17
```

These commands use the original Torch model and require the AF-22 dependency stack; the source-only numerical scorer is pure Python.

## Track B: proof-branching gym

The new [branching gym](branching_gym.py) uses the existing public **15-atom, 14-rule** model interface. It places six enabled rules at provable initial states (four decoys and two productive choices). The shortest goal path requires three, four, or five legal rule firings, depending on the registered split. Rules are physically permuted separately for each case; train/development layouts 170–175 do not overlap held-out layouts 202–206.

| Split | Cases | Designed provable | Layouts |
|---|---:|---:|---|
| Train | 768 | 320 | 170–173 |
| Development | 384 | 160 | 174–175 |
| Held layout | 32 | 16 | 202 |
| Held depth 4 | 192 | 80 | 203–204 |
| Held depth 5 | 192 | 80 | 205–206 |

The independent bitset BFS cross-checks the host-state shortest-proof oracle. Structural preflight requires at least five initially enabled actions, at most half optimal, exact registered shortest distances, disjoint public feature hashes, and at least 25% first-legal failure on the positive held-out cases.

```bash
python -m unittest discover -s experiments/activation_funnel_af23/tests -v
python -m experiments.activation_funnel_af23.branching_gym > /tmp/af23_branching_manifest.json
```

The **Track B neural training/evaluation** is a separate eight-seed, 240-update experiment requiring all five random-legal streams, fixed and shortest-proof controls in the final table. Neither mathematical consistency nor an easy benchmark qualifies as a positive proof-search result.

## Status discipline

1. Commit preregistration, graph generator, host/oracle preflight, neural interface, and independent audit.
2. Verify a source-locked CI preflight and publish the deterministic manifest.
3. Only then release new held-out scores. Track A and Track B have independent verdicts; report every failed seed.
4. Keep AF-23 draft and unmerged pending receipts. Preserve original AF-22 negative evidence.
