# FC-06: strict five-action LR-01 proof-choice result

**First frozen CI #43415640416** (November 13, 2026) passed
training-only unit tests, frozen checkpoint evaluation, exact
countermodel/trace verification and Lean 4.19.0 kernel compilation
of all 384 accepted **model-selected positive proof terms**.
[Workflow](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43415640416).

The [public first-run result receipt](https://github.com/MoralityLabAI/Compactification-Theory/blob/feat/refutationbench-fc06-binding-budget/research/refutation_bench/results/fc06_strict_five_step_20261113.md)
provides complete per-family scores, structural limits, exact
source provenance, and post-hoc controls.

## Frozen data protocol

- [Registered before source/test](https://github.com/MoralityLabAI/Compactification-Theory/blob/81f53bcf606e45d8b3fea9acba7907641c339cab/research/refutation_bench/fc06/PREREGISTRATION.md).
- FC-06 train-only source checkout: `1bf514325788b633432eb9f3e49627bd923a216e`.
- Canonical 960 train/dev cases hash:
  `c841727d6a3e82c5ce5769b2b324e9bde078d15930d03fad5462292657d73ccf`.
- New test generator checkout:
  `63c7c411e32a25b447b1ca92ae0db9e33b135d27`.
- Canonical 384 test cases hash:
  `6d570d3c6b21466c8df0032cac0cab115ebf3a586e82160f18b0a96f9efb9d9e`.
- [Source train CI passed](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43414157412)
  with 480 compiled valid Lean proofs.
- [Source test CI passed](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43414735943)
  with 192 compiled valid Lean proofs and independent
  verification of 192 negative countermodels.
- First model train job checked out only the earlier commit,
  asserted test_data.py absent and uploaded SHA-identified
  frozen weights before the dependent evaluation job saw
  the new test generator.

The FC-04 negative/provisional gate used a newly trained
11,521-param four-pass tied LR-01 graph model. It trained
on FC-02 train, with θ=0.55 for both seeds selected on
FC-02 development only. A **separate actor** copied the
trunk and trained its 11,521 parameters jointly with a
4,705-param rule-ranking head, 300 optimizer steps on
1,149 FC-06 train-only proof-prefix states. Total two-network
parameters: 27,747. The actor's logged training-only
128-prefix accuracy was 100%.

## First held-out neural result

Every positive source goal has an exact shortest derivation
of at most five rules, in a graph with ten reachable novel
consequences. The action budget was **five**, and
distractor selection could use it up.

| Method | Positives /96 same depth | Positives /96 deep | Negatives /96, both splits |
| --- | ---: | ---: | ---: |
| Joint LR-01 seed17 | **96** | **96** | **96** |
| Joint LR-01 seed29 | **96** | **96** | **96** |
| Random legal | 4 | 2 | 96 |
| Immediate-goal-first | 26 | 4 | 96 |

Every accepted positive rule trace was replayed by the
independent strict five-rule verifier, then rendered and
compiled **from the actual selected action sequence**.
All negative claims retained complete verified Boolean
countermodels. **Zero false formal certificates**.
The 384 positive Lean proofs contain seed/case repetition;
there are 192 distinct positive test cases.

The learned actor met both preregistered criteria against
the two weak policy baselines in both splits/seeds: higher
positive certified coverage and more verified positives
per 1,000 *total neural cell invocations*. On a 192-task
test split, the actor used 2,112 cells (same-depth) and
2,784 (deep), versus 960 and 1,056 for either heuristic.
Corresponding positive certificate efficiency was
45.45 and 34.48 per 1,000 calls for the actor, compared
with goal-first's 27.08 and 3.79.

## Stronger post-hoc counter-control

[FC-06 post-hoc backward-slicing theorem](https://github.com/MoralityLabAI/Compactification-Theory/blob/feat/refutationbench-fc06-binding-budget/research/refutation_bench/fc06/POSTHOC_BACKWARD_SLICE.md)
reveals that these test graphs allow a **zero-parameter
reverse-ancestry graph traversal** to select exactly the
required chain. That policy also certifies all 192 valid
test cases, under Lean, while spending **zero neural
cell calls**. [Its separate CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43416698132)
passed all four regression tests and compiled its
192 proof terms.

The post-hoc control preserves the original registered
first-run result. It prevents a stronger claim of general
neural proof-search superiority. FC-06 has demonstrated
learned transfer of goal-relevant rule choice under
a binding action budget, relative to the registered
random and goal-first heuristics, while a simple
symbolic dependency method still dominates.

A new benchmark must put plausible decoys inside the
goal-ancestry slice, with alternative goal producers and
shared subgoals. Pre-register that stronger static
baseline before training the next actor.

## Reproduce

Use a private LoopedTransformers checkout at the FC-06 PR
and two separate Compactification-Theory checkouts at
the source SHA values above.

~~~sh
export FC06_TRAIN_ROOT="$(realpath ../Compactification-Theory-FC06-train)"
python -m unittest discover -s experiments/refutation_transfer \
    -p 'test_fc06_joint.py' -v

python experiments/refutation_transfer/fc06_joint.py train \
    --source "$FC06_TRAIN_ROOT" --out /tmp/fc06_weights

python experiments/refutation_transfer/fc06_joint.py eval \
    --source "$(realpath ../Compactification-Theory-FC06-test)" \
    --weights /tmp/fc06_weights --out /tmp/fc06_scores
lean /tmp/fc06_scores/FC06_model_proofs.lean
~~~

The GitHub Actions two-job workflow enforces separate
access at training versus test. Its run artifacts retain
every per-case failed/accepted proposal and the Lean log.
Formal claims concern bounded propositional Horn logic,
with no raw human manuscript verification in this trial.
