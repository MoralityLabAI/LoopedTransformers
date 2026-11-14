# FC-08: trained recurrent actor against bounded symbolic AND/OR search

**First frozen evaluation: 13 November 2026, successful.**
[GitHub Actions #43444713589](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43444713589)
completed both train-only and held-out jobs, compiled **352 model-
proposed positive Lean theorem terms**, and independently verified
another **380** negative Boolean countermodels. There were
**zero false finite formal certificates** across 768 seed/case
evaluations.

The [public permanent research report](https://github.com/MoralityLabAI/Compactification-Theory/blob/feat/refutationbench-fc08-conjunctive/research/refutation_bench/results/fc08_conjunctive_first_run_20261113.md)
preserves full provenance, paired counts, all cost measures and the
registered interpretation.

## Frozen source and training

Registration committed first:
b6c8ffbef6606e89737bbfc32d73b8732b765e8d

Training source only:
51d0082f27e4c6214ec0cebb4886df44a4962321

Training JSONL SHA-256:
0d51135f606d44085f2bf0643e44469c36cbebae2469c5ceb8dcbb52b75f763a

Test source:
1408daf07c34a133a794662c31f8fbdf789ac840

Test JSONL SHA-256:
3b7d93c04e6565fc8e46e30947c0347d75ad6b61d575aeb7a598e6c189969dd1

[Source train CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43442906680)
verified 960 train/dev cases and Lean-compiled 480 true source
theorems. [Source test CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43444039591)
verified 384 held-out cases and compiled 192 positive reference
theorems. Every valid goal requires a multi-premise inference;
**no single-premise path reaches it**.

Both seeds 17 and 29 trained a separate 11,521-param four-pass
LR-01 gate on FC-02 train, with theta=**0.55** calibrated using
FC-02 development alone. An independent actor containing the
original 11,521-param tied cell and a 4,705-param rule-ranking
head was jointly trained for 300 AdamW steps on **1,344 FC-08
training-only positive proof-prefix states**. Both trained
components together have 27,747 parameters. In-sample
next-rule accuracy on the first 128 train prefixes was 81.25%
and 84.375%, respectively.

The first training job checked out only the frozen
train-source commit and asserted the held-out generator
and symbolic control code were absent. A dependent
evaluation job loaded SHA-256 checkpoint artifacts,
then checked out the test-source revision. Model input
features excluded the reference proof and ground truth.

## First held-out performance

| Seed / split | Learned positives /96 | 24-pop best-first /96 | Learned negatives /96 | Abstained |
| --- | ---: | ---: | ---: | ---: |
| 17 / familiar depth | **94** | 89 | 96 | 2 |
| 17 / deeper | **72** | 28 | 96 | 24 |
| 29 / familiar depth | **96** | 89 | 92 | 4 |
| 29 / deeper | **90** | 28 | 96 | 6 |

Other pre-registered source controls: familiar/deeper
random-legal 53/5, goal-first 68/18,
backward-ancestry 48/9 and unary-path BFS 0/0.
The exact unrestricted shortest-proof oracle
scores 96/96 on both splits by construction.

The registered **positive-proof learning criterion passed**
on both seeds and both test depths against the
bounded best-first primary baseline. Exact paired
model-only/baseline-only positive counts were:
5/0, 45/1, 7/0, 62/0 in table order.

The **compute-efficiency criterion failed**. The learned
actor uses four tied-cell forward passes at each selected
rule, beyond the fixed gate's own recurrent cost.
Learned vs best-first total cells per 192 tasks were
2,450 vs 1,078 (seed17 familiar);
2,926 vs 1,006 (seed17 deep);
2,472 vs 1,096 (seed29 familiar);
2,918 vs 998 (seed29 deep).

Measured CPU milliseconds per certified positive:
learned/best-first **7.39/3.49**, **11.53/10.51**,
**7.46/3.53**, **8.64/9.64**, respectively.
Only the final seed/depth condition favors learned
compute per certificate. The full symbolic
24-state expansion cap is separately recorded.

## Exact certificate and artifact contract

Every learned positive proof consists of at most five
original source-rule IDs in the *model-selected order*.
The independent checker replays the sequence, verifies
every premise is previously available and the exact
goal is derived. Only those successful sequences
are rendered as Lean source, which the CI job
compiles using pinned Lean 4.19.0.
Negative releases require exact complete
Boolean countermodels. All failed proof
attempts and stopping-gate abstentions are retained.

The evaluation artifact
fc08-conjunctive-43444713589
holds FC08_report.json, **24 prediction JSONL files**
(total 4,608 per-case outputs across six policies,
two seeds and two splits), model-generated Lean source
and the compiler log. A separate archive inspection
verified that all original model traces respect the
five-action cap, the predicted files contain 192
unique cases each, and the Lean source has 352
theorem declarations matching the SHA256 in the
score report. GitHub artifact retention is 30 days.

The result establishes useful **budgeted, learned
rule selection** on newly generated conjunctive
Horn graphs compared with a preregistered
24-state symbolic best-first heuristic. It does
not establish a performance advantage over
unrestricted exact symbolic BFS or over all
AND/OR graph algorithms. The corpus is public
finite logic, not a source-to-semantics evaluation
of untrusted mathematical proof manuscripts.

## Reproduce

Pin the two public source checkouts to the commits
above. In private LoopedTransformers, Python3.11
with CPU PyTorch2.5.1:

~~~sh
export FC08_TRAIN_ROOT="$(realpath ../Compactification-FC08-train)"
python -m unittest discover -s experiments/refutation_transfer \
  -p 'test_fc08_trial.py' -v

python experiments/refutation_transfer/fc08_trial.py train \
  --source "$FC08_TRAIN_ROOT" --out /tmp/fc08_weights

python experiments/refutation_transfer/fc08_trial.py eval \
  --source "$(realpath ../Compactification-FC08-test)" \
  --weights /tmp/fc08_weights --out /tmp/fc08_results

lean /tmp/fc08_results/FC08_model_proofs.lean
~~~

The Actions two-job workflow enforces the strict
train/test source separation for the registered run.
