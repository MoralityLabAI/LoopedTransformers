# Deuterium paper completion and optional Silico work

**19 November 2026.** The completed manuscript is [deuterium_complete_20261119.md](deuterium_complete_20261119.md), with a [PDF edition](deuterium_complete_20261119.pdf). Its scope is an audit of historical control claims, the completed synthetic and native steering studies, and the measured behavior of a recurrent VPD adapter. Existing CPU results support that scope. A further Silico run would extend the research question.

## Claim ledger

| Claim | Evidence available | Scope and disposition |
| --- | --- | --- |
| The replication contains 16 accepted grafts | Complete archived-row recount; per-seed counts 3, 4, 1, 0, 8, 0 | All accepted grafts tie their selected guarded random controls on task deltas. Acceptance comes from the logit-movement tie breaker. Retain this result with the rule that produced it. [Historical audit](../../experiments/deuterium/HISTORICAL_AUDIT.md). |
| Historical recovery reproduces | Hosted audit reports 38 comparable output files byte-identical | This establishes reproduction of the archived analysis. The source ledger records which historical runs have raw rows. [Hosted audit](../../HOSTED_REPLAY_AUDIT_20261119.md). |
| Synthetic head steering versus selected random steering | Three independent model/decomposition seeds; 2,048 held-out rows per seed | Hosted mean random-minus-VPD target NLL is +0.0008899869669 nats; 95% t interval [−0.0001600500971, +0.0019400240309]. Registered verdict: `NO_REGISTERED_ADVANTAGE`. [Exact hosted summary](evidence/hosted_gate/paper_summary.json). |
| The synthetic result survives independent saved-data verification | 6,144 rows; 265/265 checks pass | The verifier checks arithmetic and artifact consistency from saved outputs. The hosted replay adds reproducibility on already exposed sources. [Verification receipt](evidence/hosted_gate/verification.json). |
| Native recurrent-weight steering versus selected random steering | Three previously examined teacher roots; the same 256 held-out strings for each root; 17 saved arms | Mean random-minus-VPD goal NLL is −0.00002608100808 nats; descriptive 95% t interval [−0.0002873765254, +0.0002352145092]. Registered verdict: `DEVELOPMENT_NO_REGISTERED_ADVANTAGE`. [Exact recovered verdict](evidence/native01/registered_verdict.json). |
| Stock VPD reuses the final-pass importance mask across recurrent calls | Four physical calls observed; one cached input; the same mask object used four times | With the original fitted CI networks, first-pass mean absolute mask mismatches are 0.27464578, 0.26785272, and 0.29798546. These measurements identify adapter behavior. Causal attribution for the steering outcome remains untested. [Original fitted-CI receipt](evidence/cache_audit/old_train_source_only.json). |
| Pointwise stock equivalence under forced-identical masks | The original nine tests passed in Actions; the expanded ten-test suite passes locally on excluded untrained seed 61077 | Tests cover logits, losses, gradients, one AdamW update, and the upstream leaky-importance range. [Original run 44483849471](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44483849471); [current tests](../../native_trm/adapter02/tests/test_pass_indexed.py). |
| Strict full-schedule optimizer trajectory parity | Stock, forced-last-repeat, and native-pass arms each complete 32 warmup plus 64 main updates with matched initial state, batches, and random draws | The stock-versus-last-repeat trace exceeds the added 2e-5 trajectory tolerance. Release status: `BLOCKED_UNMATCHED_OPTIMIZER`; all fresh roots remain unopened. [Source-only receipt](../../native_trm/adapter02/results/source_schedule_20261119/schedule_receipt.json). |
| Historical task-level steering in the recovered 11,191,328-parameter classifier | 91/91 frozen tensors match; original forward and dataset classes remain absent | Status: `BLOCKED_ARCHITECTURE_PROVENANCE`. The archived configuration has `reasoning_steps=1`. Preserve the matrix audit and the unresolved source requirements. [Recovered preflight](evidence/hosted_gate/recovered/recovered_preflight.json). |

The synthetic and native studies have different architectures and endpoints. Their seed labels also refer to separate models. Each result retains its own interval. The native teachers' prior use and shared evaluation cohort restrict that study to development evidence. An equivalence claim would require a registered margin and an interval contained within it.

## Full-schedule engineering outcome

The new source-only check carries the existing absolute tolerance of 2e-5 through the complete fitting trajectory. This adds an engineering condition beyond the protocol's forced-identical-mask pointwise comparison. Every arm starts from the same factor and importance-network state on the excluded 61077 source. All arms complete 32 warmup and 64 main optimizer updates; the pinned optimizer also evaluates the main loss at its terminal step, producing 65 main loss evaluations and 67 adversarial draws including evaluation calls.

Warmup parameters and gradients match exactly. The largest stock-versus-forced-last-repeat discrepancies over main fitting are 0.000583648681640625 for loss, 0.00035858154296875 for parameters, and 0.00011508853640407324 for gradients. These exceed the strict trajectory threshold, so the prospective release remains `BLOCKED_UNMATCHED_OPTIMIZER`. The ten pointwise tests still pass. The frozen twelve teacher roots and every reserved data split remain unopened. See the [first complete schedule receipt](../../native_trm/adapter02/results/source_schedule_20261119/schedule_receipt.json).

FP32 rounding amplified by the sign-based PGD update is a hypothesis about the divergence. The present receipt establishes the discrepancy and the matched schedule; identifying its cause requires a focused numerical diagnostic. Such a diagnostic can run on the same excluded CPU source, with any revised scientific requirement recorded prospectively before opening fresh test data.

## What Silico could contribute

The remaining work with a specific historical payoff is recovery of the large classifier's original execution environment. Silico would be useful if its retained workspace contains files absent from the recovered archive. The first request would be an artifact search with a manifest of the files found.

### Optional provenance-recovery packet

**Question:** Can the authenticated recovered classifier be executed with its original input encoding and checked against independently preserved original outputs?

The existing [preflight receipt](evidence/hosted_gate/recovered/recovered_preflight.json) identifies the target checkpoint SHA-256 as `72b7d6225adcf3f3b07b4ae9b6769729f09085943012ac989d88a22aa447e2de` and the decomposition checkpoint as `dd4dc46093e3b3ea79b54c7ba628c892d22e70a1496cb8367d66ffc4224eab9c`. A useful return packet contains the following items:

| Required item | Acceptance evidence |
| --- | --- |
| Original `experiments.hermes_skill_gym.train_hrm_hermes.HermesHRMClassifier` source | Complete source, its repository revision or original file path, and SHA-256 |
| Original `experiments.hermes_skill_gym.hermes_dataset.HermesSkillDataset` and tokenizer/input encoding | Exact preprocessing, vocabulary or tokenizer revision, label ordering, and a serialized input batch with hashes |
| Historical runtime | Python and dependency versions, including `x-transformers`; original invocation and configuration where retained |
| Independent reference logits | Original inputs, expected tensor shapes, saved logits, byte hashes, and a record of how these outputs were produced |
| Replay receipt | Strict checkpoint load, 91 frozen-tensor equality checks, reference-logit comparison at a declared tolerance, clean outputs, and measured CPU/GPU time and memory |

Return `RECOVERED_PROVENANCE_INCOMPLETE` with the searched locations and missing items when the source bundle remains incomplete. A reconstructed architecture would begin a separate experiment. Fresh steering, retraining, and held-out evaluation belong after successful provenance review. The completed paper retains the archived outcome while this optional recovery remains open.

### Optional prospective adapter experiment

[DEUTERIUM-NATIVE-02](../../native_trm/adapter02/PROTOCOL.md) already registers the stock-last versus pass-indexed comparison. It can become a later paper extension after the optimizer discrepancy is resolved and the complete scoring implementation is frozen. Its present experimental contract is:

| Field | Registered value or current evidence |
| --- | --- |
| Unit and count | 12 independent new teacher initializations planned; 0 opened |
| Cases | 256 new scored strings per teacher, 3,072 across the planned study; nested within teacher roots |
| Primary contrast | Mean held-out goal NLL under stock-last minus pass-indexed steering |
| Physical dose | Actual applied FP32 rank-one change, 1% of the original shared recurrent weight's Frobenius norm |
| Scientific variation floor | 0.0002 nats as an exploratory engineering target |
| Matched-sham floor | Population tail and confidence level remain uncalibrated; repeated tokens cannot supply independent sham roots |
| Planned comparisons | One registered primary contrast; each teacher receives 32 signed calibration candidates per learned arm and 32 candidates from one shared random bank |
| Actual comparisons | 0 new-root scientific comparisons |
| Resolution and power | Twelve-root t interval and ≥9 positive root effects are registered development gates; a population power claim needs an independent development estimate and measured sham variation |
| Exposure | Reserved training, validation, calibration, and test sources remain unopened |
| Causal controls | Shared random bank, literal zero edit, clean model, separately described label-supervised task gradient, exact weight restoration, rank/dose checks |
| Current verdict | `BLOCKED_UNMATCHED_OPTIMIZER` at the added full-trajectory engineering gate; scientific comparison remains unscored |
| Provenance | [Protocol committed before implementation](https://github.com/MoralityLabAI/LoopedTransformers/commit/0512c69d63a16b58fbcf838292d1394f4d8d18e3); pinned upstream `2df0092d5be4d8e4748b3c66ef8325dedef17549` |

The release packet must freeze the teacher trainer and both decomposition paths, then retain paired batches, optimizer traces, all selected matrices, and their hashes. An exposure marker precedes each first test generation. A model replay produces neural logits; the independent stdlib auditor recomputes metrics and statistical decisions from the saved logits. Any required clarification to the prospective protocol belongs in a versioned amendment before new test exposure.

Measured CPU runtimes favor completing the numerical diagnostic in the existing environment. The full source-only schedule took 4.69 seconds for stock, 4.20 seconds for forced last-repeat, and 4.10 seconds for native-pass fitting. The earlier three-teacher fitted-CI cache diagnostic took 2.2923 seconds. In the original native calibration run, teacher 103's fitting and signed-candidate search command completed in about 12 seconds, excluding environment setup and downloads. A full future study needs its own runtime estimate after its prerequisites pass. Sources: [schedule receipt](../../native_trm/adapter02/results/source_schedule_20261119/schedule_receipt.json), [cache receipt](evidence/cache_audit/old_train_source_only.json), and [native calibration job](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44463652662/job/135357327416).

## Durable evidence

The `evidence/` directory preserves exact small receipts from completed Actions artifacts. Each subdirectory has a custody manifest giving its original run, artifact identifier, ZIP digest, and hashes of copied members. The [native result](evidence/native01/provenance.json), [fitted-CI diagnostic](evidence/cache_audit/provenance.json), and [hosted synthetic replay](evidence/hosted_gate/provenance.json) remain inspectable after their Actions downloads expire. Large original checkpoints and per-example raw archives retain their original artifact references.
