# Deuterium: recovered TRM/VPD steering study

This is the dedicated home for the Deuterium retrospective and the follow-up steering experiment within Loop​edTransformers/mechinterp.

## Study archive and source provenance

The complete review package, including the frozen protocol, executable implementation, Git bundle, paper PDF/Markdown, checksummed data inventory, regeneration scripts, matched controls, per-seed records, and verification, is saved in the user's ChatGPT Library as **Deuterium_review_package_20261119.zip**:

https://chatgpt.com/library?file_id=libfile_fde3_f-183248d654408191aeaf5de422992eab_c26eca361130819193bd5d8e21b255a7&account_id=personal

Its incremental Git bundle contains:
- base repository: `MoralityLabAI/param-decomp`, base `2df0092d5be4d8e4748b3c66ef8325dedef17549`
- historical audit: `da2c3197fdc02693d00360533ee6a4c8b125ecfd`
- **protocol frozen before held-out scoring:** `f1e3e9ba663ea906a6468d76426bb7381a6e23ad`
- reported results/paper: `2441c4330bb138f3218183eb46cde9c6db9eb903`

The original June 2026 source archive remains on Hugging Face, fetched by an immutable revision and SHA-256 verified:

`https://huggingface.co/datasets/AlephFunk/vpd-trm-original-records-202606/resolve/b208d547d20e89dc6c221bc640525320cc67a2f8/VPD_TRM_original_records_202606.zip?download=true`

Archive SHA-256: `a45db5ef5e603852efb4f247112655c5572eff5aa85f9ab94674a4ae9dc8f003`

Archive: 1,326,795,233 bytes; 406 members, 401 verified original source-file manifest entries; 3,716,109,041 uncompressed bytes.

## Reconciled historical record

Six-seed *replication* accepted **16** grafts (seed counts 3/4/1/0/8/0). The older 12-graft statement is a truncated preview, not the sweep total. All 16 accepts tie the selected guarded random control on the recorded task deltas and pass through a logit-movement tiebreak; the acceptance rule does not show strict task-performance superiority to random. The separate earlier full-arena summary records 23 accepted grafts, but its original sweep remains missing.

The four ARC Challenge cards contain 128 row appearances of 62 distinct trajectories. Held-out cards overlap training cards by 27, 28, 30, and 26 examples of 32. The cached ARC rule yields 44/62 to 48/62 on deduplicated rows, with zero gains on the truly unseen portion of each card.

The Wordle constrained-score rule improves 24/72 to 42/72 distinct observations, with 18 rescues and zero damage. This is a rule-level/candidate-score intervention rather than evidence of a faithful TRM-weight intervention. The governance cleanup fails collateral preservation: from 0.6775 baseline collateral accuracy to 0.2975 under the raw patch and 0.0100 after cleanup.

## Follow-up and status

Recovered 11,191,328-parameter model lineage matches 91/91 frozen tensors. Its genuine task replay is **BLOCKED_ARCHITECTURE_PROVENANCE** because original forward/tokenizer definitions and reference logits are missing.

A distinct synthetic `RecursiveGateTRM` rank-one, equal-dose, equal-search-budget VPD study completed locally after an immutable protocol freeze. Independent seeds 103, 211 and 307 all passed the clean/replay/dose/rank prerequisites.

Random-minus-VPD target NLL differences: **+0.0013782, +0.0006288, +0.0006642 nats**.
Mean: **+0.0008904 nats**; 95% across-seed t interval: **[-0.0001599, +0.0019407]**. The frozen verdict is **NO_REGISTERED_ADVANTAGE**; a task-gradient baseline has lower target NLL in every seed. Maximum observed collateral accuracy drop: 0.145 percentage points. Local seed-region CPU runtime: 10.73 seconds total.

## Materialization in this repository

This folder currently serves as a **review pointer and provenance landing page**. The verified source implementation, detailed results, regenerated paper and workflows remain in the linked review package. They have not yet been copied into this repository, and GitHub Actions for this repository have not run the study. The incremental Git bundle must first be imported into a clone containing the param-decomp base commit, then the directory `experiments/deuterium/`, `papers/VPD_TRM_Feature_Steering/`, and the two related workflow files may be copied into this `mechinterp/deuterium/` folder, with module imports and workflow paths adjusted and tested before a hosted scientific reproduction is claimed.

Import in a **clone of param-decomp**:

```bash
BUNDLE=/path/to/deuterium-review.bundle
git bundle verify "$BUNDLE"
git fetch "$BUNDLE" refs/heads/research/deuterium-recovered-followup-20261119:refs/heads/research/deuterium-recovered-followup-20261119
git switch research/deuterium-recovered-followup-20261119
sha256sum --check experiments/deuterium/results/SHA256SUMS
```

Keep large recovered data off Git and preserve the original bytes. This work is for review; do not merge or publish it automatically.
