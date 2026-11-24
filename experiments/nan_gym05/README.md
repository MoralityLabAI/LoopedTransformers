# NAN-GYM-05

Prospective development evaluation of native Nanbeige across the wider local
suite, with read-only layer/module states and a small CAL-trained SAE tower.
Read [PROTOCOL.md](PROTOCOL.md) and [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json).
The original 168-case ARG-03 test stays sealed and BPD transfer stays disabled.

The source cohort has 322 contexts/backend, 89 source-root groups (32 CAL and
57 DEV), one pinned checkpoint and separate BF16/NF4 conditions. The included
290 public contexts exclude the local routing corpus's 32 contexts. The
combined model prompt/gold hashes are registered; raw routing prompts, token
IDs, trajectories, predictions and hidden arrays must remain local.

## Native inference

Use the existing offline pinned snapshot/environment. Reconstruct the combined
cohort only from the explicitly registered routing corpus and public source
modules. Verify its hashes, source locks and the archived engineering fixture
before any model load. Save new fixture rechecks separately: elapsed metadata
means their receipt bytes will differ from the original registered fixture.

    python -m experiments.nan_gym05.run --backend bf16 --compactification-root ../Compactification-Theory-arg02 --snapshot PINNED_LOCAL_SNAPSHOT --runs runs/nan_gym05
    python -m experiments.nan_gym05.run --backend nf4 --compactification-root ../Compactification-Theory-arg02 --snapshot PINNED_LOCAL_SNAPSHOT --runs runs/nan_gym05

Run sequentially, HF_HUB_OFFLINE=1, no automatic downloads. At most one explicit
hash-bound --resume per backend, for the same pending roots, preserving failed
attempts. A successful 4 GB run using this placement is not evidence that the
full model was resident or that shared allocation was paging.

## Separate host grading and tower fit

    python -m experiments.nan_gym05.grade --runs runs/nan_gym05 --out runs/nan_gym05/graded

summary.json is aggregate/publication-safe. local_graded_records.json stays
local because it includes routing responses. --allow-partial supports clearly
labelled progress reports; no source/threshold/recipe is changed using it.

After a complete backend run and process exit, fit its four shared-loop
calibration dictionaries. Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting
Python for deterministic CUDA matrix operations:

    python -m experiments.nan_gym05.sae --backend bf16 --runs runs/nan_gym05
    python -m experiments.nan_gym05.sae --backend nf4 --runs runs/nan_gym05

The fitting code refuses DEV input, incomplete CAL coverage and overwrites.
It fits exactly 256 vectors/layer from 32 CAL pointer roots, excludes copy
controls and uses the frozen final-step recipe. The untrained dictionary is
retained. The language model is not trained or edited.

Rank-8/16 effective-weight SVD baselines are algebraic controls, not learned
VPD/BPD. Operator-plus-residual numerical closure is measured separately from
native kernel parity. Actual trained VPD/BPD cross-references need compatible,
provenance-bound assets and the correct physical site/loop/backend. No missing
asset or failed competence gate is silently replaced by an easier claim.

## CPU contracts

    python -m unittest discover -s experiments/nan_gym05/tests -v
    python -m unittest discover -s mechinterp/nanbeige_bpd_gym/tests -v

Public ARC/Sudoku are tiny exposed microbenches, storyworld uses full planned
rollouts under the native rival, and routing measures environment identity,
not Intellect/GSM8K/ARC QA problem-solving. Report all conditions separately.
