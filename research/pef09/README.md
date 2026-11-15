# PEF-09 | Ungated pair-interaction projection and residual recovery

This independent mechanistic study uses six frozen PEF-07 models from [PR #67](https://github.com/MoralityLabAI/LoopedTransformers/pull/67), continuing on top of the still-unmerged [PEF-08 PR #75](https://github.com/MoralityLabAI/LoopedTransformers/pull/75). It changes only this new `research/pef09` subtree and an isolated unit-test workflow.

## Two registrations and decisions

- **PEF-09**: [PROTOCOL.md](PROTOCOL.md), standalone first commit `2bbd53e`. It predeclares pair-conditional selection by probe-only hidden SVD, then an **ungated** rank-one projection across all 16 source patterns, eight closest-energy random nulls, cross-pair and global-PC controls. Its preregistered pair-specificity criterion **failed 0/3 seeds**, despite total majority-class collapse of native heads after the target and global-PC ablations.
- **PEF-09R**: [PEF09R_PROTOCOL.md](PEF09R_PROTOCOL.md), committed `4ea1812` **after** inspecting the PEF-09 outcome and before testing ridge recovery. Its separately preregistered criterion **passed 3/3 coalition seeds**: the frozen h12 head fell to 480/768, while a new ridge fitted on ablated probe contexts recovered 745–768/768 at h12.

[LOCAL_RESULTS.md](LOCAL_RESULTS.md) reports the full tables, essential caveat about inadequate random-control energy matching, and complete provenance.

## Running the nine invariant tests

```bash
python -m pytest -q research/pef09/test_pef09.py research/pef09/test_pef09r.py
```

Run both experiments with PyTorch 2.10.0+cpu; use the **original local PEF-07 checkpoint archive** accompanying this experiment. A new checkout without the archived checkpoints can run the invariant tests but cannot reproduce the six trained-model results.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DNNL_MAX_CPU_ISA=AVX2
python -m research.pef09.pef09 \
  --evidence /path/to/registered \
  --out /tmp/pef09_results
python -m research.pef09.pef09r \
  --evidence /path/to/registered \
  --selection-ledger /path/to/original/pef09_results/frozen_selection_ledger.json \
  --out /tmp/pef09r_results
```

The separately registered PEF-09R recovery requires the original PEF-09 selection ledger with SHA256 `f206d88f5f2ea9703edc385d47bcf6fd6cd11a6cb4a3ece6edeae230dd734157` and fails closed if another ledger is provided. This supports replay against the **same previously frozen directions**. Source-only unit tests run under the new GitHub Actions workflow.

Complete model checkpoints, full per-condition JSON, readout weights, hashes, and execution logs are in the local evidence ZIP, rather than committed as repository binaries.

## Scientific boundaries

The **selection** of SVD directions uses the known pair identities and probe-only model activations; it uses neither task labels nor classifier weights. The **intervention** applies the same affine rank-one projector to all source histories and therefore avoids PEF-08's pair-gated erasure.

In PEF-09, randomly sampled directions capture only 11.7–38.7% of target projection energy at visit 8, so they are inadequate energy-matched controls. The task's hidden geometry is highly dominated by the first PC. PEF-09R's recovery shows information retained in the residual model state for a new linear decoder, while removing the *exact* global top PC leaves the same ridge at majority-class accuracy. This highly angle-sensitive observation does not establish an identifiable unique circuit or transfer to formal reasoning.
