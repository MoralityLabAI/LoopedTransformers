# NEC-02 | Hosted trained PEF-07 and neuron community exploratory replay

**Start:** 2026-11-21. This is a *new CPU execution* and a posthoc mechanistic pilot. The PEF-07 November 15 preregistration and historic local six-checkpoint results remain intact. [NEC-01](README.md) supplies the discovery/causal instrument.

## Why

The original PEF-07 local checkpoint bundle (`pef07_12bit_six_checkpoint_local_evidence.zip`) was created in an earlier session and is not accessible as binary bytes in the connected GitHub repository. Its six SHA256 hashes and result tables remain in `research/pef07/LOCAL_RESULTS.md`. GitHub Actions can reproduce the *training protocol* on free hosted CPU and generate its own sealed byte hashes. Any differences in host math or checkpoint serialization will be documented; these new checkpoints are never silently represented as the original November 15 models.

## Manifest digest discrepancy

The November 15 local report records the source-manifest digest `849d039e47dec472b0b4835d24928e9a547e262bf84fd7dfed16741df9911ac1`. Independently reconstructing the partition from the **committed PEF-07 implementation** produces `de4e0263864ee70c2bf94ca95008ace10fce1323d1a51b4c8829ec26e40bbd9b`. This discrepancy is tracked explicitly. NEC-02 independently reconstructs all 256 context memberships directly from the registered SHA256 sorting algorithm, verifies full 16-cell assignment closure and the hosted canonical digest, and preserves both digests in the audit output. Until the original local manifest file can be read and diffed, the provenance relationship between the two historical serialization formats remains unresolved. This pilot is a fresh hosted execution.

## Pipeline (single hosted CPU job)

1. Pin CPython 3.12, PyTorch 2.10.0 CPU, NetworkX 3.4.2, 1 CPU math thread and oneDNN AVX2.
2. Run PEF-07's existing tests and re-execute the original **exact six 450-step training streams** across coalition/threshold × seeds 103, 131, 167. PEF-07 seals all six weights before scoring. This is an **exploratory new execution**, since the historic task results have already been viewed.
3. Verify the committed code's complete context-closed source manifest `de4e026...`, six checkpoint bytes, 450-step metadata, full roster and task-head accuracy receipts.
4. For each of the two **seed-103** models, run NEC-01 source-only neuron attribution on 160 training contexts (16 related histories each); freeze the roster, graph and frequency-matched shams to SHA256. Run ablation, frequency sham, retain, hubs-only and blank on all **48 heldout context groups**. Record original clean-head behavior, descriptive KL and correctness deltas, and context-cluster bootstrap intervals for every candidate, including zero-candidate outcomes.
5. Audit outputs mechanically and publish an Actions artifact with all six new checkpoints, original PEF-07 training/evaluation receipts, both discovery graphs and full causal effect tables.

## Evidential scope

- **Independent unit:** Each heldout eight-bit context, containing 16 counterfactual old-four-bit variants. 48 units per fixed model.
- **Model independence:** All six models are deterministic executions of the already studied seeds. Only seed 103 receives NEC-01 interventions in this pilot, once per separately trained task.
- **Primary exploratory causal contrast:** KL(clean || native-community ablation) minus KL(clean || same-visit frequency-matched sham ablation), averaged by context. A directional measure of necessity is a reduction in clean-model confidence or correctness under the target intervention beyond matched sham.
- **Additional descriptive contrast:** KL(hubs-only) minus KL(community + hubs retained), restricted to the community-bearing visits and MLP residual branch.
- **Physical dose:** Set native GELU activations to exactly zero, on the memory token, at frozen visit:neuron occurrences, then continue the model naturally.
- **Unresolved sham variation floor:** Frequency matching can leave activation-scale, gradient and downstream importance unmatched. The 48 contexts and reused fixed seeds do not support a calibrated 95th-percentile independent-world null floor at 95% confidence. EvalMath's rule recommends >=59 independently generated sham roots even to use their maximum as the one-sided tail bound.
- **Minimum meaningful effect:** Pending independent development calibration, so no new success rule or familywise significant result is asserted here.
- **Current verdict:** `REDESIGN_OR_PILOT_ONLY`. The purpose is to establish a measurable signal and cost, then preregister NEC-03 using fresh source/task/model seeds and a stronger dose-matched sham.

An unchanged native class after ablation can coexist with a large KL. A large KL by itself is insufficient to establish the causal Boolean-operation circuit, and community retention still leaves attention and residual computation intact. The learned threshold teacher is a helpful function-level control with different class structure, so per-task effect sizes need their clean-score baseline reported.

For source-locked SAE comparisons, the next step is to align NEC-01's GELU pre-down tensor with an SAE dictionary *trained and evaluated only on the same discovery source*, preserve equal intervention sites and match active output perturbation magnitude. Existing LS-SAE01 studies concern different model states and cannot be substituted without a new adapter and coordinate audit.

## Reproduce with GitHub Actions

Workflow: [`nec02-hosted-trained-pef07.yml`](../../.github/workflows/nec02-hosted-trained-pef07.yml)

To replay the full CPU training and exploratory analysis in an environment that already has `torch==2.10.0`, `networkx==3.4.2`, and `pytest`:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DNNL_MAX_CPU_ISA=AVX2 PYTHONHASHSEED=0
python -m pytest -q research/pef07/test_pef07.py mechinterp/neuron_communities/test_nec01.py mechinterp/neuron_communities/test_audit_hosted.py
python -m research.pef07.pef07 --out /tmp/nec02/pef07

for task in coalition threshold; do
  python -m mechinterp.neuron_communities.pef07_adapter discover --checkpoint-dir /tmp/nec02/pef07 --out /tmp/nec02/analysis/seed103_${task} --task "${task}" --seed 103
  python -m mechinterp.neuron_communities.pef07_adapter evaluate --checkpoint-dir /tmp/nec02/pef07 --out /tmp/nec02/analysis/seed103_${task} --task "${task}" --seed 103
done
python -m mechinterp.neuron_communities.audit_hosted --root /tmp/nec02
```

The final `nec02_audit.json` is a strict manifest, generated from both tasks and all frozen checkpoint bytes. This workflow uses only GitHub's public CPU runner. A failed or timed-out run should be reported as engineering failure, with partial receipts retained. Future GPU experiments and larger independent model seeds require a separately frozen plan.
