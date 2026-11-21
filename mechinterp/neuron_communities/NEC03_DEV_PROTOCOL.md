# NEC-03D: physical-dose native-neuron versus SAE comparison

**Status:** source-locked retrospective **development** study. This implements the preparatory half of [NEC-03 issue #239](https://github.com/MoralityLabAI/LoopedTransformers/issues/239), building on [NEC-02B's six-model 20-community pilot](NEC02B_RESULTS.md). The original 48 heldout PEF-07 contexts have already been observed in earlier studies. These numerical results belong to development even though the *new* SAE fit and feature selection use only source-training contexts.

## Actual comparison

Native features are the **64-dimensional GELU activations entering \`block.mlp_down\`**, at the recurring memory token, for each of 12 tied-weight visits. Feature IDs pair a channel with its visit. A source-trained overcomplete 128-feature SAE encodes the *same* GELU vector. We compare native candidate neurons chosen by the old attribution community graph, a new hub-disjoint source-frequency-matched native-neuron sham, source-only top-attributed SAE features with the same number of occurrences per visit, and a budget-matched alternative SAE feature set.

SAE fitting uses the first **80 complete 16-state context quartets** of the PEF-07 parameter-training cohort (1,280 histories × 12 visits). The next **80 training contexts** select SAE features by activation × downstream-gradient. Both cohorts are source-disjoint and exclude PEF probe and heldout contexts. Model weights remain frozen. SAE fitting uses exactly 240 CPU AdamW steps, coordinate normalization fitted only on the training group, 128 ReLU features, a linear decoder and 0.002 mean latent L1 penalty. A \`prepare.json\` and \`source_sae.pt\` are SHA256-sealed *before* evaluation can read heldout.

The SAE contribution in physical GELU coordinates is \(z_f D_f\), where the decoder columns include the source-only coordinate standard deviation. We add selected features to the exact native model's GELU activation, preserving the no-op baseline when the feature list is empty, rather than forcing full SAE reconstruction every visit.

## Fixed intervention family

At every visit involved in a selected native community, the experimental baseline retains only native always-on hub channels of the **live** GELU vector. It then adds a **clean-trajectory donor** from the same input's original unperturbed GELU vector at that visit. The prompt token, other visits, residual path and attention remain active.

| Arm | Additive donor at the selected recurrent sites |
|---|---|
| \`hubs\` | Zero |
| \`native\` | Original activation of candidate native neurons |
| \`sham\` | Original activation of same-visit, source-frequency-matched hub-disjoint neurons |
| \`sae\` | Source-ranked SAE feature decoder contribution |
| \`sae_sham\` | Source-ranked alternate SAE features with identical per-visit cardinality |

The direct physical dose is the norm of the frozen \`mlp_down\` applied to the selected native donor, measured in the 32-D MLP output basis. For each example/visit, control donors are scaled by \(g=\min(8,d_{native}/\max(d_{control},10^{-10}))\), with gain zero for zero native dose. Scaling uses activations and frozen weights, independent of outcome labels. Complete target/realized norms, gains, clipping flags and mismatches are logged in the raw evidence.

**Dose comparability gate:** at least 90% of each candidate's source-visit/example control doses must match the native target within 5%. Failures remain in the results, flagged as noncomparable. Gains saturating at eight can fail this gate.

This **clean-graft** comparison differs from NEC-02B's original \`retain\` condition, which preserved live candidate activations inside an altered trajectory. Their effects should be compared with that distinction intact.

## Research evidence scope

The behavioral endpoint is **correct predictions recovered relative to hubs-only**, while matched native sham and alternate SAE features provide dose-comparable controls. We also store KL, full clean/intervention logits and source-context bootstrap distributions. The independent unit within one frozen checkpoint is the eight-bit context, which contains 16 related focal variants. The same 48 context groups are reused across seeds and candidate screens. Bootstrap intervals remain descriptive, with no Holm adjustment or calibrated meaningful-effect threshold. The original PEF heldout families were already exposed before NEC-03D. **FSI-01: REDESIGN_OR_PILOT_ONLY.**

This pipeline can diagnose context-dependent recovery and measure whether a given trained SAE feature basis rescues more correctly classified cases than native units under comparable direct MLP-down dose. It cannot establish a unique Boolean circuit, general SAE superiority, self-model properties or transfer to pretrained language models. A 128-feature autoencoder with a weak sparsity penalty may have high L0 density, so reconstruction MSE and mean active-feature count must be disclosed.

## Fixed sources and CPU commands

Input: [NEC-02B hosted CPU archive](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44790504320), artifact \`nec02b-six-model-raw-causal-and-crossseed-receipts\`. Its six checkpoint hashes and frozen source/selection rosters are verified before use. Historical November 15 local source-manifest differences remain documented in PEF-07 PR #67. This adapter uses the hosted source SHA256 \`de4e0263864ee70c2bf94ca95008ace10fce1323d1a51b4c8829ec26e40bbd9b\`.

With PyTorch 2.10.0 CPU, NetworkX 3.4.2, and pytest:

~~~bash
python -m pytest -q mechinterp/neuron_communities/test_nec03_development.py
for task in coalition threshold; do
  python -m mechinterp.neuron_communities.nec03_development prepare \
    --root /tmp/nec03 --out "/tmp/nec03/nec03d/seed103_$task" \
    --seed 103 --task "$task"
done
for task in coalition threshold; do
  python -m mechinterp.neuron_communities.nec03_development evaluate \
    --root /tmp/nec03 --out "/tmp/nec03/nec03d/seed103_$task" \
    --seed 103 --task "$task"
done
python -m mechinterp.neuron_communities.audit_nec03 --root /tmp/nec03 --seed 103
~~~

The GitHub Actions CPU workflow runs the same order and archives SAE weights, provenance hashes, complete raw logits and dose audits. NEC-03's eventual confirmatory study will need fresh task and training roots, a frozen hypothesis family with Holm correction, independently calibrated sham variation and a minimum meaningful behavioral effect.
