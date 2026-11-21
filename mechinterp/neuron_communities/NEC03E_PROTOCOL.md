# NEC-03E | Attenuation-only paired-dose causal diagnostic

**Status:** exploratory read-only extension of the [NEC-03D failed matched-dose feasibility pilot](NEC03D_RESULTS.md). This protocol was written after the NEC-03D outcomes were exposed and therefore carries a **posthoc development** evidence class. It is intended to calibrate what kind of physical matched-donor intervention is possible before attempting a fresh confirmatory NEC-03 study.

## Original evidence and fixed inputs

- Full six-checkpoint PEF-07 CPU archive and 20 source-discovered communities: [NEC-02B run 44790504320](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44790504320), artifact **14655419843**. The source-manifest SHA is `de4e0263864ee70c2bf94ca95008ace10fce1323d1a51b4c8829ec26e40bbd9b`; the historic November 15 local digest discrepancy remains in parent PR #67.
- Source-trained SAE checkpoints and sealed feature-ranking manifests: [NEC-03D run 44860214701](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44860214701), artifact **14689443261**. Each SAE uses the native 64-dimensional memory-token GELU coordinates and trains only on 80 PEF-07 training contexts. Feature choices were frozen on 80 independent PEF training contexts and do not use the already-opened PEF heldout labels.
- Primary retrospective model/task examples in the pilot: seed 103 coalition and threshold. The same original 48 heldout context groups are reused and should not be relabeled as independent confirmation.

## Physical-dose policy

Consider two candidate donor vectors \(d_a,d_b\) from the same clean input and recurrent visit, and let \(W_\mathrm{down}\) be the frozen 32 × 64 MLP-down weight matrix. Compute the input-conditioned direct physical doses

\[
r_a=\|W_\mathrm{down} d_a\|_2,\qquad
r_b=\|W_\mathrm{down} d_b\|_2.
\]

The **common achievable dose** is \(r_*(x,t)=\min(r_a,r_b)\). The modified donors are \(d'_a=(r_*/r_a)d_a\) and \(d'_b=(r_*/r_b)d_b\), with the respective gain zero when its denominator is zero. Both gains lie in [0,1]. **No donor is amplified.** This allows exact local norm matching, including zero-dose samples, and avoids the unsafe eightfold amplification threshold that failed for every NEC-03D candidate.

Run these three separately dose-matched pairs:

| Comparison | Left donor | Right donor |
|---|---|---|
| Native specificity | Frozen native community | Train-frequency-matched, hub-disjoint native sham |
| Native versus SAE | Frozen native community | Source-ranked SAE features |
| SAE specificity | Source-ranked SAE features | Same-count source-ranked alternative SAE features |

Each pair is injected into its own live hubs-only baseline at the fixed native community's recurrent visits. Its recovered count and difference between donor arms is computed over 48 independent eight-bit context quartets (16 related histories each). The prepared SAE and native source rosters are read-only and checked by SHA256. The raw artifact preserves the common target, full original doses, attenuation ratios, achieved doses and exact logits for both interventions at every source context and recurrent visit.

## Feasibility and power limits

Two separate physical gates prevent a small shared dose from masquerading as a meaningful causal comparison.

1. **Numerical matching:** at least 99% of per-site/example pairs must have achieved output norms within 1% of each other. This is an engineering requirement and should ordinarily pass under attenuation-only scaling.
2. **Retained signal:** at least 80% of site/example pairs must have positive common dose \(>10^{-6}\), and the median common dose must preserve at least 25% of the left donor's original direct output norm. Failure means the comparison has too little effective perturbation, even though numerical dose matching succeeded.

All three pairs are analyzed, including zero-gain cases, vanished correction signal and negative rescue. The selection of strongest candidates in already seen PEF-07 heldout data remains posthoc; no Holm-corrected significance or cross-task generalization is claimed. The independent source unit is the context group within each fixed model, and all candidate arms reuse the same 48 contexts.

A future prospective NEC-03 needs new independently seeded model/task roots, calibrated matched-sham variation, minimum meaningful improvement on an exact behavioral endpoint, and a frozen familywise test specification. **Current FSI-01 gate: REDESIGN_OR_PILOT_ONLY.**

## CPU replay

GitHub Actions uses original frozen six-model weights from NEC-02B and the sealed SAE artifacts from NEC-03D. It executes source-preserving pairwise interventions, runs standalone unit tests and an independent source/physical-dose correctness audit, and uploads complete raw evidence. No GPU, native-model training or SAE refitting is needed.

~~~bash
for task in coalition threshold; do
  python -m mechinterp.neuron_communities.nec03e_pairs \
    --root /tmp/nec03e --out "/tmp/nec03e/nec03e/seed103_$task" \
    --seed 103 --task "$task"
done
python -m mechinterp.neuron_communities.audit_nec03e --root /tmp/nec03e --seed 103
~~~
