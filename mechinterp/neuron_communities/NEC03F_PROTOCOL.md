# NEC-03F | TopK16 native-coordinate SAE with source-ranked active feature routing

**Research class:** posthoc development, separate from the proposed confirmatory NEC-03 hypothesis family in [issue #239](https://github.com/MoralityLabAI/LoopedTransformers/issues/239).

The first SAE pilot ([NEC-03D](NEC03D_RESULTS.md)) trained a 128-feature ReLU autoencoder on native GELU activations and recovered the source-state with normalized MSE around 0.007, but used roughly 63 active features per input. Its fixed source-ranked features frequently failed to produce physical MLP-down doses comparable with native neuron donors. [NEC-03E](NEC03E_RESULTS.md) fixed the dose mathematics by attenuating two donor vectors to their shared attainable output norm; all 27 pair comparisons matched locally, although all 18 fixed-feature SAE comparisons failed a minimum retained-signal gate.

## The new experiment

This version trains an **actual TopK=16 SAE**, with 128 features and exactly sixteen retained encoder activations per input. Its physical input is the original **64-D native GELU vector** entering `block.mlp_down` at the recurring PEF-07 memory token. We preserve the original trained PEF-07 weights and the previously discovered communities.

**Source-only separation:** first 80 context quartets from the PEF-07 parameter-training set fit the new SAE (480 CPU AdamW steps, 256-example minibatches, MSE loss). The other 80 parameter-training contexts rank SAE features using mean absolute activation × gradient-through-physical-decoder attribution. Every context contains all 16 four-bit historical variants. The source train and dev cohorts contain the same 80/80 partition as NEC-03D; PEF-07's 48 downstream-probe contexts and 48 previously opened heldout contexts are excluded from SAE optimization and feature-ranking.

**Label-blind active-feature policy:** the rank-order list of all 128 SAE features is SHA256-frozen separately for each of the twelve visits, using only source development activations. During evaluation, the SAE computes its TopK16 active codes on the current input's **clean** GELU state. If the corresponding native community has `n` nodes at that visit, the policy selects the `n` highest-ranked *active* SAE features, and an alternative control selects the `n` lowest-ranked active features. Feature identities vary with the input because its active dictionary support varies, but selection uses only activation values and the previously frozen source rank. The full 16-example active feature indices and top/bottom selections are archived and audited against the frozen source ranking. The same `n` per-visit component budget applies to native and SAE donors.

This is an *input-conditioned sparse routing policy*, rather than another fixed-list community. Results cannot be attributed to feature identity alone, since the routing policy is part of the intervention.

## Fixed physical comparisons

Each selected recurrent GELU memory-token visit is reduced to its always-on hubs. Add the donor from that input's clean trajectory. Native community and frequency-matched source-hub-disjoint sham are unchanged from NEC-03E. SAE contributions use the TopK16 decoder in physical native coordinates, with the top/bottom active source-ranked features.

Run all three fixed pairs for each native community, on the original 48 heldout context groups:

1. Native community against matched native-neuron sham.
2. Native community against top source-ranked **active** TopK16 SAE features.
3. Top source-ranked active SAE features against alternate low-ranked **active** features.

For each candidate/pair/input/visit, each donor is attenuated to the *smaller* of their two direct `mlp_down` output L2 norms. Both gains lie in [0,1], and the raw record includes realized doses and attenuation ratios. Matching must succeed to within 1% for at least 99% of site/examples. A comparison also needs 80% positive common-dose sites and a median of at least 25% of the left donor's original dose to pass the **physical-signal feasibility** gate. Doses and ratios are independent of labels.

The original PEF-07 heads and checkpoints remain frozen. Raw source records save all clean, hubs-only, donor-left and donor-right logits for each of the 16 focal configurations; every selected active TopK code and its route; physical norm/gain receipts; source and model hashes; and the independently recomputed class effects.

## Evidential limits and next step

The hypothesis is that active-feature selection may rescue the **physical availability** of feature contributions, making native/SAE fixed-dose comparisons more informative. Improvement in available dose is an engineering outcome; a behavioral advantage over the sham needs a separate causal test with the same dose and a prospective minimum meaningful task endpoint.

The original 48 PEF-07 context groups are already exposed by earlier NEC studies. Existing trained seeds and teacher tasks do not constitute new independent model/task roots. Source-only SAE fitting prevents further training leakage yet cannot reset the evidential status of the evaluation population. This experiment remains **REDESIGN_OR_PILOT_ONLY** under EvalMath FSI-01.

The eventual confirmatory NEC-03 should preregister the SAE architecture, TopK and active routing, native/sham selection, smallest meaningful correctness effect, paired null floor, full familywise hypothesis comparisons, and newly sampled independent task/model roots, with raw paired outcomes and independent checkers.

## Public CPU execution

Input: the six immutable PEF-07 checkpoint/source/community records from [NEC-02B run 44790504320](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44790504320), artifact ID **14655419843**. No local GPU or paid compute is needed.

The workflow first runs unit tests, then prepares both coalition and threshold TopK16 dictionaries in full source-only mode. It checks their SHA256 feature roster, validates disjoint context ownership, evaluates source-matched donors on the heldout groups, independently recomputes labels and physical doses, and uploads complete evidence.

~~~bash
for task in coalition threshold; do
  python -m mechinterp.neuron_communities.nec03f_active prepare \
    --root /tmp/nec03f --out "/tmp/nec03f/nec03f/seed103_$task" \
    --seed 103 --task "$task"
done
for task in coalition threshold; do
  python -m mechinterp.neuron_communities.nec03f_active evaluate \
    --root /tmp/nec03f --out "/tmp/nec03f/nec03f/seed103_$task" \
    --seed 103 --task "$task"
done
python -m mechinterp.neuron_communities.audit_nec03f --root /tmp/nec03f --seed 103
~~~
