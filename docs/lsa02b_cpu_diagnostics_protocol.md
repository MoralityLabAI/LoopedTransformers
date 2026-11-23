# LSA-02B: CPU-only checkpoint and action-choice diagnostic

**Status:** frozen, read-only follow-up to [the completed CPU training pilot](../reports/lsa02b_cloud_cpu_pilot_20261123.md). Source evidence is [Actions run 45167489504](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45167489504). This follow-up reopens **saved training data and calibration episode transcripts only**; it runs no new Lean proofs, optimizer steps, external API inference or locked-test evaluation.

## Question

The two-seed native pilot trained 57,475-parameter Plan B checkpoints on 124 Lean-derived shortest-continuation frames. It closed 1/8 or 2/8 of the same eight calibration proofs, despite a native reference ceiling of 6/8. Swapping AB/BA block order at inference changed no first tactics. What limits the effectiveness of dispatch-order comparison on these roots?

Two distinct explanations have different experimental implications:

- The public candidate menu may contain mostly singleton or otherwise uninformative choices. A recurrent architecture cannot express a different action if only one action is legal.
- The model may see meaningful choices but fail to learn their proof relevance. A low training-root shortest-continuation accuracy would implicate optimization or representation; high training accuracy with poor native calibration would implicate generalization and trajectory shift.

The diagnostic also tests whether AB and BA differ *at later decision states* despite identical first tactics. It compares action decisions only on identical public packet hashes and identical candidate-menu hashes. Divergent later proof states are reported as different trajectories rather than presumed equivalent decision sites.

## Fixed inputs and calculations

\`ppt_gym.cpu_diagnostics\` retrieves the exact artifacts from the successful run:

1. The 24-root, 124-frame native Lean teacher, including standalone compiler and source-hash receipts.
2. Four matched-order Plan B checkpoints at seeds 101 and 103.
3. Four eight-root native calibration studies and their independent replay receipts.

The source train-only choice audit counts (a) singleton menus, (b) menus with multiple tactics, (c) forks with both shortest and nonshortest actions, and (d) forks admitting multiple equally optimal actions. Correlated steps are clustered by independent theorem root.

For each frozen checkpoint, the runner measures optimal-label accuracy on the **original training roots** with both AB and BA, along with continuous output displacement and action flips. It separately reports **root-weighted accuracy on genuinely discriminating forks**, the exact uniform-within-menu optimal-action baseline for those forks, and the number of fork-specific block-order choice changes. Singleton-state correctness is reported separately because it is forced by the public menu. It uses exactly the trained weights and two recurrent passes, with zero updates. Training-root accuracy is a development diagnostic and cannot replace the native held-out proof-completion endpoint.

For each native calibration transcript it counts visited states with singleton or forked menus, actions differing between AB and BA on identical public packets, and full-proof closure including the common native oracle ceiling. Source/weight/episode artifact hashes, proof-root lineage disjointness and identical same-seed initial weights must all verify. An incomplete or altered artifact aborts.

## Decision boundary

This is a post-pilot mechanism diagnosis on disclosed development data. Its statistics are descriptive and carry no mechanism promotion claim. The **32-root competence gate** remains closed.

A high fraction of singleton menus or a small number of truly discriminating forks motivates a new source generator with explicit, verifiable action-choice margins. Low train-only optimal-label accuracy would motivate an optimizer/representation preflight. Strong train accuracy with weak native proof completion would motivate an independently seeded generalization task and a better typed public-state serialization.

Any subsequent CPU training-size change must be prospectively registered on fresh train/calibration lineages. The original eight calibration roots and all their receipts remain unchanged. The locked \`ordinary_test\` and \`ood_family\` splits remain untouched.
