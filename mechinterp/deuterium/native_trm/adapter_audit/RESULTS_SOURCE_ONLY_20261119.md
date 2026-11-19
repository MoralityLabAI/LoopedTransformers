# DEUTERIUM-ADAPTER-01: stock recurrent VPD cache is last-call-only and reuses one mask across four passes

**19 November 2026. Source-only engineering result: `ONE_PATH_CACHE_LAST_PASS_AND_REUSED_CI_MASK`.** The [full trained-VPD checkpoint native instrument run #44468845945](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44468845945) completed every source and audit job successfully. Its original result artifact is **14503228548** (\`deuterium-native-adapter-cache-source-only-*\`). The separate [untrained diagnostic-only run #44468293775](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44468293775) also passed. All runs were limited to old original teacher-training strings (31041) or excluded untrained 61076. Neither experiment generated or rescored the now-exposed 731057 heldout cohort.

## Original source and learned CI lineage

Model: original frozen 44,001-parameter four-pass weight-tied \`TinyRecursiveModel(48,4)\`, seeds 103/211/307, SHA-locked original first LGA-02 teacher bundle \`d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266\`, metadata \`7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d\`. The **trained** component and CI networks were loaded byte for byte from [original Stage B first calibration #44463652662](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44463652662): each \`seed_NNN/vpd/model_64.pth\` SHA agrees with that seed's frozen original \`seed_result.json\`, and all serialized \`target_model.*\` tensors were required to equal the original teacher tensors. The full original checkpoint was loaded with strict \`ComponentModel.load_state_dict\`.

This matters because an untrained diagnostic CI network can show the *source-code semantics* but cannot quantify the mismatch in the **VPD network actually fitted** for the prior negative result. The trained-source run does both.

## Registered source-only observations

The stock pinned \`param_decomp\` \`ComponentModel.forward(cache_type='input')\` attaches a hook to \`block.chan_mix.2\`. A separate native forward-prehook observed **four distinct physical activation inputs** at the one weight-tied module, while the component cache held exactly **one tensor**, byte-equal to the **fourth/last original call's input**. \`calc_causal_importances\` computed exactly the importance mask from this last input. During actual masked four-pass native recurrence, \`LinearComponents.forward\` received the **identical selected mask tensor object on all four calls**.

Using each saved originally learned VPD CI network, compute its prospective per-pass importance mask for each *actual clean reached-state pre-weight activation*, holding the same fitted network and source fixed. The mean absolute difference from the stock final-call-derived mask was:

| Original teacher | First native application | Second | Third | Fourth/stock |
|---:|---:|---:|---:|---:|
| 103 | 0.27464578 | 0.18320383 | 0.08888695 | 0 |
| 211 | 0.26785272 | 0.18549871 | 0.09476270 | 0 |
| 307 | 0.29798546 | 0.20903246 | 0.10494941 | 0 |

Corresponding FP32 128D physical pre-weight input RMS differences against the final call were 103 **[0.532576, 0.395449, 0.206518, 0]**, 211 **[0.533027, 0.388617, 0.201937, 0]**, and 307 **[0.547779, 0.404447, 0.210591, 0]**. The original 8-sequence training minibatch's raw physical 4-pass input ledger requires 4× the tensor bytes of one cached input; the new source-only instrument records both actual byte counts.

These numbers are **descriptive**. Per-pass masks come from the same originally fitted VPD CI model evaluated at differently reached clean states, while the stock optimizer applied its **last-state mask** at each original recurrent pass. They are not four separately trained original CI networks. Differences do not imply that the pass-indexed masks themselves have higher task-level causal fidelity.

## Interpretation and next proper experiment

The original Stage C native VPD steering verdict is [independently recovered](../stagec/recovery/README.md) as **`DEVELOPMENT_NO_REGISTERED_ADVANTAGE`**: random minus VPD goal NLL mean −0.00002608 nats, across old teacher t interval [−0.00028738,+0.00023521]. The signed task-gradient control achieved lower original heldout goal NLL than VPD on all three frozen teachers, but it used direct target supervision during source training.

The source-only adapter result establishes a **concrete training-objective identification problem**: stock VPD maps the final-pass input to a mask reused at all tied recurrent depths, even when the same fitted network would map the earlier real inputs to substantially different masks. It provides an experimentally grounded candidate mediator for the old negative result; the old study never randomized mask-source semantics, so causal attribution would be unwarranted.

Follow-on requires a separately preregistered, source-frozen comparator with the **same 16 component count, rank-one recurrent target, original parameter reuse, training and search budgets, source-family controls, and across-model root uncertainty**. Cross-fit a pass-indexed importance adapter against the stock last-call adapter on new teacher roots and new test sources. Record the four selected masks, their actual four-pass reconstruction response, true native edit dose/rank, component-only vs residual-restored fidelity, goal/off-goal NLL and accuracy, and any compatibility failures. Previously examined teachers and the exposed 731057 test are development evidence only; they cannot serve as fresh confirmation.
