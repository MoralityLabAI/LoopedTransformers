# INK-ROUTE-01: causal test of Inkling 2 Mini's coordinate 2027

**Status: development-only, native NDIF execution pending (2026-11-22).**

This experiment targets the fixed three-pass Inkling 2 Mini, the checkpoint used in [Goodfire's analysis](https://www.goodfire.com/research/interpreting-inkling-2-mini/). It is separate from the variable-loop research checkpoint and from the synthetic positive control in research_gym/inkling_positive_control. Neither those results nor this numerical harness confirm a native routing mechanism.

## Native source contract

The [NNsight + NDIF guide](https://nnsight.net/blog/2026/11/20/opening-the-box-of-inkling-2-mini-using-nnsight-ndif/) exposes the pre-reader message bank and the four query/key/value/MoE messages. Goodfire's 1-indexed **layer 18** is Hugging Face layer 17, the end of recurrent block 3. Goodfire's 1-indexed **layer 13** is Hugging Face layer 12, the reader we target in pass 2 (message_input call 6) or pass 3 (call 18). This distinction is tested.

The native source has been inspected at huggingface/transformers revision e3a65e2c21abb9c25a03704f2f25852f9ad03a18. message_input inputs [0][0] and [0][1] carry the two preceding-pass blocks, with block 3 at [1]. The output carries four separate reader messages. Two-source RMS-normalized mixing lets us reconstruct the observed block-3 routing coefficient from the bank inputs and reader output, without direct access to the learned query vectors:

    y = (1 - beta3) RMSNorm(block2) + beta3 RMSNorm(block3)

The reconstruction residual for each Q/K/V/MoE reader must be below 1e-3 before any routing interpretation. Goodfire's reported +32.4 RMS-normalized coordinate and 46% squared norm were measured on their FineWeb sample. Our eight simple public prompts are separate development data.

## Native intervention and controls

At a selected single message_input call, modify the **last-token, previous-pass block-3 bank vector**. Restore the entire original bank-vector value at the immediately following reader call. Propagation through downstream hidden states is the causal outcome.

- **clean**: observe the bank, four messages, layer-17 coordinate across three passes, and final top tokens.
- **zero_2027**: remove coordinate 2027 before reader 12 on pass 2 or 3.
- **l2_sham**: shift a SHA-256-preselected other coordinate by the same raw amount, producing equal L2 displacement.
- **norm_preserving_zero**: zero target coordinate and renormalize other coordinates to preserve raw bank-vector L2.
- **swap**: swap target and sham coordinates, preserving raw bank-vector L2.
- **reader_rescue**: perform target ablation, then replace this reader's four output messages with its clean outputs, before restoring the original bank at the next call.

The rescue tests the downstream reader-message pathway. Report both the routed-message change and final-reference log probability. L2 matching is checked from actual saved tensors. BF16 rounding and inadvertent bank mutation are instrumentation failure gates.

## Running

From repository root, first run the offline checks in an environment with numpy and PyTorch:

    PYTHONPATH=. python -m unittest discover -s research_gym/inkling_routing_01 -p 'test_*.py' -v
    PYTHONPATH=. python -m research_gym.inkling_routing_01.ndif --limit 2

The second command only prints a deterministic development manifest. It does not use NDIF. To execute two native development roots, follow the NNsight [installation instructions](https://nnsight.net/blog/2026/11/20/opening-the-box-of-inkling-2-mini-using-nnsight-ndif/), pin nnsight 0.8.3 and the transformers revision above, and set NDIF_API_KEY obtained from https://login.ndif.us. Then:

    PYTHONPATH=. python -m research_gym.inkling_routing_01.ndif --run --pass-number 2 --limit 2 --out runs/ink_route_01_dev

The runner refuses native execution without that key. CI runs offline tests only. Increase --limit to 8 after the native hook and restoration audit works. The script saves a SHA-256 manifest, paired JSONL records, compressed raw vectors, and artifact hashes. Do not commit NDIF keys or traces of private text.

## FSI-01 evidence gate

Independent unit: source prompt root. Tokens, arm variants, repeated passes and reader channels are paired repeated measurements on the same root. The eight public prompts are development debugging roots, not a sealed confirmation set.

Primary hypothesis: coordinate-2027 removal decreases block-3 beta3 routing more than matched shams at the layer-13 Q/K/V/MoE readers, and the reader-output rescue restores downstream behavior. Set the minimum meaningful causal effect and sham-variation floor after independent development calibration; the exploratory 0.10 beta change is not a registered significance threshold. Retain all arms and unsuccessful traces.

Before any confirmation, use the [EvalMath mechanistic-resolution preflight](https://github.com/MoralityLabAI/EvalMath/tree/main/skills/evalmath-mechinterp-power), register comparison multiplicity, acquire independent roots and an untouched readout, and apply root-level [FSI-01](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/finite_sample_inference.md) inference. Twenty independent roots is the minimum positive gate, not a power calculation.

Local tests establish numerical identities and torch tensor-edit reversibility. Native inference, patched routing and task effects are **unmeasured** until the remote pilot executes successfully.
