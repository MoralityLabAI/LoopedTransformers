# Nanbeige pass-2 K rescue

November 23, 2026 · CPU commissioning and development registration

The BF16 rescue harness is CPU-verified. The next native test asks whether restoring clean pass-2 K reduces the pass-2 donor-versus-recipient margin advantage of an active pass-1 write over a physically matched sham. The RT-BPD-02 archive motivated this test when the best K reconstruction lowered mean matched-dose donor cosine from 0.900 to 0.805. [RT-BPD-02](https://github.com/MoralityLabAI/BPD/pull/146) · [Nanbeige synthesis](NANBEIGE_RESEARCH_SYNTHESIS_20261123.md)

## One writer, one reader, one primary outcome

At the last semantic token, the frozen BOUNDRESCUE109 active edit is the signed T2-minus-T1 write chord driven by the other world's layer-21 down-projection input contrast. The write lands in the BF16 raw residual at the end of pass 1. CENTERSHAM108 supplies a separate chord at the exact same rounded midpoint, with both endpoint radii and the chord norm matched within 3%. It also uses the established first-pass full-head and excluded-row matching gates.

At the same token in pass 2, restore the clean layer-0 K projection before RoPE in both the active and sham endpoints. The native hook checks that the edited residual, pre-reader state, raw Q, raw V, K before replacement, and the full first-pass head remain identical to the corresponding intact branch. The clean companion and freshly rerun clean heads also close.

The outcome is the pass-2 donor answer logit minus the recipient answer logit. For each world and sign, compare active and sham half-chords first with K intact and then with clean K restored. The signed difference between those two contrasts measures the donor advantage removed by restoration. Average over both worlds and signs within a source graph-pair root. The independent host scorer recomputes this directly from the recorded label logits using donor-minus-recipient orientation. It cross-checks against the original host scorer's recipient-minus-donor convention.

## Verified on CPU

| Check | Reproduction |
| --- | --- |
| Rounded active/sham planner | Four exact BF16 candidate pairs on one previously exposed archive root, covering both worlds and signs |
| Staged BF16 fixture | 32 edited and rescued full-head records and 224 audited physical vectors matched serial execution exactly |
| Scoped clean-K restoration | 16 exact restorations, with stale plans refused before staging and grouped interruption recovery exercised |
| Independent host scorer | Recovered −0.125 logits of sham-adjusted removal in the deliberately recipient-favoring fixture. It refused synthetic evidence as pretrained and rejected tampered resources and stage counts |
| Preregistered rule | Seven deterministic tests cover donor orientation, sham-only movement, zero effect, positive and negative decisions, incomplete roots and altered head records |

The staged decoder and tokenizer-host verification ran in separate CPU processes under their original memory limits. The staged probe used randomized small BF16 weights. The host fixture recovered its manufactured effect with the registered sign. The planner replay used frozen archive data and a pinned CPU head. The verification made zero pretrained decoder forwards and initialized no GPU.

## Registered development decision

The frozen BOUNDRESCUE109 roster has 16 fresh graph-pair roots, 32 focused prompts and 96 prompts including controls. The pre-existing source and tokenizer audit excluded 1,300 previously exposed prompts and 640 previously exposed labeled graphs. The closed source payloads matched their locked digests. The registered native attempt directories were absent at design time. Before any native readout, the commissioning receipt must establish first-use custody for these roots.

Advance to larger sham calibration only if the native pilot meets all of these conditions.

- Both worlds pass the parent competence gate on at least 12 focused roots and all 16 literal-copy roots. At least 12 roots also close the physical and head checks and have positive signed removal.
- Across the full 16-root panel, mean intact donor advantage and mean removal are both at least 0.125 logits. A refused root has no measured outcome and contributes zero credit toward this engineering decision.
- Mean removal exceeds the observed maximum absolute sham-only K removal across the measured roots.

Every other complete native result is a no-go. Missing or altered terminal evidence invalidates the score. The verified EvalMath rule requires at least 59 independent matched-sham roots for a 95/95 sample-maximum bound. The 16-root panel can decide whether to pursue that calibration. [EvalMath](https://github.com/MoralityLabAI/EvalMath/tree/accf4d9040bee8daeabc6b07003bb484708996c0/skills/evalmath-mechinterp-power)

The CPU work qualifies the instrument and the prospective decision rule. The native effect is unmeasured. The grouped 391-token BF16 device parity check and first use of the development roster belong to native commissioning. The ARG-03-N protected cohort remains sealed.

## Provenance

The experimental registration was committed locally at 5e513a8 before the new verification runs. Its SHA-256 is 6301b2bb423ac9dbcd8969d5a69252e657c47eaf78e3e5118231cff8f380277c. The CPU-verified implementation is on local branch research/nan-krescue114-cpu-20261123 at 63d6c56, based on 7c7fcb9. Its grouped runtime manifest hashes to 0948efca3dd87a2b8c30eaa955a827259641fd50137cbc75c358ed38f14af383. The replay guide and aggregate JSON receipts are included with the source patch. Raw prompts, token IDs, weights, native records and private root rows stay local.
