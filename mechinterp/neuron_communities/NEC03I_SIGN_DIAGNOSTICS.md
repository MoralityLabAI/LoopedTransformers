# NEC-03I: Where the TopK circuit effect changes sign

**2026-11-21. Research evidence class: retrospective diagnosis on NEC-03H's already-opened development worlds.** Every root, source assignment, checkpoint, SAE codebook, feature rank and registered primary/controls comes from the [locked NEC-03H-v2 run](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44878411694). This analysis trains no model and opens no additional world or reserved context.

## Reproducible evidence and provenance

- [Successful NEC-03I independent CPU audit](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44884385085); artifact **nec03i-eight-root-readonly-effect-direction-and-sham-audit**, ID **14699128006**, ZIP SHA256 **deb85c3aab287e83d95a02b7e34b48c12c4a427e090efa4beedec6be4c5cf960**.
- Uses the original eight **nec03h-v2-development-root-8** through **root-15** artifacts, downloaded from the *original* run ID **44878411694**. The source registry, v2 protocol lock, all 16 native checkpoints, available source-only trained TopK16 dictionaries, and original independent-checker receipts are SHA256 verified.
- Recalculates every 16-bit truth function, full 48-context × 16 focal cases, clean/hubs/left/right logits, source-ranked *active* code selection, direct MLP-down output-norm attenuation, signed error difference, dose gate and intention-to-treat effect. All additive and native-sham comparisons are retained.
- The CI deliberately prepends invalid bytes to the original dev-013 raw transcript. Its independent auditor **rejects** the corruption, restores the original bytes and validates the root again.
- Source code: [pure-stdlib forensic auditor](nec03i_trace_audit.py); [adversarial unit tests](test_nec03i_trace_audit.py). The GitHub evidence ZIP contains both complete JSON diagnostics and the compact signed-world table.

## Distinct teacher-world effects

These eight roots have different four-variable Boolean truth tables and different four-focal-position quartets. Each one remains the **statistical unit**; its 48 contexts contain related counterfactual histories.

| Fresh development root | Native source site | Exact raw TopK minus lower-ranked active TopK correctness /768 | Registered ITT effect /768 | Matched physical signal gate | Primary context signs (+/−/0) |
|---|---|---:|---:|---|---|
| dev-008 | Graph | +22 | +22 | Pass | 23 / 3 / 22 |
| dev-009 | Graph | +26 | +26 | Pass | 19 / 1 / 28 |
| dev-010 | Fallback | −1 | 0 | Pass | 0 / 1 / 47 |
| dev-011 | Graph | +73 | 0 | **Fail** | 39 / 2 / 7 |
| dev-012 | Graph | +1 | 0 | **Fail** | 5 / 4 / 39 |
| dev-013 | Fallback | **−25** | **−25** | Pass | 3 / 21 / 24 |
| dev-014 | Graph | +3 | 0 | **Fail** | 3 / 0 / 45 |
| dev-015 | Graph | +3 | +3 | Pass | 3 / 0 / 45 |

**dev-010** was assigned ITT zero because the composed native head scored 526/768, below its registered 90% clean-competence gate, even though the local donor dose gate passed. **dev-011** had a raw apparent gain of 73/768, with 97 top-only and 24 alternative-only correct predictions, but the median retained donor dose was **22.62%**, below its **25%** hard floor. The positive raw result therefore remains a **failed comparison** under the original registration. The separate dose-failure reasons for dev-012 and dev-014 likewise remain intact.

The registered ITT vector is **(+22, +26, 0, 0, 0, −25, 0, +3)** over 768 predictions per world. Its world-mean signed effect remains **+0.4232 percentage points**, with **4/8 primary-admissible roots** and only **2/8** whose full null/sham family was independently feasible. The precommitted 2pp scientific minimum is unmet.

## The dev-013 negative sign

- The dev-013 teacher is **balanced AND/XNOR/OR** with literal inversion mask 5, truth table **0x4ff4**, focal positions **[0, 4, 5, 7]**. The composed model was perfectly clean **768/768**.
- Its source graph lacked a qualified multi-visit community, so the already-frozen source-only fallback selected memory-token GELU nodes **5:11, 5:25, 7:58, 7:59**. Their native weights/SAE source rankings were frozen before scoring.
- At equal, physically adequate local output norm, **30** focal cases were corrected only by the alternative active TopK donor, versus **5** corrected only by source-top-ranked active features, yielding **−25** net correct. The negative root contrast appeared in **21/48** source contexts, with positive contrast in 3 and zero contrast in 24.
- Both true-label classes contributed to the negative sign: **−15 cases for class 0** and **−10 for class 1**. The largest negative focal-mask contributions were mask **1: −7**, mask **3: −8**, and mask **7: −7**, jointly accounting for −22 of the −25 net. Alternative mask and class effects are preserved in the raw audit.
- Every sample/site for the primary pair had **positive common physical dose**. The median original TopK donor dose retained after pairwise attenuation was **46.65%**. The negative result therefore holds in an intervention with appreciable physical strength and exact donor norm equality.
- On an independently source-frequency-matched native-versus-native-sham pair in this same model, the native candidate itself produced **193 errors versus 170** for the native sham. This is a second example that the source-selected candidate direction can be less helpful than an alternative at a matched site and dose. These are different donor pairs and their shared attainable doses differ.

This evidence rejects a simple explanation in which dev-013's negative sign comes from an absent or underpowered donor. It identifies a genuine **context- and direction-dependent** difference on this synthetic recurrent world. Because dev-013 is the only eligible fallback primary among these eight roots, its sign cannot be statistically attributed to fallback selection across worlds. The data also cannot establish that a particular focal input bit or Boolean operator caused the sign reversal; a new controlled intervention is needed.

## Positive graph-selected worlds and sign instability

**dev-008** (truth 0x9666, focal positions [0,1,4,10]) gained **22/768** with 37 top-only and 15 alternative-only corrected cases; its strongest single focal-mask contribution was mask **15: +17**. In **23/48** contexts the top-ranked donor won; in three it lost.

**dev-009** (truth 0x009f, focal positions [1,5,6,8]) gained **26/768**, with 35 top-only and nine alternative-only correct cases. All 26 net improved predictions came from class **0**, and focal masks **12** and **14** each contributed +9. Both worlds used graph-selected native communities and passed all primary/null physical gates. **dev-015** gained only +3/768 under otherwise adequate matched signal.

The root-mean effect is unstable under single-world deletion. Excluding **dev-009** leaves mean ITT effectively zero. Excluding the negative **dev-013** leaves approximately **+0.949pp**, still below the 2pp scientific floor. Excluding dev-008 leaves approximately +0.074pp. These are posthoc diagnostics of world heterogeneity, **not** alternative registered estimates.

## Limitations of matched physical dose and rank attribution

Direct MLP-down output **L2 norm equality** ensures the donor vectors have matched instantaneous residual-branch magnitude at the specified recurrent visit. It does not match their **direction**, effect on subsequent recurrent dynamics, gradient alignment, route frequency, or final model decision boundary. A label-blind source-ranked TopK policy may learn features that dominate common training contexts while interfering with a different truth function or donor site arrangement.

The TopK active router chooses input-conditioned SAE features from 16 positive codes out of 128 according to a rank fitted on the source-training contexts. The alternative uses lower-ranked **active** codes under the same visit-specific component count. The rank is frozen, yet the model's source-trained ranking remains part of the treatment. This benchmark tests that **whole routing policy**, rather than an isolated neuron identity or a claim that SAE features outperform native neurons generally.

Only two new development roots have all predeclared auxiliary sham controls feasible; the 95/95 sham-tail variation floor remains uncalibrated. The eight worlds share a finite Boolean grammar and the same recurrent model architecture. **EvalMath FSI-01: REDESIGN_OR_PILOT_ONLY.**

## Next test on opened worlds

A defensible NEC-03I-B experiment can reuse exactly these eight frozen native checkpoints and source-trained TopK16 dictionaries. Before running fresh intervention outputs, freeze a small family of **source-label-blind hash-scrambled active-feature ranks**. Each comparison should select the same number of active features per recurrent visit, exclude the original source top-ranked active set from its control, and attenuate both donors to their shared attainable direct MLP-down output dose. Record overlapping/hub constraints, all 16-state raw outputs, source contexts, controls' physical strength and both signed outcomes.

The **interpretation** of that extra experiment will remain posthoc development on already seen task worlds. A source-rank benefit that survives multiple active-feature shams would help justify a stronger null-control design; it would not itself calibrate independent-world sham variation or unlock any of the **64 sealed confirmation worlds**.

The remaining 48 development worlds from the original NEC-03G registry, if later run, would bring total v2 independent development worlds to only 56. The separate 59-independent-root sham-tail calibration target therefore requires expanding the generator under a new SHA256 source registry, after a further development-only source policy and power review.
