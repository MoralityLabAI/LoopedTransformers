# DEUTERIUM-NATIVE-01: recovery of first 731057 audit and registered negative result

**Source date: 19 November 2026.** The 44,001-parameter original LOOPSCOPE four-pass teacher weights, VPD calibration, and frozen first 731057 heldout native scorer were registered in [PROTOCOL.md](../../PROTOCOL.md) at commit `d808db9122a60131c395804d7780bb3adec80fa7`. The actual first score happened in [GitHub Actions run 44465490737](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44465490737) at merge commit `67b248fdbcc2ac1f73ed58fce3f171db3e0db8d0`. The source-lock job and all three native first-scoring jobs passed. **All three original independent audit jobs failed after scoring** at one Python `NameError` in the auditor's independent source-label validation (`len(row)` used a free name in the target-row shape check).

The first experiment is permanently source-exposed. The same 256 registered first-test strings were used for all old teacher roots, with each full 81×10 logits array saved for all 17 predeclared arms. Any new scoring of 731057 would be a reproduction. This recovery is **read-only**. The original failed audit code is preserved with SHA-256 `3cd934d079e9ed2af1765efb77f4783a6d9e6c48443c07989734744e0cad5c44` and the original scorer remains at SHA-256 `1d01c087b8b35d8363b8d4a1ccec9fd70980c332966d11fb631c1893a9b92450`.

## Historical immutable bytes and correction

The first original pre-test source lock has SHA-256 `4f39f9e9b33ac538aab8eeb2b977a243b8b3b6d83e7dd800677e3a1678daffd1`. Its complete source SHA register, first calibration edit receipts and first heldout exposure markers were uploaded before scored data existed. [replay_first_artifacts.py](replay_first_artifacts.py) requires that same source lock, the original frozen science code bytes, exact original score artifact SHA-256s, and the same original three 103/211/307 trained selected-edit SHA receipts from [Stage B first run 44463652662](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44463652662).

[audit_recovered.py](audit_recovered.py) is a copy of the previously frozen independent Python-stdlib auditor, with **exactly one expression changed**, to `len(y[i])` so the target row is bound during validation. Original scorer, labels, logits, matrices, scientific protocol and Stage C test-source code are unchanged. The corrected code runs without importing Torch, and is prohibited from generating source strings. The [recovery workflow](../../../../../.github/workflows/deuterium-native-stagec-readonly-recovery.yml) fetches only original GitHub Actions 44465490737 first-score artifacts and Stage B 44463652662 calibration artifacts. It makes a new first-run audit/aggregate artifact while retaining the initial failed workflow as diagnostic evidence.

## First-scoring effects, from original immutable native receipts

Primary sign convention is selected **random minus VPD** goal-token native negative log likelihood (nats). Higher positive favors VPD. Whole-source bootstraps use 256 independent 81-digit strings within each teacher. The three old teachers are the across-teacher units for the Student-t interval.

| Original teacher | Clean full-sequence accuracy | Native random − VPD goal NLL | 95% paired source interval |
|---:|---:|---:|---:|
| 103 | 98.7895% | +0.00004177 | [−0.00011149, +0.00019687] |
| 211 | 98.4182% | −0.00014725 | [−0.00026838, −0.00002166] |
| 307 | 98.7124% | +0.00002724 | [−0.00011581, +0.00018175] |

Across old roots, the mean effect is approximately **−0.00002608 nats**, 95% t interval **[−0.00028738, +0.00023521]**. Teacher 211 nominally favors the matched selected random edit, while the positive teacher 103/307 differences are too small for their source-bootstrap intervals to exclude zero. The predeclared positive rule requires a positive effect in **all three teachers**, an across-root t lower bound above zero, and collateral accuracy preservation. It is therefore **DEVELOPMENT_NO_REGISTERED_ADVANTAGE** once the recovered independent arithmetic receipts pass.

The earlier synthetic Deuterium head-only study was also `NO_REGISTERED_ADVANTAGE`; this new development result instead concerns the authentic four-pass recurrent block. The three original teacher roots have appeared in many prior LOOPSCOPE interventions, and all share the same scored 256-source cohort. There is no independent novel trained-model confirmation. These effects are at the scale of approximately 10^-4 nats.

## Further mechanism investigation

The original external energy-potential trajectory result and the stronger LOOPSCOPE donor-pair causal results suggest that physically targeted directions may carry more task-specific signal than VPD's rank-one weight factors. The registered label-supervised task-gradient baseline was descriptive and should be reported separately from target-blind components. Stage B's three selected VPD directions were favorable on their **own 731055 selection cohort**, so the change of sign on heldout makes selection overfitting a plausible explanation to test prospectively.

The original \`param_decomp\` component cache stores one input tensor per module path. A repeated weight-tied target module may overwrite that path at successive recurrent applications; validating the optimizer's per-pass mask semantics is a useful source-only next step. We should inspect and test full pass-specific activation capture before extrapolating this weight-factor result to larger looped models. Any follow-up requiring new test outcomes must use new registered teacher roots and fresh source cohorts.

## Reproduction and evidence boundaries

The code and workflow can be executed repeatedly as an **audit of already exposed bytes**. It must not be described as a new prospective test. The first failed auditor is never overwritten; the recovery artifact has a different ID and explicit audit-correction provenance. The actual registered status appears in the independent aggregate \`registered_verdict.json\` only after all three first-run audits pass.
