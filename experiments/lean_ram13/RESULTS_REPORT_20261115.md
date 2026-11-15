# RAM-13 | Fresh MathCan theorem-family transfer under native Lean 4.35

**Date:** 15 November 2026  
**Scientific status:** Completed first prospective source-family test; **registered recurrent-only criterion not met**.  
**Repositories:** `MoralityLabAI/LoopedTransformers`, pinned public MathCan `f427e0ef4a61093304d12ca2c0dc943d8d2429e5`.

## Executive result

The pre-scored cohort contains the first two original Lean theorem declarations from each of six never-before-used RAM source modules: ChromaticObservers, DeferredObservability, GramianTransfer, ObserverPermutationNull, PhaseMemory, and SpectralStability. All six public Git blob SHAs, twelve theorem names, three original RAM08 checkpoint hashes and three RAM12 registered first-run residual-head hashes were locked before proof scoring.

Each theorem was searched by nine policies under 20 seconds / 32 Lean tactic requests / eight transitions, and by four policies under 40 seconds / 64 tactic requests / twelve transitions. The original 20-family candidate generator matched the **entire ordered pool** on all **714** independently replayed original RAM08 states, accounting for **20,930 historical candidate effects**. All policies received the same grammar and theorem source. Source reference proof scripts entered neither model input nor generated candidates.

| Policy | Primary (12 theorems) | Secondary (12 theorems) |
|---|---:|---:|
| Native ordering | **10** | **10** |
| TRAIN-only global frequency | **10** | **10** |
| TRAIN-only shape-conditioned frequency | 6 | 6 |
| Looped32 RAM08 | 6 | n/a |
| Looped64 RAM08 | **10** | n/a |
| Flat64 RAM08 | **10** | n/a |
| Looped32 + RAM12 registered residual | 6 | n/a |
| Looped64 + RAM12 registered residual | **10** | **10** |
| Flat64 + RAM12 registered residual | **10** | n/a |

Across all **156** policy-theorem cases, **114** were accepted by two independent complete original-source Lean compiles within their individual wall deadline. A third, separate **unscored native Lean 4.35 compile** independently passed for all **114**, leaving **342 complete theorem-source compiles** on accepted runs, byte-identical source copies, permitted transitive axioms, and **zero false certificates** observed.

**Registered result:** no theorem certified by a RAM12 residual recurrent policy while missed by all three non-neural controls. No additional theorem coverage from training the RAM12 heads. Original Looped64 matched the residual Looped64 and the original Flat64 at 10/12. Two GramianTransfer theorems had no certified proof under any registered primary or secondary policy. Failing to solve under a finite action/time budget is not a mathematical unprovability result.

## New-source theorem matrix

| Family | Selected theorem(s) | Native | Loop64 | Loop64+RAM12 | Flat64 | Shape prior8 |
|---|---|---:|---:|---:|---:|---:|
| ChromaticObservers | `robustSixCode_in_range`, `robustSixCode_margin_three` | 2/2 | 2/2 | 2/2 | 2/2 | 0/2 |
| DeferredObservability | `present_blind_to_second`, `next_observation_is_hidden` | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| GramianTransfer | `transport_iterate`, `observable_transport` | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| ObserverPermutationNull | `native_population_certificate`, `native_histogram_mass_certificate` | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| PhaseMemory | `toggle_zero_to_one`, `toggle_one_to_zero` | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| SpectralStability | `sixth_roots_distinct`, `six_phase_character_orthogonality` | 2/2 | 2/2 | 2/2 | 2/2 | 0/2 |

The primary audit identifies two RAM08 Looped32 variants with a request timeout on the first SpectralStability theorem. These are preserved as budget-limited unknown continuations rather than negative compiler evidence.

## Provenance and release boundary

- Frozen preregistration: RAM13 `PREREGISTRATION.md`, initial commit `5935f96ea4fe0dc06ef1a115a1e4a55491b00d98` before access to the selected theorem source.
- Locked cohort: `LOCKED_MANIFEST.json` at initial commit `d7324f48c179e6a6c3968e2c01b72181661cba36`, then one disclosed pre-score integrity correction in `AMENDMENT_A.md`: the published 40-digit Git blob of ObserverPermutationNull was accidentally transcribed with 39 digits. No scored theorem case preceded this correction.
- Public Lean runtime: MathCan Actions run `43721176783`, original source commit `f427e0ef4a61093304d12ca2c0dc943d8d2429e5`, compiler version 4.35.0, all SHA-256 portable runtime parts verified before local extraction. The build used only public MathCan code.
- Private original RAM08 frozen model train archive SHA256 `a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc` and compiler-effect SHA256 `f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f`.
- RAM12 residual heads come from the **registered first-run CPU experiment**, SHA256 loop32 `c288f38a2d912a8a544ef057257aacde1f93da202fe42679884a2e79cfa259cd`, loop64 `b281d349d56de154025cc567c80ad920ae99e24a50796c9dd182ff1d47b7b92a`, flat64 `90d1ee19694116e432b755924208f1f493482b6838647fe33ae133fc89e61f62`. The earlier posthoc strict-rejection sensitivity heads were excluded from this test.
- No original RAM08 PEF theorem case or RAM10 first registered prospective theorem was opened or scored as part of RAM13.
- Private original model checkpoint bytes and RAM08 compiler training corpus were **not** sent to the public MathCan repository. The public CI export branch supplies only the public MathCan source and Lean toolchain.

## Reproduction

See the sibling `grammar.py`, `harness.py`, `run_one.py`, `secondary.py`, `independent_audit.py` and `build_receipt.py` for the original registered policy evaluation. Those scripts assume a local pinned MathCan source, compiled Lean4.35 runtime, and original private RAM08/RAM12 checkpoint archives. The source control manifests include exact original SHA256 and Git blob pins. Run the primary and secondary comparisons against a new empty results path to avoid overwriting the original scored evidence; the audit is pure post-score recompile and cannot change the certificate coverage.

This is a **first procedural source-family transfer**. The twelve targets are from six public MathCan modules, two each, and share imported definitions and mathematical neighborhoods. The original checkpoint was trained on earlier MathCan source, while RAM12 residuals were fitted to historical TRAIN theorem roots. Novel target statements provide source-family separation, but are not mathematically independent samples.

Next test: protocol-freeze a different cohort and introduce a richer, kernel-verified multi-step training signal with truly source-disjoint examples. The current native ordering baseline is already at 10/12, so the two resistant GramianTransfer targets provide a sharper challenge than the eight easy targets. Preserve original test results and avoid selecting hyperparameters using these twelve theorem outcomes.
