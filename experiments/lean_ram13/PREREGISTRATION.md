# RAM-13 preregistration | genuinely new MathCan source-family proof search

**Frozen 2026-11-15 before reading any selected TEST theorem declaration, proof body, compiler goal, action outcome, or model result.** This is the first prospective test after the certified historical RAM-11F/RAM-12 training curriculum. It does not alter or count towards the separately frozen RAM-08 PEF and RAM-10 test studies.

## Source and cohort, registered using Git tree paths and blob metadata only

Exact new source commit: `patrickdugan/MathCan@f427e0ef4a61093304d12ca2c0dc943d8d2429e5`, Lean4.35.0. The following **six whole new top-level modules** are absent from the old RAM-08 TRAIN source `c84e2025334e9a967c1297425a490f98d5b3d189`, and are outside RAM-10's eight TRAIN, two DEV and four TEST modules:

| Registered TEST module | Original git blob SHA |
|---|---|
| `MathCan/ChromaticObservers.lean` | `61917ac9f1890cc90055b45defd9f1f78ef4d9a3` |
| `MathCan/DeferredObservability.lean` | `e1bf593dac5629f39e5b03d8e60012874ad9ff4d` |
| `MathCan/GramianTransfer.lean` | `f07e99a3f27acba22c2c0c6b3f675a68ab774ab2` |
| `MathCan/ObserverPermutationNull.lean` | `dad53a0e1a05d4b780d5da8d0eabff1db0f2669` |
| `MathCan/PhaseMemory.lean` | `615f20a32b0e190408ebee0b3b2a4f9bb695ee02` |
| `MathCan/SpectralStability.lean` | `a140af4fb5f89b753d8c1402803eb9cfe6e05f47` |

Select the **first two top-level `theorem` or `lemma` declarations** in original source order from every registered module, or all if fewer than two. Empty module is a registered exclusion with an explicit zero-case receipt, never replaced. Preserve the exact original source, imported dependencies, declaration statement, source order, and raw SHA in a locked TEST manifest before scoring. Do not use gold proof prefixes, theorem names or imports for model/policy ranking.

This is a **procedural fresh-source holdout**. Six public module names were chosen based on path/blob metadata and thematic diversity; they are not cryptographically hidden. Imported mathematical definitions may overlap old work, so avoid claims of fully disjoint underlying mathematical vocabulary.

## Frozen original neural policies

No new training or checkpoint selection from TEST. Compare the three originally frozen RAM-08 policies with their three fixed RAM-12 residual-head variants. Original RAM-08 model weights (train run 43689874169):
- `loop32` / seed29: `4a88bc80456743a89a70882911060cc90e76785a5ae1d435a0e46a8f8c09778f`;
- `loop64` / seed43: `5cbd327904714ac698897e1c2333cae9ec4125aeb70e7c56cd6236db8a737dac`;
- `flat64` / seed29: `2aa101edbd56f6563ebb2edcfc8c61c07dcc027559ea4cdc551187d6ec5303d8`.

Original RAM-12 registered **first**, not posthoc, residual-head artifacts:
- `loop32_residual`: `c288f38a2d912a8a544ef057257aacde1f93da202fe42679884a2e79cfa259cd`;
- `loop64_residual`: `b281d349d56de154025cc567c80ad920ae99e24a50796c9dd182ff1d47b7b92a`;
- `flat64_residual`: `90d1ee19694116e432b755924208f1f493482b6838647fe33ae133fc89e61f62`.

The source RAM-12 registered first-run archive is `RAM12_certified_native_states_and_rankers_20261115.zip` (the checksum manifest and source archive must be independently validated). Every residual uses the frozen `r_0+0.35 tanh(w^T z_0+b)` policy. The strictly compiler-rejected-only RAM-12 **posthoc** checkpoints are excluded from this prospective evaluation. No step/seed selection after any TEST access.

## Matched controls and budgets

Fixed 20-family RAM-08 source-visible grammar, same action pool/candidate script spelling for every policy: native fixed order, original TRAIN-only global-frequency, original TRAIN-only shape-frequency (prior=8), three original neural rankers and three bounded-residual rankers. Source of baseline frequency counts is original RAM-08 TRAIN only, with all original RAM-08 development roots excluded.

Primary: each of the selected TEST theorem roots × 9 policies; **20 seconds wall**, at most **32 actual Lean candidate evaluations**, eight accepted tactic transitions, at most eight open goals, three-second per-call timeout, two Lean session restarts. Secondary: the same TEST roots × native, global-frequency, shape-frequency and `loop64_residual`; **40 seconds wall**, at most **64 evaluations**, twelve transitions; same axiom and process constraints. TEST is evaluated in the first registered source order, with policy order cyclically rotated by case index to mitigate predictable process startup bias.

Every accepted original theorem requires full original-source no-`sorry` Lean 4.35 compilation with only `propext`, `Classical.choice`, `Quot.sound` or no axioms; second byte-identical full-source compilation inside each per-theorem wall budget; third independent external compilation, hash audit and action-grammar whitelist after scoring, leaving original scores immutable. Resource ceilings: 2 CPU threads, 6GiB process-tree RSS, predeclared whole-run cap, first attempt preserved even if negative or interrupted.

**Primary criterion:** one original novel TEST theorem certified by `loop32_residual` or `loop64_residual` and missed by native, global-frequency and shape-frequency under identical primary budgets, with zero false certificates. Report whether both corresponding original unmodified recurrent policy and flat variants miss the same theorem. Pairwise improvements in first-action rankings, previously exposed TRAIN targets, and source-reference proofs earn **zero** full-theorem capability credit.

## Execution isolation and privacy

The exact test source may be checked out only in an independently labeled prospective EVAL stage **after** the above original policy SHA files and comparator definitions are frozen and verified. Use the functioning public MathCan CI for a source/toolchain build or public-only theorem oracle, while preserving private LoopedTransformers checkpoint bytes and training dataset in private/local execution. **Do not publish private checkpoint weights or original private training examples in MathCan.** If a local private-model executor cannot run Lean, publish an explicit infrastructure blocker and the prepared test harness. A public-only symbolic result must not be compared with unexecuted neural policies as if it were a complete matched study.

All outcomes, including zero-score and negative outcomes, must remain in source-controlled receipts. No merge or change of the original RAM-08 PEF and RAM-10 test registrations.
