# LGA-10E phase R: archived 61067 precision attribution

**19 November 2026 · retrospective, exposed original seed 61067 · no new source.**

The original archived capture was retrieved from [GitHub Actions run 44363670991](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44363670991), verified against its original SHA-256 manifest, and re-audited using the unchanged LGA-10D NumPy auditor. The latter passes **64/64**. The new [LGA-10E CI run 44466898826](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44466898826) passes its **six synthetic analytic tests** and reconstructs all four curvature scalars, physical FP32 additions, lifted probes and existing resolution errors for **1,152 edit observations** (64 cases × two arms × nine edits).

## Exact diagnostic identity

For a physically applied edit `a` and representable FP32 HVP direction `p`, the original native HVP uses `H32(p)`. The curvature discrepancy has an exact signed decomposition:

```text
q_native - q64_actual
  = [a·H32(p) - p·H32(p)]          left-contraction displacement
  + [p·H32(p) - p·H64(p)]          same-representable-probe HVP arithmetic
  + [p·H64(p) - a·H64(a)]          probe substitution
```

Subtracting the donor scalar from each sham scalar yields an equally exact
sham-minus-donor identity. Components can cancel; removing one is only a
counterfactual arithmetic subtraction and is **not** a real corrected model
or a new engineering contract.

## Results on the independent-base source

| Measure | Native-state derivatives | Reference-state derivatives |
|---|---:|---:|
| Original promoted-ready cases | 64/64 | 9/64 |
| Original full native-FP32-ready cases | 1/64 | 1/64 |
| Native-FP32 sham-minus-donor curvature numerical gate only | 7/64 | 6/64 |
| Numerical gate with the HVP arithmetic contribution set to zero (proxy) | **64/64** | **64/64** |
| Original donor-only representable-probe failures | 21/64 | 21/64 |
| Fixed-donor case ceiling under that original requirement | 43/64 | 43/64 |

Across the 64 native-state cases, the **median of each case's largest absolute sham-minus-donor** signed contribution is:

| Component | Median case maximum | Largest case maximum |
|---|---:|---:|
| Left-contraction displacement | 8.44e-10 | 3.69e-9 |
| Same-probe FP32 HVP arithmetic | **2.83e-8** | **1.26e-7** |
| Representable-probe substitution | 1.69e-9 | 7.37e-9 |

The corresponding reference-geometry median maxima are **9.16e-10**, **3.05e-8**, and **1.83e-9**. These descriptive component magnitudes identify HVP arithmetic as the first mechanism to investigate under an FP32-only implementation. The component counts themselves do not establish achievability of its removal.

All eight original input bases remain in the denominator. The 64 goals are correlated within eight bases and one untrained model initialization, so the evidence class is `RETROSPECTIVE_EXPOSED_61067_NUMERICAL_DIAGNOSIS`.

## Reproduction and fixed evidence

- Prospective preparation protocol commit: `d270cc1b051983f759fd2b02c87ac11ae43a62b6`
- Original 61067 raw SHA-256: `20f264a7fe0044ca2b142a5d1b70ffb27b6915a914bb91f96c30ccb5de65246d`
- Original feasibility JSON SHA-256: `223d7a247e1a4e581839017e6f44b1aad11a4ae20004e5de9757616a17a9a44f`
- [LGA-10E archived diagnostic artifact](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44466898826) with `diagnosis.json` (SHA-256 `d99828478eae3734e789ae3e8eb8a5591c8a5e5412d0cb49eb9ec410864128c5`) and `edits.csv` (SHA-256 `25b1db39707b0f6a13b2a7322f1e8fe952c23a2a594bef1a9c1fd878de703038`).
- Zero new HVPs; zero new model initializations; no trained weights opened; protected science seeds 31059, 31064 and 31065 remain closed.

## Next implementation target

Implement and compare preregistered **native-FP32 numerical algorithms** that reduce the same-probe HVP discrepancy, with proposed edit selection using only FP32 data. Preserve a frozen LGA-10D baseline and an explicit donor-repair arm. For each FP32 candidate, measure the revised donor-only floor, physical changes to the source intervention, full native-FP32 engineering readiness and the unchanged promoted contract. Commit tests and a source-guarded implementation before opening excluded engineering seed 61068. Trained-model correctness and causal efficacy require a later independently sealed study.
