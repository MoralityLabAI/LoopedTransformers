# PEF-08 | Local frozen-checkpoint intervention results

Run date: 2026-11-15. Registered PEF-08 protocol committed first at `1dd57e6427c15cd512ffc113b1b986f2df78b6a7`. Parent PEF-07 head: `d9c2087cbe37f4787e3e56ef9d64ab2eb148f214`. Python 3.13.5, PyTorch 2.10.0+cpu, CPU-only, `DNNL_MAX_CPU_ISA=AVX2`. This is **local evidence**, not a successful Python 3.12 hosted CI run.

## Input integrity and controls

Loaded the six existing 450-update checkpoints from the PEF-07 evidence ZIP, verified each checkpoint byte SHA256 and exact parameter SHA256 from its sealed ledger, verified the archive's source manifest hash `849d039e47dec472b0b4835d24928e9a547e262bf84fd7dfed16741df9911ac1`, and reproduced **768/768 h12 native and 768/768 untouched h14 fitted-ridge** for every coalition and additive-threshold seed. All original checkpoint bytes and parameter hashes remained unchanged following intervention scoring.

For each pair and visit, eight channels were selected exclusively on the 48 probe contexts using the absolute hidden four-corner signal weighted by the **already trained native-head margin coefficients**. They were compared with eight deterministic random 8-channel masks plus a bottom-8 mask, on the independent 48 heldout contexts. Every source group retains all 16 focal patterns. The same original h12 head and untouched, once-fitted h14 ridge readout were used after each intervention.

## Aggregate loop trajectory

At the zero-focal anchor, mean absolute *native-head* margin D_ij over the three coalition seeds and their intended pairs versus the four crosspairs:

| Visit | Coalition intended | Coalition cross | Ratio of pooled means | Threshold intended/cross ratio |
|---:|---:|---:|---:|---:|
| 4 | 2.3999 | 0.0655 | 36.66 | 0.97 |
| 6 | 7.7527 | 0.1511 | 51.30 | 0.95 |
| 8 | 14.0128 | 0.1798 | 77.95 | 0.98 |
| 10 | 20.1090 | 0.1982 | 101.44 | 0.98 |
| 12 | 26.0074 | 0.2142 | 121.43 | 0.99 |

These descriptive ratios use the ratio of pooled means, whereas PEF-07 reported a per-seed ratio of separate means. The fixed h12 native head is being applied to intermediate states, so intermediate visits measure a diagnostic projected interaction rather than separately trained intermediate accuracy. h14 uses a separately fitted ridge scale and is excluded from this cross-visit table.

## Pair erasure and classification

Aggregate means over three seeds and both intended pairs:

| Task | Patch visit | Top-8 relative margin attenuation | Random-8 mean attenuation | Mean patched h12 correct /768 |
|---|---:|---:|---:|---:|
| Coalition | 4 | 0.4400 | 0.1724 | 715.2 |
| Coalition | 8 | 0.6745 | 0.2339 | 642.5 |
| Coalition | 12 | 0.7765 | 0.2579 | 624.0 |
| Threshold | 4 | 0.0782 | 0.0331 | 768.0 |
| Threshold | 8 | 0.4787 | 0.2109 | 756.2 |
| Threshold | 12 | 0.6403 | 0.2337 | 604.0 |

At visit 12, the preregistered descriptive top-minus-mean-random attenuation contrast was positive for **both pairs in all three coalition seeds** (3/3). For seed 103 the contrasts were 0.5107 (01) and 0.4605 (23); seed 131: 0.5998 and 0.4613; seed 167: 0.5320 and 0.5475. The pair-specific effects therefore transfer over all the 48 previously unseen heldout context groups.

Every trained coalition model lost **exactly three focal masks per heldout context** when the selected eight-channel visit-12 interaction term for one pair was subtracted. For pair 01 the wrong-mask integers were 3, 7 and 11; for pair 23 they were 12, 13 and 14. Each seed scored **624/768** h12 native accuracy after either pair-specific edit. An exact source-only check confirms these masks are the three positive cases attributable to the removed pair with the other AND pair inactive.

The **fixed, already fitted h14 ridge** obtained after two natural continuation visits (0,1) still scored 768/768 on five of the six coalition pair edits. Only seed 131 with pair 01 scored 721/768. All six threshold pair edits retained 768/768 at h14. This suggests functional information survives in the frozen recurrent state or is accessible through continued processing, even when the original h12 decision breaks.

## Interpretation limits

The edit directly subtracts the observed four-corner hidden difference, gated on both bits being one. Vanishing selected-channel hidden D_ij at the patch location is therefore an algebraic property of the intervention. Similarly, the zero-anchor off-target pair effects can remain unchanged because the gating excludes their source patterns. These facts cannot count as independent proof of causal specificity.

Top-8 channel choice also uses native-head weights, so a reduction in its own h12 margin is partly expected by construction. The stronger observations are the exact class-error pattern on heldout contexts, the effect of patching earlier visits and propagating through later nonlinear steps, the full threshold-control comparison, and the preservation of most h14 fixed-readout performance. The experiment does **not** uniquely identify natural computational features or show transfer to unseen Boolean operators or Lean reasoning.

Five local pytest invariants passed, including source grouping, fixed readout equivalence and no-op replay. Full raw loop traces and every random-mask comparator were saved as `summary.json` in the local evidence bundle. PEF-07's GitHub Actions Python 3.12 failure occurred before runner execution, so this result remains local.

## Reproduction

Extract the original PEF-07 local archive, then from the Loop​edTransformers repository root:

```bash
python -m pytest -q research/pef08/test_pef08.py
DNNL_MAX_CPU_ISA=AVX2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m research.pef08.pef08 --evidence /path/to/registered --out /tmp/pef08
```

The full PEF-07 checkpoint archive is external to the PR. The PEF-08 code refuses mismatched archive manifests, sources, parameter digests, and checkpoint metadata. The standalone local evidence bundle contains full results and the historical PEF-07 model artifacts.
