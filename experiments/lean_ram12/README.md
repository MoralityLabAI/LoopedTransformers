# RAM-12: state-conditioned proof-verified tactic training

**Status:** New native Lean state extraction complete; retrospective CPU ranking experiment completed with a negative registered criterion. Source-only original PEF and RAM-10 heldout results remain unscored.

## Original native Lean evidence

[Public MathCan PR #80](https://github.com/patrickdugan/MathCan/pull/80) merged into `main` at `04e15e1`. The original **run [43716751350](https://github.com/patrickdugan/MathCan/actions/runs/43716751350)** successfully compiled all 76 historical MathCan original theorem scripts twice, with read-only instrumentation inserted immediately before every original tactic. Exact original source `MathCan@c84e2025`, pinned Lean4.35.0.

The original immutable ZIP archive (`ram12-pinned-native-lean-state-traces-43716751350`, artifact 14162833338) has SHA256 `327dbd1382aca1a1c3c790e434341d582aafb85e5ad1b4f9b70ca84f21cae924`. Its positive-labels JSONL has SHA256 `f744f779012c04d50a41798ee3805c91fbb8da922a174b09ccbde3310eb95e8f`.

- 152 original full-source accepted Lean compiles, same scripts and printed proof states on both runs
- 282 actual tactic-position observations across 76 complete scripts
- **230 unique source-certified state/action labels** from 176 exact-prefix goal contexts
- 22 original MathCan TRAIN theorem roots: 14 fit, 3 development, 5 historical internal holdback
- Entire compiler output/source log pairs, SHA-256 and axiom inventory independently checked by [the new audit](independent_native_audit.py)
- 138 state/action pairs exactly match RAM-08's independently verified original observations; **92 state/action positives** were new to that previous collector
- Of 137 TRAIN positives, 83 have recorded Lean-rejected alternatives at the *same exact theorem and prefix*, while 54 remain **positive-only**

A published complete theorem proof is a witness that a selected tactic can be extended to closure. Alternative actions without proof evidence stay censored.

## State-conditioned frozen-model pilot

The [protocol](PROTOCOL.md) was committed before first fitting, on a separate LoopedTransformers branch. It freezes 150 AdamW updates, 32 pairs/update, three exact original RAM-08 selected trunks, a separately trained bounded residual head, and an unchanged 14/3/5 theorem-root split. All 20 original candidate families are available, with negative class in the registered first run drawn **only** from the original RAM-08 `label=0` failure-or-unchanged outcomes at exactly the same original prefix. An auxiliary length preference compares two *successful* full-source suffixes where one is shorter.

Registered primary historical diagnostic: number of 19 original holdback contexts whose highest-scoring candidate is a verified positive action among only **known positives and original compiler failures or unchanged outcomes**.

| Historical frozen trunk | Baseline | With RAM-12 residual |
|---|---:|---:|
| Looped32 | 6/19 | 6/19 |
| Looped64 | 4/19 | **7/19** |
| Flat64 | 6/19 | 6/19 |

The descriptive target required two of three to improve, so **the target failed**. A follow-up audit found six compiler-accepted but unchanged `label=0` candidates among 1,652 unique negative records. These remain in the original first run. An explicitly **posthoc strict-rejection sensitivity** excludes the six and uses only original `after.accepted=false` negatives: the holdback top-1 scores remain 6→6, 4→7, 6→6; holdback pairwise positive-versus-rejected wins become loop32 594→624/688, loop64 492→530/688, flat64 531→549/688. See [the separate receipt](receipts/strict_rejected_posthoc_20261115.json). For Looped64, all pairwise verified-positive-over-failed comparisons on holdback improve from 492/688 (71.5%) to 531/688 (77.2%). Looped32 moves 594/688 to 624/688, and Flat64 from 531/688 to 548/688. These selected known-action diagnostics are not proof-search capability scores.

The original 85K/207K/182K-parameter trunks are byte-frozen; the three saved residual heads and complete metrics are retained in `receipts/cpu_first_20261115.json` and in the downloadable local output package. Five independent local software contract tests passed. Earlier RAM-08 training used these theorem roots, meaning this is a **retrospective diagnostic** and provides no independent novel theorem-family transfer conclusion.

## Reproduce

With the original RAM-08 training artifact `ram08-frozen-audited-train-43689874169.zip`, the public native RAM-12 artifact ZIP from run `43716751350`, and the existing committed original [RAM-11F certified positive prefix corpus](../lean_ram11f/curriculum/certified_positive_prefix_action_labels.jsonl):

```bash
python experiments/lean_ram12/independent_native_audit.py \
  --artifact /data/RAM12_native_state_traces_43716751350.zip \
  --worklist /src/MathCan/research/ram11f/locked_worklist.jsonl \
  --out /out/ram12_independent_audit.json

python experiments/lean_ram12/state_cpu_pilot.py \
  --ram08 /data/ram08-frozen-audited-train-43689874169.zip \
  --ram12 /data/RAM12_native_state_traces_43716751350.zip \
  --certified-prefixes experiments/lean_ram11f/curriculum/certified_positive_prefix_action_labels.jsonl \
  --out /out/ram12r_first
```

For the distinctly posthoc strict-compiler-rejection sensitivity, add `--strict-rejected-only` and choose another fresh `--out` directory. The original registered result must remain available as the default.\n\nUse new output paths. PyTorch 2.10.0+CPU was used for the original local replay; the original RAM-08 optimizer used PyTorch 2.6.0+CPU. This difference is recorded. Full historical status and negative findings remain immutable.

**Next native experiment:** collect compiler-tested *failure alternatives at the 92 newly reached exact-prefix states*, using the original grammar and the pinned old source, then evaluate complete proof search against the full native, global-frequency, shape-frequency and flat-model controls on a genuinely fresh theorem family. Retain the original RAM-08 PEF and RAM-10 protocols and never use their results for checkpoint selection.
