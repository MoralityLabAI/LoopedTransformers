# Lite-Braille Fresco gym

A CPU-first symbolic observation/action gym for proof and looped-transformer traces. It renders a **pass-by-pass luminous fresco** using an injective 144-glyph Braille-cell code, six frame families, and four intensity bins. The included synthetic reader benchmark is a wiring check, and native Lean certification remains outside this module.

[Formal spec: `LITE_BRAILLE_FRESCO.md`](LITE_BRAILLE_FRESCO.md) · [PPT/LSA-02 block-order protocol](../../docs/lsa02b_block_order_protocol.md)

## Run the deterministic engineering gym

```bash
cd research/lite_braille_fresco
python lbf_gym.py fixture --roots 40 --seed 20261123 --out runs/lbf-fixture
python lbf_gym.py encode --observations runs/lbf-fixture/observations.jsonl --out runs/lbf-encoded
python lbf_gym.py render --encoded runs/lbf-encoded/encoded.jsonl --out runs/lbf-demo.svg
python lbf_gym.py score --encoded runs/lbf-encoded/encoded.jsonl \
  --labels runs/lbf-fixture/labels.jsonl --out runs/lbf-score.json
python -m unittest discover -s tests -v
```

Every command writes to a fresh destination. Use a new directory or filename for repeated runs. `fixture` emits separate public observation and private label files. `score` requires both, excludes any roots with censored candidate outcomes, and writes a manifest-like JSON score report. Outputs are deterministically hashed.

## Inspect a frozen PPT checkpoint (optional CPU PyTorch)

```bash
cd research/lite_braille_fresco
python ppt_bridge.py \
  --checkpoint ../lean_cgrt_hrm400/runs/SOURCE_CHECKPOINT/checkpoint.pt \
  --episodes ../lean_cgrt_hrm400/runs/SOURCE_NATIVE_RUN/paired_episodes.jsonl \
  --depth 3 --max-episodes 4 --out runs/ppt-observations
python lbf_gym.py encode \
  --observations runs/ppt-observations/observations.jsonl --out runs/ppt-glyphs
python lbf_gym.py render --encoded runs/ppt-glyphs/encoded.jsonl \
  --out runs/ppt-fresco.svg
```

The source folders above are **placeholders** for files in the user's environment. This bridge only uses public, predecision episode packets and saved checkpoint weights. It measures Plan B **coda slot output** and candidate logits, not raw middle-block cross-pass gates. No labels or verifier claims are exported. The codec cannot score native claims until a separately audited, complete-candidate receipt bank is available.

## Interface

`lbf_gym.FrescoEnv(encoded, labels)` offers `observation()` and `step('continue' | 'select' | 'abstain', candidate_id=None)`. Agents see only the encoded current/past visits. The private labels are consumed inside `step`. The environment returns incremental reward and total accumulated utility, including any charges for looking at another recurrent pass.

Source measurements should identify their instrumentation clock, role index, and calibration rule. For a real architecture experiment, compare against raw-slot readouts and equal-compute controls, preregister source and test lineages, and keep native Lean proof validity independent of the visualization.

## Completed native studies

- **LBF-09..11 (Horn/tactic gym):** [LBF-09 multi-shortest](LBF_09_NATIVE_MULTISHORTEST_RESULTS.md) · [LBF-10 structural transport](LBF_10_NATIVE_STRUCTURAL_TRANSPORT_RESULTS.md) · [LBF-11 loss factorial](LBF_11_NATIVE_LOSS_FACTORIAL_RESULTS.md) · [LSA-02J Horn saturation](../../reports/lsa02j_branch_utility_cpu_20261124.md).
- **LBF-12..14 (Equational rewrite & bidirectional layer scheduling):** [LBF-12 results](LBF_12_REWRITE_FRESCO_RESULTS.md) · [LBF-13 budget utility](LBF_13_BUDGET_FRESCO_RESULTS.md) · [LBF-14 bidirectional layer utility](LBF_14_BIDIRECTIONAL_FRESCO_RESULTS.md) · [LBF-14 continuous tensor & frontier geometry mechanics](LBF_14_MECHANISTIC_AND_FRONTIER_ANALYSIS.md).
- **LBF-15 (Geometric meeting-potential fresco):** [Mathematical specification](LBF_15_MATHEMATICAL_SPECIFICATION.md) · [Registration](LBF_15_MEETING_POTENTIAL_REGISTRATION.md) · [Results & sham-inversion diagnosis](LBF_15_MEETING_POTENTIAL_RESULTS.md) (`+11.11 pp` vs symbolic `bidir`; falsified `sym_emb` alphabet non-equivariance and planar sibling blurring).
- **LBF-16 ($\mathfrak{S}_6$-equivariant horizon-separated meeting fresco):** [Specification & registration](LBF_16_EQUIVARIANT_FRESCO_REGISTRATION.md) · [Prospective results](LBF_16_EQUIVARIANT_FRESCO_RESULTS.md) (`95.83%` vs `58.33%` symbolic `bidir` and `83.33%` geometric `MM`; `-60.3%` and `-49.3%` penalized work; all four pre-declared competence, recurrence, and causal-memory screens passed with `1,469` standalone Lean 4.19.0 certificates; sealed `ordinary`/`ood` splits preserved unopened).
- **LBF-17 (Relational abstract interpretation & provenance-preserving equivariant frescoes):** [Mathematical theory & Galois hierarchy](LBF_17_RELATIONAL_ABSTRACT_INTERPRETATION.md) · [Standalone zero-axiom Lean 4.19.0 formalization](lean/LBF17AbstractInterp.lean) · [Registration](LBF_17_REGISTRATION.md) · [Prospective results](LBF_17_RESULTS.md) · [Archive manifest](results/lbf17/manifest.json) (`100.00%` calibration closure across all 3 seeds at `29.15` penalized work vs `62.50%` `bidir`, `77.08%` charged `exact_lookahead`, `81.25%` `graph_bridge`/`geom_mm`, and `97.92%` / `35.66` work frozen LBF-16; `97.94%` symbol false-positive reduction at `100%` recall, `0.9985` / `0.9899` glyph AUROC, 5-way factored causal attribution, and `2,584` standalone zero-axiom Lean 4.19.0 declarations; both LBF-16 and LBF-17 sealed `ordinary`/`ood` cohorts preserved unopened).

## Illustrated research notes

- [LBF-16 → LBF-17 intermediate illustrated report](LBF_16_17_INTERMEDIATE_ILLUSTRATED_REPORT.md): closure and budget graphics, recurrence interpretation, and the horizon-support counterexample.
