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
