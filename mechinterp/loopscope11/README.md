# LOOPSCOPE-11 / LGA-08: 75D causal pair fibers

This experiment follows [LOOPSCOPE-10C's independently audited whole-state donor transfer](../loopscope10/RESULTS_LGA07C_FIRST_NATIVE_20261118.md), with 3/3 **development** teacher seeds passing a natural-dose same-K random-sham gate. Its current-head-only arm behaved almost like the energy-tangent arm, and naturally reached predecessor/goal two-token patches recovered almost all of the full donor outcome.

[PROTOCOL.md](PROTOCOL.md) was prospectively committed at `538f01ad54d183c6adc42e07c794141990375248` **before implementation**. [POWER_PREFLIGHT.md](POWER_PREFLIGHT.md) freezes an EvalMath development-only gate and protected confirmation policy. The new source seed `31062` remains ungenerated until a complete Stage A pretrained-byte/tensor/protocol/executable source lock is verified and uploaded. Original `31060`/`31061` results and reserved `31059` are not reused.

## Instrument

- `pair_fiber.py`: 2-token × 48D physical source/donor projections, using the existing stable head-null SVD and the full frozen shared scalar energy gradient. Joint K typically has **75 dimensions**.
- `pair_fiber.plan_pair`: source-blind donor candidate and eight equal-L2 random K shams; then target-aware strict controls constructed to match **exact signed future donor-minus-recipient logit-margin derivative** through two actual tied blocks.
- `native_pair.py`: first source-locked 64 independent IID base strings × eight legal donors × three old teachers; true k2→k4 native trajectories, all 23 arms, full 10-class goal logits, all 81 predicted classes, finite E/head changes and complete f32 applied pair-edit/source gradient SHA receipts. The naturally reached whole-donor state is an explicitly non-dose-matched positive control.
- `audit_pair.py`: **Python stdlib only**, independently checks the original source roster, every physical edit/source vector, all signed future derivatives and matched sham margins, exact task scoring and 2,000 whole-source bootstrap decisions without running the models or selecting new cases.
- `.github/workflows/loopscope11-pair-fiber.yml`: source-only excluded `61062` CPU tests, safe on PRs.
- `.github/workflows/loopscope11-first-native.yml`: one-time original source-release job, gated by an explicit merge-commit title; Stage A freezes original model/observer/code hashes before Stage B first source 31062, then Stage C audits the original artifact.

Run the source-only tests:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s mechinterp/loopscope11/tests -p 'test_*.py' -v
```

The registered strong incremental development PASS requires two of three historically exposed teachers to exceed +10 percentage points over signed-future-margin-matched shams and +15 points over random same-K shams, with within-teacher base-source bootstrap lower bounds >0 and the full natural donor/pair positive controls competent. It is a stringent **development engineering** gate, not a root-level scientific conclusion. Formal confirmation would require new independent trained teacher roots and an EvalMath resolution calibration.
