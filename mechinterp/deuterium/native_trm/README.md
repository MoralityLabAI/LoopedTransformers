# DEUTERIUM-NATIVE-01: native recurrent TRM steering

This continuation is **separately preregistered** from the frozen Deuterium historical/tiny-gate follow-up. It targets an authentic 44,001-parameter, four-pass weight-tied TinyRecursiveModel from LOOPSCOPE-04/05, using the original pretrained LGA-02 teacher/SAE bundle and an unmodified native forward implementation.

The historical Deuterium result remains **NO_REGISTERED_ADVANTAGE**. The recovered 11M-parameter historical checkpoint remains **BLOCKED_ARCHITECTURE_PROVENANCE**. Neither result is changed by this development-only four-pass experiment. The three teachers (103, 211, 307) have been examined in previous LOOPSCOPE studies, so any future fresh-source effects will be **development findings**, pending independent newly trained teacher roots.

- [Immutable first registration](PROTOCOL.md) at commit `d808db9122a60131c395804d7780bb3adec80fa7`.
- [Source-only native preflight](preflight.py) checks original bytes, canonical teacher/SAE tensor fingerprints, old 31043 clean-task accuracy, direct/manual four-pass parity, hidden-state injection, actual rank-one recurrent-weight injection, exact restoration and dose.
- [Independent standard-library audit](audit_preflight.py) checks receipt arithmetic and source lineage without Torch or model execution.
- [Untrained smoke tests](tests/test_preflight.py) deliberately use excluded source 61073.
- Active CI workflow: [.github/workflows/deuterium-native-trm-preflight.yml](../../../.github/workflows/deuterium-native-trm-preflight.yml).

## Execution stages

**Stage A: source fidelity, currently implemented.** On GitHub Actions it fetches original run `44222863870` artifact `lga02-source-train-only-teacher-and-six-SAEs` and verifies these exact SHA-256 values:

- Bundle: `d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266`
- Train metadata: `7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d`

All source-only checks use the old and disclosed 31043 evaluation strings or a newly initialized untrained smoke model. Receipt artifacts preserve failure diagnostics. A green workflow is source/instrument engineering evidence only.

**Stage B: pending.** Fit VPD with the pinned original `param-decomp` implementation using original training source 31041, then open newly reserved calibration seed 731055 to select among 16 learned directions and 16 dose/rank/search-matched random outer products. Freeze selected matrices, source-only config and hashes before any new heldout evaluation.

**Stage C: pending.** Use a separately registered one-time release after source and selected-edit receipts pass to open reserved heldout seed 731057, calculate native four-pass goal/off-goal effects, and independently audit saved logits and confidence intervals.

### Source-only command

After downloading the original artifact into `runs/deuterium-native/original/`, with CPU PyTorch 2.7.1:

```bash
python -m unittest discover -s mechinterp/deuterium/native_trm/tests -p 'test_*.py' -v

python -m mechinterp.deuterium.native_trm.preflight \
  --teacher-bundle runs/deuterium-native/original/bundle.pt \
  --teacher-meta runs/deuterium-native/original/train.json \
  --out runs/deuterium-native/preflight/preflight.json

python -m mechinterp.deuterium.native_trm.audit_preflight \
  --preflight runs/deuterium-native/preflight/preflight.json \
  --teacher-bundle runs/deuterium-native/original/bundle.pt \
  --teacher-meta runs/deuterium-native/original/train.json \
  --out runs/deuterium-native/preflight/audit.json
```

Historical 31043 performance can verify model competence and internal native replay; exact independent archival per-token logits remain unavailable. A new source score from 731057 must never be described as a second independent result after the first exposure.

No fresh calibration/heldout study has run from this package yet.
