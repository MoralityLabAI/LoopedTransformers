# LOOPSCOPE-09 / LGA-06: How far does learned recurrent energy descent transfer?

A frozen single scalar potential E_shared was trained in earlier LOOPSCOPE-08 to align with the native F(h)-h update of a 44,001-parameter, four-pass weight-tied TRM. LGA-06 uses the ORIGINAL frozen teacher/SAE/potential checkpoint bytes and tests E_shared under three source shifts and twelve actual recurrent passes, all on new independent matched examples.

- [Source-frozen prospective protocol](PROTOCOL.md)
- [Complete first-science negative results and interpretation](RESULTS.md)
- [Source byte lock and canonical model hash verifier](source_lock.py)
- [Four distribution shifts, twelve native passes, matched causal edits and curl](eval_shift.py)
- [Independent standard-library source/energy/native/curl auditor](audit_receipt.py)
- [Eight source-independent CPU tests](tests/test_shift_energy.py)

The registered outcome is **FAIL, 0/3 teacher seeds** for robust geometric **and** task-preserving transfer over all four distributions. All 12 teacher/distribution cells passed all four original-depth geometric checks. Only 0–3 of eight extra-depth checks passed per cell. Across three seeds, four-pass IID accuracy averaged 98.6%, dropping to 10.2% by twelve passes; twelve-pass task CE rose above 4.7 nats per token. The shared potential lost monotonicity during passes five through nine even on IID sources. Early OOD energy descent also failed to guarantee task skill: Block9 sources maintained many positive-energy-drop steps while 4-pass task accuracy averaged 56.6%.

The first source-locked [GitHub Actions run #44239311003](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44239311003) passed eight tests, source artifact identity checks, fresh test, and an independent Python standard-library audit. The original fresh full.json SHA256 is 364dce0d03f3186afae0e4606a6d7acc057f90eadd94dd70d7d7006da90fdcf6. It includes all per-source 12-loop energy/CE trajectories, 12,288 native source-goal cases and 98,304 physical arms, with original curl witnesses.

## Reproduce as an engineering replay

Requires Python 3.11 and PyTorch 2.7.1 CPU. Download the original LGA-02 first-run teacher bundle+manifest (Actions #44222863870) and the original LGA-05 potential bundle+manifest (Actions #44235767614) to the corresponding paths. These entrypoints lock all source/checkpoint hashes.

    python -m unittest discover -s mechinterp/loopscope09/tests -p test_shift_energy.py -v
    python -m mechinterp.loopscope09.source_lock --source-bundle runs/lga06/source/teacher/bundle.pt --source-meta runs/lga06/source/teacher/train.json --energy-bundle runs/lga06/source/potential/potentials.pt --energy-meta runs/lga06/source/potential/fit.json --out runs/lga06/source_lock/source_lock.json
    python -m mechinterp.loopscope09.eval_shift --source-bundle runs/lga06/source/teacher/bundle.pt --source-meta runs/lga06/source/teacher/train.json --energy-bundle runs/lga06/source/potential/potentials.pt --energy-meta runs/lga06/source/potential/fit.json --lock runs/lga06/source_lock/source_lock.json --out runs/lga06/test/full.json
    python -m mechinterp.loopscope09.audit_receipt --source-bundle runs/lga06/source/teacher/bundle.pt --source-meta runs/lga06/source/teacher/train.json --energy-bundle runs/lga06/source/potential/potentials.pt --energy-meta runs/lga06/source/potential/fit.json --lock runs/lga06/source_lock/source_lock.json --test-receipt runs/lga06/test/full.json --out runs/lga06/audit/audit.json

The standalone standard-library audit can verify the immutable original artifact without executing PyTorch. Re-running the fixed 31058 source test after reading the first result is an engineering reproduction and must not be presented as a second independent prospective result.

The learned potential is an **external observer** and the native F Jacobian has nonzero measured skew. Early energy decrease remains useful as a geometric diagnostic within its fitted horizon; stable task optimization across new structures or extra loops remains unproven by this model.
