# LOOPSCOPE-07 / LGA-04: Matched Sobolev terminal-gradient critics

A small weight-tied 44,001-parameter four-pass TRM can make useful activation updates without aligning those updates with the final task-loss derivative. The completed LGA-01, LGA-02 and LGA-03 experiments isolated this distinction.

LGA-04 tests an **external** horizon-conditioned value critic with supervised gradient matching against a same-start, equal-budget **value-only** critic. Teacher weights and original six train-only SAEs remain frozen. Source-only terminal-gradient calibration uses previously unobserved cyclic-copy sequences seeds **31052/31053**. The first fully new 256-sequence heldout test is seed **31054**.

- [Prospective protocol](PROTOCOL.md)
- [Full source-locked scientific results and limitations](RESULTS.md)
- [Calibration-only source graduate / equal-budget critic trainer](train_sobolev.py)
- [New terminal-gradient and native matched-dose causal evaluator](eval_sobolev.py)
- [Independent Python standard-library receipt auditor](audit_receipt.py)
- [Seven independent CPU tests](tests/test_sobolev.py)

**Original registered outcome: POSITIVE, 3/3 frozen TRM seeds**, 10/12 teacher-by-depth paired cosine/MSE cells passing. Absolute true terminal-gradient cosine remains very small, and the current-head teacher-forced gradient aligns better with terminal loss than the trained external critic. Native edits descending the gradient-matched critic beat the equal-budget value-only critic in 5/6 nominal 95% whole-sequence paired comparisons, while all six against random remained inconclusive.

Original fully source-locked [GitHub Actions #44232573535](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44232573535) passed sequential critic calibration, first unseen test and independent stdlib audit. The raw original matched critic bundle SHA256 is a9f553058f193421861b0d9d464001639bf0ab7be1fc45166820fb0e02486858; test JSON SHA256 bde2be845ca52c400ac0e753add7e0cb99ca10663c9dd5dd2139cca4e3280ad1. The original source/test artifacts are available at that run.

## Reproduction

From repository root using Python 3.11 and pinned PyTorch 2.7.1 CPU, download original source teacher+SAE bundle from LGA-02 Actions #44222863870 and original frozen LGA-03 critics from Actions #44228381515. Their source bytes are checked by the entrypoints. Reproduce only as an engineering replay of the already public seed31054 test, without claiming a second prospective result.

    python -m unittest discover -s mechinterp/loopscope07/tests -p test_sobolev.py -v
    python -m mechinterp.loopscope07.train_sobolev --source-bundle runs/lga04/original_teachers/bundle.pt --source-meta runs/lga04/original_teachers/train.json --prior-critic runs/lga04/original_value/value.pt --prior-meta runs/lga04/original_value/calibration.json --out-bundle runs/lga04/paired/pair.pt --out-meta runs/lga04/paired/calibration.json
    python -m mechinterp.loopscope07.eval_sobolev --source-bundle runs/lga04/original_teachers/bundle.pt --source-meta runs/lga04/original_teachers/train.json --prior-bundle runs/lga04/original_value/value.pt --prior-meta runs/lga04/original_value/calibration.json --paired-bundle runs/lga04/paired/pair.pt --paired-meta runs/lga04/paired/calibration.json --out runs/lga04/new_test/full.json
    python -m mechinterp.loopscope07.audit_receipt --source-bundle runs/lga04/original_teachers/bundle.pt --source-meta runs/lga04/original_teachers/train.json --prior-bundle runs/lga04/original_value/value.pt --prior-meta runs/lga04/original_value/calibration.json --paired-bundle runs/lga04/paired/pair.pt --paired-meta runs/lga04/paired/calibration.json --test-receipt runs/lga04/new_test/full.json --out runs/lga04/audit/audit.json

The original calibration critic can improve on a matched value-only control without proving that the TRM's own tied recurrent update implements an optimizer. The next mechanism-level test should identify a potential function predicting the *native state transition*, rather than just externally approximating terminal-loss sensitivity.
