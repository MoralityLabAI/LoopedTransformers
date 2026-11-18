# LOOPSCOPE-05 / LGA-02

**Research question:** after a weight-tied 44K TRM learns a cyclic-copy task through four applications of its shared block, does its native activation update align with the negative gradient of the **final** four-pass task loss, and do trained SAE-active and head-null physical directions mediate native remaining-loop effects?

- [Prospective frozen protocol](PROTOCOL.md)
- [Full results and scientific limits](RESULTS.md)
- [Original train-only model/SAE fitter](fit_and_freeze.py)
- [Fresh terminal-gradient and native intervention runner](terminal_sae_test.py)
- [Independent standard-library source/causal auditor](audit_receipt.py)
- [Regression tests](tests/test_terminal_sae.py)

The study-level registered result was **FAIL, 1/3 original trained final-only seeds**. This does not change LOOPSCOPE-04's earlier 3/3 positive LOCAL head-gradient result. Its source checkpoint reconstruction is exact to original old-test printed accuracy. New test seed 31044, 256 independent 81-token strings; the model/SAE checkpoint is frozen before test generation.

Observed mean local cosines after supervised training were +.268,+.569,+.525,+.173 across four loops; mean future-loss cosines were +.012,+.004,−.014,−.039. Train-only SAE active decoder spans hold roughly 28% and 36% of the native update energy at loops 1/2, while about 80% and 76% of terminal gradient norm lies in the instantaneous output-head nullspace. Native equal-dose interventions are task-sensitive but their SAE-active-versus-complement or head-null-versus-random differences lack familywise robust causal evidence.

The original complete model+SAE and first test artifacts are pinned to [Actions #44222863870](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44222863870). An [independent audit #44223836675](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44223836675) replays exactly those original source bytes and all 2,000-draw whole-sequence confidence intervals without loading Torch.

To run the full experiment from the repository root with PyTorch CPU (the hosted source used torch==2.7.1):

    python -m unittest discover -s mechinterp/loopscope05/tests -p test_terminal_sae.py -v
    python -m mechinterp.loopscope05.fit_and_freeze --bundle runs/lga02/source/bundle.pt --meta runs/lga02/source/train.json
    python -m mechinterp.loopscope05.terminal_sae_test --bundle runs/lga02/source/bundle.pt --meta runs/lga02/source/train.json --out runs/lga02/test/full.json
    python -m mechinterp.loopscope05.audit_receipt --bundle runs/lga02/source/bundle.pt --meta runs/lga02/source/train.json --receipt runs/lga02/test/full.json --out runs/lga02/audit/audit.json

The source-only unit smoke case uses seed 51044, leaving the new heldout test seed 31044 for the post-freeze test stage. Running the complete campaign again on its public fixed test seeds is an engineering reproduction, not a new prospective confirmatory experiment.
