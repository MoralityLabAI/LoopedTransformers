# ARG-03 pre-unseal development audit

Pure development analysis and prospective controls responding to Fable's independent review.

- data/dev96_compact.json contains 96 already published development rows, with per-depth correct/top1 flags and per-loop swap/random state-readout distances. It explicitly lacks 64-program ranked lists and gold final-op labels.
- data/gramian_compact.json contains 240 already evaluated LOOPSCOPE-04 calibration spectra (two checkpoints, 120 boards each), with exactly eight sampled tangent directions in a 501D kernel.
- audit.py computes paired depth overlap/phi, per-case response-per-displacement, and robust stratum median/eigen concentration.
- controls.py implements a NEW per-depth local random intervention that matches the structured change's full-state L2 radius per case while leaving scored positions untouched.
- prepare_final_r.py generates a **new, final-operator R development-only** source roster split by the underlying final-R program into 48 fit and 48 validation cases; host labels are separated from model inputs. It does not call the old ARG-03 combined development/test roster generator.
- calibration.py fits a 3x4 per-operator-site bias (12 stored parameters, 9 degrees of freedom after gauge-fixing). It reports source-program rank and top1 on new development fit/validation score tapes.
- tests/test_preunseal.py exercises the frozen development receipts, safeguards and pure known-answer controls.

Run from LoopedTransformers repository root:

    python -m unittest discover -s mechinterp/dark_fiber_gyms/arg03_preunseal/tests -p 'test_*.py' -v
    python -m mechinterp.dark_fiber_gyms.arg03_preunseal.audit --output /tmp/arg03_dev_preunseal.json

The original test.jsonl (168 cases) remains sealed. The new final-R split is a separate development experiment with a NEW seed and must not be conflated with the original dev/non-R versus test/all-R preregistration. Its overlap against sealed test cannot be certified without an independent blinded digest-only comparison; this has not occurred. No new model logits or GPU traces are claimed.

The frozen original test protocol remains unchanged. The dated amendment is at papers/shared_weights_dark_fibers/ARG03_FABLE_PREUNSEAL_AMENDMENT.md.
