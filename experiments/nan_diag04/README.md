# NAN-DIAG-04: reciprocal components

Fresh prospective DEV diagnostic separating reciprocal coefficient computation from operator-label selection. The original 168-case ARG-03 test stays sealed, and no BPD transfer runs. Read [PROTOCOL.md](PROTOCOL.md) and [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) before using this condition.

Registration commit: 10936e4caac06115e4a53b600cbab658295cc246. Two CAL roots and eight DEV roots; 24 independent contexts/root, one pinned checkpoint, BF16 and NF4 conditions. Roots are the analysis units, not scalar answers or token prefixes. Global input deduplication means iid root sampling is not asserted. All source generation is outcome-blind; no sources are replaced after inference.

## Weight-free audit

From the repository root with Python 3.12 standard library only:

    python -S -m experiments.nan_diag04.audit --out independent-audit.json

This recomputes independent final-answer regex acceptance, finite-convolution reciprocal certificates, root correctness, 15 exact conditional sign-flip sensitivities and Holm accounting from archived raw model tapes. It does not import the model, Torch, EvalMath or the primary parser/grader. CPU contracts require Torch and NumPy:

    python -m unittest discover -s experiments/nan_diag04/tests -v

## Reproduction boundaries

Use the source and checkpoint revisions in SOURCE_MANIFEST.json; pretrained inference must verify the manifest, public prompt files and archived GPU engineering fixture before loading. Host labels never enter the model worker. BF16 uses staged decoder weights, while NF4 uses CUDA decoder weights; both retain CPU BF16 embeddings/head and use the full-vocabulary argmax at every new token.

The engineering fixture includes elapsed-time metadata, so a new fixture run cannot byte-match its registered receipt. Save any recheck to a separate path and retain the original sealed receipt at runs/nan_diag04/engineering/gpu_fixture.json. Likewise keep every original raw run, error and resume receipt; do not overwrite archived evidence with reruns.

## Evidence layout

- fixtures/: public prompts, private host gold and source-only draw/rejection receipt.
- receipts/bf16 and receipts/nf4: original raw root tapes, execution and placement manifests, append-only memory samples and final memory summaries.
- receipts/: primary grades, 15-contrast family, actual power-planner inputs/refusals, tokenization audit, independent audit and post-freeze presentation diagnostics.
- analyze.py: post-freeze tables and scientific plotting only; it is not part of the frozen inference/grading contract.

The integer grammar waits for a delimiter or EOS before accepting a bare integer, so the token 1 can grow into 17. Full reciprocal accuracy requires all four independently generated coefficients to pass a finite multiplication certificate. Any failed numeric/label/code copy control overrides bottleneck interpretation with readout_control_failure.

All statistics remain development diagnostics. Conditional signed-law sensitivities impose assumptions that are not established here; unconditional p-values are withheld. Eight roots and eight repeated-erasure shams cannot support confirmation or a 95/95 semantic-noise tail bound. No later best-arm selection changes the original competence gates.
