# FSI-01: EvalMath evidence gate for AF, LS-SAE, SMC, PEF and research gyms

The paired binary reporting adapter is `research_gym/scripts/evalmath_fsi_gate.py`. The source of the statistical math is [EvalMath FSI-01](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/finite_sample_inference.md), with the precise implementation pinned by the optional CI workflow. Keep raw training, calibration, selection, and **readout** stages distinct before scoring.

The generator/replay exports one JSONL record per observed task:

```json
{"case_id":"case-0","root_id":"root-0","candidate_correct":1,"control_correct":0}
```

The registration file, committed *before* running the scored models, declares the target population, `alpha`, minimum effect, planned comparisons, independence unit, and provenance flags. The adapter rejects missing keys and derives no independence or sealing claims from a filename. Its output hashes the registration and actual outcomes, reports the exact paired discordance counts, group-macro effect, cluster-aware inference, and decision. A 64-row evaluation with eight roots reports `n_cases=64, n_clusters=8`. Bootstrap intervals are descriptive and do not upgrade those eight roots to a high-confidence finding.

From a checkout where EvalMath is installed (`pip install -e path/to/EvalMath`) or with `PYTHONPATH=path/to/EvalMath/src`:

```bash
python research_gym/scripts/evalmath_fsi_gate.py \
  --registration research_gym/fixtures/fsi01_synthetic.registration.json \
  --cases research_gym/fixtures/fsi01_synthetic.cases.jsonl \
  --report /tmp/fsi01-report.json
```

Add `--require-confirmed` when the goal is to gate publication. It exits with code 2 for suggestive/inconclusive/refuted results. The included 16-item/4-root fixture is **constructed** and must never be cited as model evidence.

## Enrollment for existing experiments

- **AF / activation funnel:** provide the matched control's binary outcomes on the same graph, and cluster all perturbations of one source graph together. An epoch or checkpoint does not make the same evaluation graph an independent replicate.
- **LS-SAE / causal transfer:** score paired puzzle outcomes, cluster by original puzzle family/source, and record whether intervention doses or directions were selected using any of the scored puzzles. Quantitative margins (rather than binary flips) require a future extension of the inference module.
- **SMC / SAE controls:** cluster by originating seed/source/task family, and keep Gaussian-control and real-trained-feature hypotheses in the preplanned comparison family. Do not merge repeated SAE features into the case count.
- **PEF / loop extrapolation:** cluster by held-out context group, keep different loops of the same episode in the same cluster, and distinguish complete finite truth tables from sampled future contexts.

A compiler-certified or fully enumerated finite domain uses `evaluation_mode=exhaustive` with an audited-domain claim and its coverage audit SHA-256. This remains a statement about the audited domain. Reusing previous evaluation cases during architecture search calls for a new frozen, sealed confirmation set.
