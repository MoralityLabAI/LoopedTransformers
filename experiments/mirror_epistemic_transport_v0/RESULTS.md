# MIRROR-01: initial training receipts (synthetic)

Run date: 2026-11-05. Environment: CPU, Python/PyTorch, shared-weight two-pass Transformer with 36,866 parameters. Commands follow README.md. GitHub Actions have not reported checks for the research branch; these are local results.

## Engineering verification

- Six pytest tests passed, including scope exclusion, active correction, mechanical witness soundness, neural gradients, and baseline evaluation.
- All five Python files on GitHub have blob hashes matching the locally tested files.
- Eight-step, 24-episode smoke receipt: [smoke_seed7.json](smoke_seed7.json).

## 300 optimization steps, 200 held-out synthetic worlds per seed

| Train/eval seed | Inconsistent worlds | MIRROR found | Paired found | Random found | Mean queries MIRROR | Mean queries paired |
|---|---:|---:|---:|---:|---:|---:|
| 7 / 100007 | 109 | 109 | 108 | 88 | 6.745 | 6.800 |
| 11 / 100011 | 109 | 109 | 109 | 94 | 6.770 | 6.805 |
| 19 / 100019 | 109 | 109 | 109 | 85 | 6.725 | 6.900 |

Budget was 12 questions. A mechanically checked witness is required. All tested policies had zero false positives under the synthetic verifier.

The averaged query difference is small and does not justify superiority claims. The 300-step seed-7 masked reply BCE reached 0.5461, and family-risk BCE reached 0.4481. Losses are end-of-training batches rather than a held-out likelihood evaluation.

## Budget sensitivity, seed-7 trained checkpoint, same 200 worlds

| Budget | MIRROR found | Paired found | Random found | Oracle found |
|---|---:|---:|---:|---:|
| 4 | 108 / 109 | 106 / 109 | 14 / 109 | 109 / 109 |
| 8 | 109 / 109 | 108 / 109 | 63 / 109 | 109 / 109 |
| 12 | 109 / 109 | 108 / 109 | 88 / 109 | 109 / 109 |

The oracle has privileged access to the target's answers. The model uses the same generative grammar for training and evaluation with disjoint RNG streams. Results establish operation of the recurrence, transport, learner, and witness checker. Novel semantic transfer remains untested.

## Interpretation / next gate

Investigate why the scripted paired baseline is nearly optimal on this compact synthetic task. Increase complexity by using unseen context-transformation graphs, ambiguous scope and corrections, and coherent targets with highly correlated features. Compare tied versus untied matched-size controllers and remove the transport operator. Freeze a target-independent semantic-equivalence verifier before real LLM tests; the synthetic verifier currently trusts the gym's declared equivalence relation.
