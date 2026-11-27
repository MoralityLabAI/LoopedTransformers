# INK-FP-AUDIT-01 v0.1

Independent development-only audit of [Recursive-Self-Improving-Loopers](https://github.com/eelbaz/Recursive-Self-Improving-Loopers/tree/765453d13878804c49876a656f39b57a4384dca4). No upstream code vendored. No checkpoint has been evaluated here.

## Source findings

The published gate replay follows the *reported* predecessor, not a counterfactual training history. Rounded mode percentages recover integer counts on 64 cases. The five reported selected candidates give:

| generation | predecessor | candidate | per-mode losses R3/R4/RK2 | rule |
|---|---:|---:|---|---|
| 0 | 111 | 112 | -1/-1/1 | accept |
| 1 | 112 | 114 | -4/-2/4 | reject |
| 2 | 114 | 118 | 3/-1/-6 | reject |
| 3 | 118 | 114 | -3/1/6 | reject |
| 4 | 114 | 116 | 4/1/-7 | reject |

The code allows at most two correct-answer losses in one mode; its published records mark all five accepted. This is a record/code discrepancy, not proof of invalid benchmark results. The fixed gate is adaptive validation. The FP-Solve ridge is soft preservation, not a strict nullspace guarantee. Its public solver calls torch.linalg.solve, then caps the norm. Its cache restoration does not explicitly remove state created by intermediate steps when the original slot was empty; use the synthetic roundtrip audit before any damped cache experiments.

## Reproduce

From the repo root:

    git clone https://github.com/eelbaz/Recursive-Self-Improving-Loopers upstream-rse
    git -C upstream-rse checkout 765453d13878804c49876a656f39b57a4384dca4
    python -m unittest discover -s tests -p test_ink_fp_audit.py -v
    python -m research_gym.ink_fp_audit_01.core --record upstream-rse/results/gspo_results.json --out gate-replay.json
    python -m research_gym.ink_fp_audit_01.cache_contract --upstream upstream-rse

For native comparison, use local snapshots and Transformers 5.24.0, NumPy, PyTorch, accelerate, safetensors and psutil. The existing INK-ROUTE-01 packed-loader avoids expanding expert tensors; it is verified for the base checkpoint only. Confirm the edited checkpoint has the same packed layout and tokenizer and supply its independently recorded weight SHA-256. Keep source provenance separately. No downloads or hosted inference are performed by the runner.

    python -m research_gym.ink_fp_audit_01.local --checkpoint base --snapshot PATH --sha256 65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1 --upstream upstream-rse --device cpu --roots 2 --depths 3 4 --out runs/ink_fp_base_smoke

Repeat with --checkpoint inkblot, its own snapshot/hash and distinct output directory. Only after native/wrapper parity and successful two-root smokes, expand to --roots 8 --depths 1 2 3 4 5 6 7 8. If wrapper API incompatibility or parity fails, stop and adapt against native Transformers with a fresh version. Cache-free generation, 128-token default, exact explicit integer grader; report completion and repetition separately. This is not a reproduction of DeepScaleR. No automatic editing or third-party GPU spending. Native vs wrapper parity is a prerequisite, not sufficient proof of cache or depth equivalence. No damped arms until cache roundtrip passes.

## Next-stage protocol

Measure L17 last-token shift spectra by fitting an SVD on development roots only. Test a frozen direction on distinct roots; compare it to coordinate 2027 and equal-L2 orthogonal random sham, matched sign-flip, and clean-reader rescue. Track the complete recurrent state, gate channel and routing responses. Cache parity requires both initially empty and populated state, prefill/decode equivalence and actual generation tests. Keep all failures. For direct edit follow-up compare ordinary ridge, exact preservation projector, random matched-rank edit and damped recurrence; compute matched readout cost. Use one checkpoint at a time.

Primary future causal readout: verified correctness gain of target direction intervention over equal-L2 sham, root macro mean, minimum meaningful gain 0.10; dosage in realized BF16 writer displacement. Independently calibrate sham with at least 59 development roots if estimating a nonparametric 95th-percentile/95%-confidence envelope. Freeze causal dose, comparison family, selection exposure and sealed confirmation set; run EvalMath mechanistic-power preflight, require enough independent roots, then FSI-01 using research_gym/scripts/evalmath_fsi_gate.py. Eight public roots are engineering diagnostics. No statistical confirmation, no claim of a negative eigenvalue, no claim that cosine implies contraction.

Future independent test needs checkpoint hashes, complete paired outcomes, exact verification, measured resource cost, frozen registration, and evidence class. For local execution use CPU first; CUDA cap is allocator-only and host memory is checked at entry, not hard-limited. Packed-loader compatibility and GPU headroom remain untested for Inkblot. Checkpoint expert byte identity requires a separate streamed tensor audit; the supplied hash verifies identity of a snapshot, not equality across snapshots.
