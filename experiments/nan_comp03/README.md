# NAN-COMP-03 bounded answers and reciprocal examples

This fresh development pilot follows NAN-COMP-02. [Protocol](PROTOCOL.md), [configuration](config.json), [source seal](SOURCE_MANIFEST.json), and all admitted synthetic sources were frozen and pushed in prospective commit [fca024e7435b803333f7ee09f02ef53e7125f9ba](https://github.com/MoralityLabAI/LoopedTransformers/commit/fca024e7435b803333f7ee09f02ef53e7125f9ba) before any new pretrained inference. The original 168-case ARG-03 test remains sealed. BPD transfer is disabled regardless of this readout.

The main arm adds an explicit series-reciprocal coefficient rule and three verified examples. The balanced control shares the same base prompt. Output erasure retains the curriculum but hides current demonstration outputs. Unrestricted full-vocabulary greedy tokens feed a fixed answer grammar: bare or balanced bold/italic/inline-code uppercase I/N/S/R, optional period, whitespace ignored. Generation stops at the first complete grammar prefix, impossible prefix, EOS, eight tokens or 64 characters. The model receives no logits mask, forced label, search or fallback. Later free-running prose is not measured. Full prefixes are recomputed without a KV cache through both native loops at every token.

Ten source roots (two CAL/eight DEV) yield 20 worlds. Sources exclude prior NAN-COMP-02 exposed input pairs and tutorial inputs; actual rendered tokens and unordered input pairs are unique across CAL/DEV. Only the specifically named previous public development fixture is read for exclusions. Exact sources and all rejection counts are retained. Current labels/query futures are never rendered. CAL selects or fits nothing.

## Weight-free replay

    python -S -m experiments.nan_comp03.audit --out audit.json
    python -m unittest discover -s experiments/nan_comp03/tests -v

The independent stdlib auditor uses a separate anchored regex parser and finite-series checker. It verifies all source certificates, recorded parser stops, prefix hashes, answer scores, scalar grades, root contrasts and nine-entry Holm accounting. Unit tests require Torch/NumPy. Offline tokenizer verification is a separate receipt and does not load model weights.

## Full local reproduction

Use a new empty run directory and the existing pinned checkpoint; never use an old ARG-03 roster. Keep HF_HUB_OFFLINE=1 and point HF_HOME to the existing cache. Use the sibling EvalMath and ARG-02 adapter revisions pinned in SOURCE_MANIFEST.json. GPU runs use Python 3.12.5, Torch 2.6.0+cu124, Transformers 4.46.3, bitsandbytes 0.50.2 and NumPy 2.5.5. CPU CI pins Torch 2.6.0/NumPy 2.2.6.

    python -m experiments.nan_comp03.validate_gpu --snapshot <pinned-snapshot> --out <new-runs>/engineering_recheck
    python -m experiments.nan_comp03.cohort --evalmath-root <EvalMath-checkout> --compactification-root <ARG02-checkout> --snapshot <pinned-snapshot> --out <new-runs>/cohort
    python -m experiments.nan_comp03.run --backend bf16 --compactification-root <ARG02-checkout> --snapshot <pinned-snapshot> --runs <new-runs>
    python -m experiments.nan_comp03.run --backend nf4 --compactification-root <ARG02-checkout> --snapshot <pinned-snapshot> --runs <new-runs>
    python -m experiments.nan_comp03.grade --runs <new-runs> --evalmath-root <EvalMath-checkout> --out <new-runs>/graded
    python -m experiments.nan_comp03.verify_tokenization --runs <new-runs> --compactification-root <ARG02-checkout> --snapshot <pinned-snapshot> --out <new-runs>/tokenizer_audit.json
    python -m experiments.nan_comp03.analyze --runs <new-runs> --out <new-runs>/analysis --plot

After the recheck passes, retain it separately and copy the archived receipts/gpu_fixture.json into <new-runs>/engineering/gpu_fixture.json. The registered worker binds that original receipt byte-for-byte (including its original elapsed-time measurement); a new fixture run is validation evidence, not a replacement registration.

Reproduction source/fixture bytes must match the published SOURCE_MANIFEST; the runner refuses modified sources or a different cohort. Generate sources only to reproduce the exact deterministic registered cohort, not to substitute new cases. All locked checkpoint files are verified before loading. The full-vocabulary BF16 head and embeddings remain on CPU. BF16 decoder layers stream; NF4 decoder linear weights reside on CUDA with double quantization and BF16 compute. Six independent B1 contexts share layer loading; random-weight fixtures show exact parity against serial growing prefixes. The same full-head implementation passed its earlier exact-dimension fixture.

Run BF16 then NF4 sequentially with a 75% CUDA allocator cap, 768-prompt-token cap, and 2400-second soft limit between roots. At most one hash-bound resume may finish pending fixed pairs, preserving attempts. Windows CIM GPU allocation sampling needs permission; shared allocation is not a paging measurement. Actual launch arguments and memory samples are archived. Model weights are not included in this repository or share bundle.

## Interpretation

The registered main gate separately requires 12/16 verified parsed answers, 6/8 both-correct pairs, 3/4 reciprocal worlds, and a 0.25 parsed-accuracy gain over erasure in each backend. Strict-first-token and first-token restricted ranking stay separate. Passing this operational gate would establish only a development prerequisite on this unary condition; the original compound-program failure and prior studies are unchanged. No BPD or recurrence-geometry claim follows.

Nine comparisons are frozen. Source-only global deduplication induces dependence, so FSI does not assert iid roots or paired-label exchangeability and withholds unconditional p-values. Conditional sign-flip sensitivities and Holm retain all nine contrasts. Eight roots are power-limited and cannot calibrate a 95%/95% sham-quantile bound requiring at least 59 independent sham worlds. Repeat-erasure noise is numerical/format noise only. Pre-data planning honestly refuses an empty empirical pilot; historical sham values are not substituted.

[Full competence report](../../papers/shared_weights_dark_fibers/NAN_COMP03_BOUNDED_RECIPROCAL_REPORT.md). Both registered competence gates fail; all evidence and non-inference telemetry limitations are retained.
