# InkyNano

Research program: compress Inkling 2 Mini into a text-first, locally runnable looped MoE with an LDT-controlled resident expert bank. Target: RTX 3050 4 GiB with CPU RAM backing, batch 1. This is an experimental scaffold, not released weights or an inference implementation.

## Per-token LD-MoE (current direction)
- [LD-MoE theory](LD_MOE.md) and [experiment ledger](LD_MOE_EXPERIMENTS.md).
- [Ordered native trace and exact cache replay](LD_MOE_PROGRESS.md).
- [Trainable causal eviction forecast](LD_MOE_FORECAST.md).
- [Codex implementation goal](LD_MOE_CODEX_GOAL.md).

## Cloud-to-local gym
- [GGUF conversion report](GGUF_CONVERSION.md): resumable Colab CPU conversion of all 3,072 experts + text backbone to GGML GGUF bundles, 16-expert receipts, and selective LDT loader (`gguf_convert.py`, `gguf_loader.py`, `inkynano_colab_gguf.ipynb`).
- [Stage 1 report](STAGE1.md): local controller sweep, receipts, and rerun commands.
- [Prompt-routing matrix](PROMPT_MATRIX.md) and [pilot results](prompt_matrix_receipts/pilot/PILOT_REPORT.md).
- [Stage 2 specification](STAGE2.md): native packed residency and activation-aware refinement.
- [GYM.md](GYM.md): handoff schema, cloud export and RTX 3050 post-training commands.
- [SILICO_GOAL.md](SILICO_GOAL.md): copy-ready bounded cloud job.
- gym_export.py, gym_data.py, gym_train.py, gym_manifest.py: native capture, spectral export, causal data and local controller.

## Start here
- [SPEC.md](SPEC.md): mathematical contract and comparisons.
- [SILICO.md](SILICO.md): staged spectral atlas work order, stop gates and Hugging Face packaging.
- [REGISTRATION.md](REGISTRATION.md): prospective development protocol.
- [routing.py](routing.py): dependency-free teacher-trace scoring and byte-accounted residency simulator.
- [spectral.py](spectral.py): single-matrix SVD and rank-prefix diagnostics (PyTorch).
- [test_routing.py](test_routing.py): synthetic routing unit tests.
- [test_spectral.py](test_spectral.py): optional CPU SVD tests.

Run from repository root:
```sh
python -m unittest discover -s InkyNano -p 'test_*.py'
python InkyNano/routing.py trace.jsonl --resident 8 --horizon 0 --expert-bytes 497664
python InkyNano/routing.py trace.jsonl --resident 16 --horizon 64 --expert-bytes 497664
```
The example byte cost is ideal packed INT4 at rank 96 for three 2304x1152 matrices, excluding scale and alignment overhead. Horizon zero denotes whole-turn sets. Report actual packed bytes for real runs.

## Source
- https://huggingface.co/thinkingmachines/Inkling-2-Mini
- https://huggingface.co/docs/transformers/main/model_doc/inkling2_loop
- https://thinkingmachines.ai/news/inkling-2-mini/
- https://docs.nvidia.com/nemo/automodel/model-coverage/vision-language-models/thinkingmachines/Inkling-2-Mini

Pin immutable checkpoint and implementation revisions before extraction. Source license Apache-2.0; retain license, attribution, source provenance and modification notices for derivative artifacts. The published conversion and earlier pilot artifacts are documented in their respective receipts.

## Milestones
INK-NANO-00: tensor inventory and dequantization audit.
INK-NANO-01: nested SVD pilot, then atlas if the pilot passes.
INK-NANO-02: teacher trace collection and causal residency evaluation.
INK-NANO-03: LDT mesh trained against fixed classical baselines.
INK-NANO-04: physical RTX 3050 benchmark with measured PCIe and GPU allocation.
