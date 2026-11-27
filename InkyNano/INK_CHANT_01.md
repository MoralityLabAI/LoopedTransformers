# INK-CHANT-01: import capability support into the shared expert bank

Status: importer and synthetic reference tests, no native model-quality or timing evidence.

## Source and scope

Chantless masks pinned at https://github.com/DeusExperiments/chantless/tree/b532d51f81b031fced1023676656ad263c22b4df/masks
and the published InkyNano bank both identify thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7. Chantless used a different BF16 GGUF representation (7074c9762e6769a608c7b23960a7fd2444880b1174a3910149749f6d01389aec). The bridge imports only original expert IDs, not the quantization tiers, vocabulary, FFN width, or apprentice ablations. The source checkpoint match establishes provenance; real native expert-index and output parity still require execution.

Independent decoding of all three released supports at the pinned revision:

| Mask | Expert count per physical layer | Original expert bits verified |
| --- | ---: | ---: |
| BFCL | 128 | no explicit bitmap; unrestricted |
| DeepMath | 96 | 24 x 128 |
| clsbench | 48 | 24 x 128 |

Math/classifier intersections by physical layer: 46,44,42,46,45,46,45,41,37,43,42,41,46,43,43,41,43,46,47,44,41,45,44,43. Mean 43.5 of 48 classifier experts. This is static mask overlap, not token routing overlap.

## Router semantics

For native sigmoid scores s and correction bias b, select the highest eight s+b in an immutable allowed set S_l. Normalize the selected unbiased scores with the native scaling factor. Store the ordered original physical IDs and compact-local-to-original mapping. At the same supplied scores and bias, compact slicing and full-coordinate hard masking agree with consistent tie breaking. At eight allowed experts the membership is forced; cache residency and allowed support must remain distinct. Misses request the selected original expert and never silently substitute an available expert.

The reference accepts the full 128 coordinates and returns original IDs, local IDs, weights and misses. It deliberately uses deterministic increasing-ID tie breaking. Native GPU/CPU near-tie behavior and accumulated recurrent state divergence remain untested.

## Reproduce static support using existing published GGUFs

From the repository root on a host with the published full bank, pinned original HF source metadata/tokenizer, and the compatible pinned llama.cpp conversion checkout described in LD_MOE_PROGRESS.md:

```sh
python -m unittest discover -s InkyNano -p test_chantless_bridge.py
python InkyNano/chantless_bridge.py --mask "$CHANTLESS_MASK" \
  --manifest "$BANK/manifest.json" --out math-support.json
python InkyNano/compose_published_bank.py --bank "$BANK" \
  --manifest-sha256 0601e90943abc2320e9ea8dc9ac205fc62fe4b60df26dbe66a8708472433004e \
  --llama-source "$LLAMA_SOURCE" --source-metadata "$SOURCE_METADATA" \
  --selection math-support.json --max-model-bytes 16106127360 \
  --out math-support.gguf
```

The pinned manifest checksum is the published full bank's documented value; stop on mismatch and audit a different bank before substituting a checksum. The composer copies existing packed weights and slices the original router/bias; it does not need the original BF16 weights. The max-model-bytes argument is a disk/output-size cap, not VRAM or process RSS. The output includes a composition receipt with resident_by_layer mapping. Validate native runtime identity and route traces using the ordered capture procedure in LD_MOE_PROGRESS.md with --published-composition pointing at that receipt. Use CPU/RAM-backed demand paging or a host with adequate memory; do not assume the compact model fits a 4 GiB GPU. Preserve a 12k default / 16k maximum context policy for later quality trials; start with the existing 512-context ordered smoke.

## Development experiment

Before a larger run, establish full-support quantized bank quality versus the original native teacher. Use matched prompts, inference budget, quantization and runtime. Test unrestricted, imported DeepMath support, random uniform 96-expert support, and train-only demand-ranked 96 support. Run the 48-expert controls separately for classification. The BFCL mask has unrestricted support and cannot be a pruning comparator. Keep expert IDs, task/root split, prompt, model and manifest identities, native weights, transfer counts and verified task results. Use independent roots and the repository's FSI-01 gate before any promotion; no claims from synthetic router parity.

Next phase: native runtime request-level allowed support, with exact demand paging, before testing pass-dependent or learned masks. For a 4 GiB target evaluate k=8/12/16 physical resident slots with logical support 48/96/128. Charge cold start, miss bytes, H2D/CPU time and memory. At k=8 and eight active IDs, cache-only eviction cannot reduce demand misses relative to the same ordered trace. Compare prefetch or CPU bypass with their costs; changing selection changes the model and needs a separate quality test.

## Local receipt

Six unit tests passed in a standalone Python 3 environment: bitmap decoding, released layer-zero counts, invalid inputs, mock bank accounting, 1,400 synthetic compact/full-coordinate routing comparisons, ties and bias. Exact tested file Git blob hashes: d6c4e4b4836608bde028e2bdefcec977a7452335 (bridge), f67590559e5cb36295c2039b25ced9771fb041c6 (tests). Both were verified against the committed files. No private weight access, native composition, CUDA benchmark, full repository suite, or cloud spend in this turn.
