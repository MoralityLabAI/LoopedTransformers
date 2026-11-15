# COE-09: label-synchronized phase probes

Extends the merged [COE-07 native LoopScope experiment](../phase_gauge07/PROTOCOL.md) and COE-08 shuffle null. Pair separation of a K9 evaluator is blind to digit-label permutations, so COE-09 measures **held-out digit label decoding** from fit-labeled code prototypes before making claims about preserved reasoning. The optional causal runner rotates a selected direction within the pretrained model's row-space, dark-fiber or scratchpad subspaces at fixed recurrent visit 4, while preserving the corresponding geometry.

## Portable checks

```shell
python -m unittest discover -s mechinterp/phase_gauge09 -p 'test_*.py' -v
```

Synthetic frozen source replay, only when the original COE-04 artifacts are accessible (the original checkpoint bundle lives in Compactification-Theory):

```shell
python -m mechinterp.phase_gauge09.replay_coe04_synthetic \
  --activations /path/to/frozen_loop_activations.json.gz \
  --receipt /path/to/temporal_receipt.json \
  --permutations 10000 --out /tmp/coe09_synthetic.json
```

## Pretrained execution

First obtain the original COE-07 artifacts (or run COE-07 native extraction). Both files must pass source SHA and shape validation.

```shell
python -m mechinterp.phase_gauge09.native_label_replay \
  --activations /path/to/native_activations.npz \
  --receipt /path/to/receipt.json \
  --permutations 2000 \
  --out /tmp/coe09_native_alignment.json

python -m mechinterp.phase_gauge09.native_causal \
  --activations /path/to/native_activations.npz \
  --receipt /path/to/receipt.json \
  --device cuda --threads 4 \
  --out /tmp/coe09_native_causal.json
```

For a fresh CPU native run, use `python -m mechinterp.phase_gauge07.run_native --threads 4 --out /tmp/coe09_native` before these steps and replace `--device cuda` with `--device cpu` if necessary. The full native run requires `huggingface_hub`, the pinned 7M checkpoint, NumPy and PyTorch. All 19 portable Python tests and four independent exact mathematical tests passed locally; native model evaluation remains a separate gate.

The paired MathCan manuscript is `MathCan/research/compactification/OBSERVER_LABEL_ALIGNMENT.md`. The source papers OpenAI Math #098 and #332 motivate the evaluator-relative representation problem; the combinatorial and causal-geometry claims have direct proofs in the manuscript.
