# PEF-03 | Frozen learned tied-attention history and causal memory

**Protocol was committed before evaluation:**
[PRE_REGISTRATION.md](PRE_REGISTRATION.md),
Git commit \`0394417c0aecbfb0d7d7c74c1db82ff59f3aec44\`.

This experiment is the first PEF campaign using a trained attention
recurrent model, following the pure affine-IFS results in
[Compactification-Theory PEF-01/02](https://github.com/MoralityLabAI/Compactification-Theory/pull/142)
and [MathCan](https://github.com/patrickdugan/MathCan/pull/44).

The existing \`lsa.mini_transformer_probe.MiniAttentionMlpResidualBlock\`
serves as one **tied** causal attention-MLP operator. At visit j a
two-token sequence consists of (current input bit, carried memory);
only the second token survives to the next visit. The causal mask
allows the memory token to attend to that visit's bit. Training
supervises only the final two bits, with an independent hash-split
and a separate probe-training split. All eight inputs are uniform
during parameter training.

The frozen-model readout compares:
- two-bit task accuracy on history-disjoint heldout sequences;
- bitwise ridge probe accuracy trained on separate probe-only sequences,
  with the heldout majority baseline for each bit;
- causally controlled flips of every historical input bit with the
  remaining history fixed;
- endpoint and per-loop *storage casts* to FP16/BF16, while all
  arithmetic inside the block remains FP32;
- input-embedding JVPs propagated through **ordered** hidden-state
  Jacobians, checked against independent full-recurrence autograd JVP;
- correlated Markov-source older-bit probes alongside a recent-bit-only
  analytical and empirical baseline.

Checkpoints are SHA256 sealed before evaluation and never trained on
the probe or heldout pool. A random-initialization control is measured
with the same architecture. Source-shift diagnostic examples may repeat
and should not be labeled history-disjoint.

This task measures local learned memory and tests the limitations of
linear probes. It is not a pretrained LLM, real-world task performance
measure, nor proof of the PEF affine information-capacity formulas
for neural models.

Reproduce from repo root:

    python -m pip install torch==2.10.0 pytest --index-url https://download.pytorch.org/whl/cpu
    python -m pytest research/pef03/test_pef03.py -q
    python -m research.pef03.pef03 --out /tmp/pef03_results

The default script runs **exactly** the registered 250-step/two-seed
experiment. Model binary files stay in hosted CI artifacts and the
commit contains only source, protocol and tests. Training is CPU-only.
