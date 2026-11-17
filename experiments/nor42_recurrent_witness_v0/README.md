# NW-01: recurrent information boundary, witness retention and causal state swaps

This experiment bridges the **mathematically indistinguishable paired source worlds** in [Compactification-Theory NOR42](https://github.com/MoralityLabAI/Compactification-Theory/blob/0a330602c72ec34017178857956e306710f724a1/research/nor42_paired_program_ood/prospective/RESULTS.md) to a small **eight-pass tied-GRU recurrent verifier**. The [first registration](REGISTRATION.md), committed at `42983fba14f3a616e9697523bce8a533fa8c6a84` before model training, predeclares three fitting seeds, the source roster and five acceptance gates. The original NOR42 source/grade audit had 64 fresh source pairs and two equal-prior future worlds; all five registered NOR42 verification gates and its independent 1,179,648-component replay passed.

## Core experiment

The paired clean/hidden worlds have **exactly identical public demos and query**. The hidden true world differs from the clean continuation first at coefficient D+32 by a nonzero signed `β`. In the original published NOR42 experiment the true hidden coefficient was supplied by a **synthetic hypothetical authenticated oracle**, not a real deployed verification API. NW01 reads this first coefficient discrepancy **only in the authenticated-witness arm**. It also tests public-only and label-independent sham-witness conditions, with exactly the same public source features across both truth worlds.

The model uses a 50D source-public feature vector, a 48D hidden state, one tied `torch.nn.GRUCell`, eight recurrent passes, and a shared binary readout across passes. Its authenticated witness arrives only at the **first recurrent step**; later passes receive public features alone. This makes it possible to ask how an already revealed coefficient is retained by recurrence. It is a **tiny recurrent verifier**, structurally analogous to the shared GRU state in `research_gym.neural.trm.TRMProposer`. It is not the trained pretrained TRM, an RRC25 neural program inducer or BPD component decomposition.

## Information and causal controls

- **Public-only:** both worlds receive identical features; the model's paired hidden states and readouts must agree within the predeclared 1e-7 numeric tolerance at all eight passes. On each source pair, any deterministic classifier scores exactly one of the two equal-prior labels, establishing a mathematical **50% ceiling** independent of neural optimization. The pair-averaged Brier floor is 0.25.
- **Authenticated:** the clean world receives residual zero and the hidden world the actual first discrepancy β from the completed original NOR42 grade. The witness encoder receives `present`, `β/2` and `|β|/2` as information, once at pass zero. This is an explicit easy positive control for retaining informative evidence.
- **Sham:** the model receives a source-matched, label-independent residual signal of the same three-dimensional format. Both paired worlds see exactly the same sham residual. It tests whether merely presenting something that looks like a witness induces unjustified confidence.
- **Causal matched-state swap:** after recurrent passes 1, 2, 4 and 6, transplant the complete hidden state from the opposite true world of **the same public source**, then continue the tied GRU on the identical public input. The matched, opposite-world intervention is compared against an independently sampled **L2 dose-matched random-direction** perturbation. The final patched logit must agree with the donor's original final logit. Same-world state reapplication is an identity control. These are causal state-transport diagnostics and say nothing yet about BPD component bases.

## Training, reproducibility and evidence limits

Synthetic training uses 512 independent **non-NOR42 roots**, drawn from the old dense and four-regime input laws. The official held-out readout uses all **64 original, previously published NOR42 source roots**, clustered into their original 16 generator-replicate blocks. The original first source-only JSON SHA is `f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033` and first original grade SHA is `bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae`. The immutable first NOR42 ZIP is `e60e97fd933f660c2274cb4a473ede7a98968d24eaf0dd3ae45ace1e1176cd0d` at [commit `0a330602`](https://github.com/MoralityLabAI/Compactification-Theory/commit/0a330602c72ec34017178857956e306710f724a1).

**Fixed training:** 200 AdamW updates per model, 128 paired-view examples/step, latent 48, eight tied passes, seeds 11/23/47. The test evaluates readouts at passes 1/2/4/8 and causal patches at 1/2/4/6. Data loader verifies original NOR42 source/grade SHA before deriving the 64 source-public feature vectors. Its train/test IDs must be disjoint. Model source hashes, the three model state_dict checkpoints, raw per-pass hidden states and logits (`.npz`), original 64-source paired prediction rows, 16-replicate-block uncertainty and five original NW01 PASS/FAIL booleans are retained together in the hosted first-model artifact. A later archive receipt may commit the raw outputs permanently.

This is a **retrospective three-seed neural pilot using previously released data**, with **16 clustered generator blocks** rather than 128 independent truth worlds. It is not a new sealed source cohort or mechanism identification. Information equivalence is exact, while fitted witness accuracy, causal effect size and calibration are measured empirical outputs. The EvalMath gate treats the scientifically useful donor-minus-random logit-effect floor as at least 0.10, and no neural evidence is upgraded from prior Lean results.

## CPU commands

From a LoopedTransformers checkout with the `torch` and `numpy` neural extras installed:

```bash
python -m unittest experiments.nor42_recurrent_witness_v0.test_nw01 -v
python -m experiments.nor42_recurrent_witness_v0.run \
    --archive /path/to/verified/nor42_original_first_hosted.zip \
    --out /tmp/nw01_run
```

The [NW01 GitHub Actions workflow](../../.github/workflows/nw01-nor42-recurrent-witness.yml) fetches the immutable original NOR42 archive by pinned git commit and verifies its full ZIP SHA before test or training. This README commit will trigger the first **three-seed CPU model pilot** only after the predata suite passes. Original PASS/FAIL scientific gates are preserved even if the trained model misses the positive-control threshold.

**First model pilot trigger:** Original NW-01 three-seed run is initiated after the complete SHA and paired-state preflight passed. The original model result and every unselected arm are retained even if one or more registered gates FAIL.
