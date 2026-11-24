# Native o_proj output-coordinate bridge: random preflight

Register before executing the native random fixture. The physical layer-output
SAE site is after the MLP; o_proj output is not in that site's identity frame.
Use the native remainder F(a)=r+a+MLP(norm(r+a)), with the actual pre-attention
residual r and o_proj output a, capturing the input to the pre-MLP norm
independently. Their native residual addition must reproduce that capture.
Keep the full prefix GEMM shape. Only the last-token physical o_proj output is
perturbed. Do not subtract rounded downstream tensors to reconstruct a state.

The helper computes native autograd directional derivatives of this remainder
and centered finite writes at relative o_proj-output L2 doses .001/.01/.1.
Retain actual rounded injection and post-addition displacements, all errors and
directions, including zero/rounded-away writes. Tangents are cast to the native
dtype and returned explicitly. BF16 autograd is a derivative surrogate; no
equivalence bound with finite rounded writes is asserted or tuned.

CPU proof independently derives the RMSNorm/SwiGLU Jacobian in float64, checks
its chain rule, finite-write agreement and parameter immutability, and rejects
bad coordinate bindings, live component hooks, trainable/quantized modules,
hyper-connections and inference_mode. This is not the real model proof.

Native fixture: random 22-layer model, seed2026112013, hidden64, MLP128,
eight query heads/two KV heads/head16, vocabulary4096 (no head readout), fixed
19-token prefix drawn 0..99 with the same seed, two native passes, BF16 staging.
Use 16 predetermined random physical output directions and every dose/sign at
both passes. Remove capture hooks before JVPs, preserve the actual layer21
stage and no trained components. Each baseline local remainder must be byte
exact to the captured full native layer output. For every signed write, run a
fresh full two-pass native forward, inject the write at the actual selected
o_proj output/pass, and require byte-exact equality to the local remainder's
prediction at that physical layer output. Preserve all192 signed comparisons
and 96 centered finite-write diagnostics. Require finite JVPs and unchanged
target parameters. Report any derivative/finite-write differences without
turning a diagnostic into a hidden acceptance threshold.

No pretrained tensors, task labels/corpora or sealed data. One CPU thread,
Below Normal, CUDA cap .05, 600-second between-comparison budget. Run only
after the separate small rank/site pipeline finishes and >=512MiB headroom is
verified; no third model process. Large registered inference continues unchanged.

Unit: one random checkpoint/seed, two correlated passes and repeated directions/
doses. Physical engineering doses as above; no mechanistic or matched-sham
floor/power. Formal comparisons zero. No immediate-readout preservation,
delayed causal effect, pretrained feature fidelity or donor transfer is claimed.
Real-checkpoint use still requires separately frozen scope/source bindings and
actual per-source coordinate/closure checks. The original 168-case test stays
sealed and all original failed gates remain unchanged.
