# DORY-GYM-01: elastic recurrence under resource constraints

## Native geometry

Dory1.0-4B is a dense normalized GPT with 16 input, 40 shared recurrent and 16 output layers in the paper's counting convention. One checkpoint supports k in {1,2,3,4}. Let u=I(x), s0=u; s(j+1)=N(F(N(s(j)+u))); logits=H(O(s(k))). N is L2 normalization. Verify exact runtime implementation before intercepting states. The paper mixes sliding and full attention and uses a last-loop KV cache as a memory-saving option. The released GGUF is Q5_K_M from quantization-aware distillation; do not treat BF16 benchmark numbers as measured GGUF quality.

Per-token layer-application proxy L(k)=32+40k. For k=1,2,4: 72,112,192. Count prompt tokens too; proxy work = L(k)*(prompt_tokens+completion_tokens), not FLOPs, latency or energy. Actual timings, GPU memory and cache allocation must be measured. Longer loops may reduce generated length.

## Stages

DORY-00: native build/checkpoint verification. Verify model SHA, backend, offload fraction, loop expansion and cache flags in startup log; one server at a time. Begin at 512 context, small microbatch and conservative partial offload.

DORY-01: fixed depth {1,2,4}, per_loop cache, identical roster, token cap, template, sampling seed and backend. Report paired exact answer scores, output lengths, wall time, model server timings, failures, and resource peaks. DORY-02: repeat matched cells with last_loop cache. Include k=1 cache equivalence as sanity control. Per-loop vs last-loop may alter outputs; retain raw outcomes. DORY-03: train a cheap pre-request controller on train roots only. Evaluate once on dev for diagnosis; choose depth based on available prompt features. Include fixed-depth, random allocation at matched mean depth, category/length lookup and hindsight ceiling. This scaffold implements a deterministic category/length lookup over training scores with a fixed cost penalty and a hindsight diagnostic; a learned lattice mesh and exact matched-budget random baseline are follow-up gates.

The offline selector must choose depth before observing any output, with no access to future hidden states or correctness labels. Offline replay of already captured outcomes is observational feasibility. A runtime per-token early exit requires modifying native execution and verifying cache/state semantics; variable depth per request is the first viable intervention.

## Mechanistic extension

For normalized recurrent state s_j, measure angular displacement D_j=1-cos(s_j,s_(j-1)), logit divergence across depth on the same prefix, and norm of reinjected u relative to the update before normalization. Capture in native runtime with per-layer/pass IDs and consistent token alignment. Do not infer convergence from small D alone; test whether extra passes produce correct answer flips or reduce error at fixed cost. Compare matched sham direction/size, and fixed-depth controls. BPD/LOOPSCOPE extensions should use EvalMath mechanistic-power skill and FSI-01 per AGENTS.md.

## Registration (development only)

Unit = independently sampled problem family/source root; examples derived from a template are clustered. The included exact-answer roster is a synthetic smoke suite, not a powered accuracy benchmark. Frozen train/dev split is family-disjoint. Record independent root count, family count, all candidate arms and runtime failures. The meaningful prospective utility gain is +5 percentage points correct at matched mean work (or >=15% reduction in measured work with a prespecified -5 point noninferiority margin). Neither threshold is a result. Use paired root/family inference, power planning and a fresh sealed confirmation set after pilot variance estimates; no confirmation data created here. Distinguish numerical parity, synthetic checks, offline observational policy replay and native causal intervention. Compare actual elapsed time separately from layer-application proxy.

DataGoblin transfer: difficulty-aware depth curriculum, shared loop weights, re-injection and last-loop KV cache as independently controlled factors. InkyNano transfer: loop utility and KV handling independent of expert-residency quality. Do not conflate the two architectures.
