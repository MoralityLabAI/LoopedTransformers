# INK-FP-AUDIT-02: Tinker recursive self-improvement gym

CPU-only local orchestration for verifier-driven recursive self-improvement of
`thinkingmachines/Inkling-2-Mini:peft:65536:free` through Tinker, with DataGoblin-style
curriculum, strictly replayable generation acceptance, Aleph corpus exports and a synthetic
BPD proxy for analytic adapter updates. Results and limitations are in [REPORT.md](REPORT.md).

## Modules

| module | role |
|---|---|
| `resources.py` | sets `CUDA_VISIBLE_DEVICES=-1` and 1-thread BLAS at import; Windows Job Object (4 GiB committed-memory cap, below-normal priority); DRAM governor (start >= 6 GiB, pause < 3 GiB, resume > 5 GiB held 60 s, emergency stop < 1.5 GiB); resource receipts |
| `ledger.py` | append-only hash-chained JSONL ledger; tolerates a torn final line; LF-normalised file digests |
| `curriculum.py` | `Task` with source root/world, family, generator id+version, difficulty, verifier id, answer/certificate, verifier cost; partition by world root range; semantic-id dedupe; cross-partition leakage report; sealed tasks contribute ids only |
| `gyms.py` | 11 verifier-backed families (3 reused generators, 8 new exact generators); Lean tactic ranking recorded as unavailable |
| `rewards.py` | +1 correct, +0.35 high-effort rescue of an all-failed low-effort group, -0.5 repetition, -0.25 incomplete; token-cost variant; shuffled-rescue control; GRPO advantages; GSPO sequence ratio |
| `acceptance.py` | frozen criteria, predecessor-head check, decision persisted with both case sets; `replay()`; upstream record audit |
| `engine.py` | sample (2 low / 2 medium / 2 high) -> verify -> frontier -> advantages -> update -> paired evaluation vs predecessor -> accept/rollback -> exports; resumable; arms A-H; simulated backend for engineering tests |
| `tinker_backend.py` | free-model guard, 12,288 / 16,384 context guard, OpenAI-compatible effort route (audit-01 runner), native SDK sampling, LoRA training (`cross_entropy`, `importance_sampling`, `gspo`), checkpoint save/resume, usage receipts |
| `exports.py` | `rlt_gym.aleph_corpus.v1` streams: sft, repairs, outcomes, routing, research (show-your-work, model-generated, kept apart from verifier certificates) |
| `bpd_proxy.py` | numpy proxy: ordinary, ridge (soft preservation), truncated SVD, projected `P = I - X_keep^+ X_keep`, low-rank projected, matched random; adaptive rank allocation; update-strategy router |
| `stats.py` | exact-McNemar power preflight; FSI-01 registration and bridge to `research_gym/scripts/evalmath_fsi_gate.py` |
| `preflight.py`, `campaign.py` | stage gates and the resumable campaign driver |
| `traces.py` | hill-climb research-logic traces (`ink02.hillclimb_trace.v1`) from acceptance ledgers, the upstream record, BPD and MeTTa climbs; Aleph `research` export |
| `hillclimb.py` | traced hill climb over analytic update strategies on the synthetic BPD proxy |
| `metta_driver.py` | adaptive proposer driver for DataGoblin `MeTTa/ram_hillclimb.py` (seeded-random, coordinate-sweep, file queue, `inky`); `--wide-grammar` extends the grammar in memory within the student's bounds |
| `inky_proposer.py` | Inky as the MeTTa edit proposer: prompt from grammar, configs and history; one free-tier request per turn via the audit-01 runner; invalid replies recorded, never retried |

## Task families

| family | generator | verifier |
|---|---|---|
| modular_arith | new | exact_int |
| boolean_parity (XOR/AND/OR worlds, parity query) | new | exact_int |
| dyck (bounded, 4 bracket types, closing suffix / INVALID) | new | exact_token |
| pointer_chase | `ink_fp_audit_01.core.roots` | exact_int |
| register_overwrite | new | exact_int |
| multislot_copy | new | exact_token |
| mcp_retrieval (hidden-symbol records, provenance traps, abstention) | `HybridTRMLDT/MCP/environment.py:make_cases` (subprocess) | exact_token |
| storyworld (plan to secret ending vs modeled rival) | `research_gym.envs.coupled_storyworld` | plan simulated against the env |
| recurrent_stopping (halting of an iterated map; text proxy for TRM/LDT stopping) | new | exact_int |
| bpd_nullspace (which update directions satisfy X_keep d = 0) | new | exact_set |
| experiment_design (which arm breaks the matched budget) | new | exact_token |
| hillclimb_judgment (accept, or the first failing check, under the frozen rule) | new, labels from `acceptance.decide` | exact_token |
| lean_tactic | unavailable: DataGoblin `rlt_lean_rank` needs torch and a Lean-checked pool | - |

## Tinker capabilities (installed SDK 0.40.1, static inspection)

* Loss functions: `cross_entropy, importance_sampling, ppo, cispo, dro, gspo, stable_reinforce, custom`.
  GSPO is native, so arms D-G use `gspo`; C uses `importance_sampling` (GRPO-style advantages).
* Native `SamplingParams` fields: `max_tokens, seed, stop, temperature, top_k, top_p`. No reasoning
  effort. Effort control exists only on the OpenAI-compatible route (`reasoning_effort`), which returns
  text, not sampled token ids or logprobs. Training items from that route are re-tokenised and scored
  with `SamplingClient.compute_logprobs` to get behaviour logprobs.
* Training client: `forward`, `forward_backward`, `forward_backward_custom`, `optim_step`, `save_state`,
  `load_state(_with_optimizer)`, `save_weights_for_sampler`, `save_weights_and_get_sampling_client`.
* Tensor access: logprobs and loss outputs only. No activations, gradients or adapter weights, so
  ridge / SVD / projected updates (arm H) cannot be applied natively; they are evaluated in the
  synthetic proxy only.
* Reasoning effort is an observable inference control. It is not internal recurrent depth.

## Reproduce

From the repository root, CPU only:

    python -m unittest discover -s tests -p "test_ink_fp_audit*.py" -v
    python -m research_gym.ink_fp_audit_02.preflight --out runs/ink02/preflight.json
    python -m research_gym.ink_fp_audit_02.campaign --out runs/ink02
    python -m research_gym.ink_fp_audit_02.bpd_proxy --out runs/ink02/bpd.json --seeds 20
    python -m research_gym.ink_fp_audit_02.hillclimb --out runs/ink02/hillclimb --climbs 6 --steps 24
    (cd ../DataGoblin && PYTHONPATH=. python ../LoopedTransformers/research_gym/ink_fp_audit_02/metta_driver.py --out ../LoopedTransformers/runs/ink02/metta --verify-against-run-gym --proposer seeded_random --climbs 8 --steps 16)
    (cd ../DataGoblin && PYTHONPATH=. python ../LoopedTransformers/research_gym/ink_fp_audit_02/metta_driver.py --out ../LoopedTransformers/runs/ink02/metta_inky --proposer inky --climbs 8 --steps 16)   # needs TINKER_API_KEY
    python -m research_gym.ink_fp_audit_02.traces --out runs/ink02/traces --campaign runs/ink02 --hillclimb runs/ink02/hillclimb --metta runs/ink02/metta

Remote stages need `TINKER_API_KEY` in the process environment and an explicit flag:

    python -m research_gym.ink_fp_audit_02.campaign --out runs/ink02 --allow-remote

The campaign resumes from `campaign_state.json` and the per-arm `ledger.jsonl`. Blocked stages are
recorded with their reasons and re-attempted on the next invocation. Stage A reuses the audit-01
runner (two roots x three efforts = six requests, 512 completion tokens, no retries, no paid fallback).
Stages B-F stop after Stage A and wait for its receipt to be reviewed.

The FSI-01 bridge needs EvalMath's `spiral_ln.evaluation_finite_sample` (found in a sibling
`EvalMath-mit` checkout here); `stats.evalmath_src()` locates it.
