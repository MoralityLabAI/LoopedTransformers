# Tinker behavioral pilot (INK-FP-AUDIT-01 v0.2)

Official free-tier model ID: `thinkingmachines/Inkling-2-Mini:peft:65536:free`. See [pricing](https://thinkingmachines.ai/tinker/pricing/) and [chat API](https://tinker-docs.thinkingmachines.ai/inference/openai/chat-completions/). Free tier is rate limited; prepaid accounts need a positive balance. OpenAI-compatible beta accepts base model IDs per its overview; availability of this particular free ID via that route needs a live account smoke. If it is unsupported, stop and implement a separate documented Tinker SDK sampling path; never substitute the paid ID.

From repo root:

    python -m research_gym.ink_fp_audit_01.tinker_remote
    python -m research_gym.ink_fp_audit_01.tinker_remote --run --out runs/ink_fp_tinker_smoke
    python -m research_gym.ink_fp_audit_01.tinker_remote --run --roots 8 --out runs/ink_fp_tinker_dev8

Set TINKER_API_KEY in your local environment beforehand. No credentials in flags or logs. Dry run makes no requests. With --run, the default smoke uses two public pointer-chase roots x low/medium/high effort = six sequential requests, max 512 completion tokens each. Eight roots = 24 requests, maximum requested 12,288 completion tokens. Rate limit or any error stops with partial receipts. No retries, paid fallback, training, checkpoint export, or redirected credentials. Use a fresh output directory for each attempt. Capture final text, exact answer grade, repetition, finish reason, API usage, model identity and latency; reasoning text itself is not persisted. A cap is a requested completion limit, not an assertion about actual billed tokens. Check returned usage.

The API effort setting is a reasoning-budget intervention, not an observed internal recurrent depth. Do not label low/medium/high as R1/R3/R4. API does not expose Layer 17 tensors, internal pass control or raw FP-Solve edits in the documented endpoint. The edited Inkblot checkpoint is not assumed available in Tinker. Match local prompt, completion cap and decoding where possible; backend differences remain confounders. The eight roots are exposed development; future confirmation needs new sealed roots and EvalMath FSI-01. All mock tests are network-free. No live credentials or remote sampling results included.
