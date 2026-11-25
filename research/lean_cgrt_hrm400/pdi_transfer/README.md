# PDI transfer: public compiler preflight

This is a public smoke fixture derived from EvalMath [PR #73](https://github.com/MoralityLabAI/EvalMath/pull/73). It must not be submitted to the PDI-08 scorer as held-out evidence. The existing OLR pre-command scope incident rules remain binding. The task family is a simple Horn chain; a reusable theorem abbreviates that chain. All statements are constructed by the runner from integers, with one new target per fresh Lean process. No target declaration or previous target is included in the prefix. The verified arm has the root, the sham arm has an unrelated theorem, and the revoked arm lacks the withdrawn root. Matched retrieval has no useful library in this synthetic fixture. The symbolic router searches the declared graph, including a root shortcut only when available. It produces exact proof terms from the graph, never from a model.

Run:
    
    cd research/lean_cgrt_hrm400/pdi_transfer
    python -m unittest -v
    python public_smoke.py --lean /path/to/lean --out /tmp/pdi-public-smoke

CI pins Lean 4.19.0, runs 2 depths x 4 targets x 5 arms plus two negative sentinels. Every positive target must compile with no axioms. The stale root and self-reference must be rejected; timeouts do not pass. Sources, logs and receipts are artifacts. Compiler wall time includes root replay; expansions are symbolic graph operations, not inference tokens or Lean heartbeats. Repeating the root proof per target deliberately prevents claiming amortization from this smoke.

## Real pilot handoff

Before the 32-family experiment, authenticate a corrected pre-command gym and independently audit spent OLR/RAM/Lcgrt families. A custodian must freeze new 32-family/128-target data (hashes alone do not establish semantic independence), a model checkpoint, runner and equal family budgets. Do not train on this public smoke or reuse its counts as validation. The symbolic baseline must use the same allowed theorem library and charge search, replay, failed proposals and root overhead. Model proposals go through the existing hardened native environment and independent standalone replay, with dependency and axiom checks on every authorized target. Preserve root versions and acquisition/withdrawal chronology. Pass real receipts to EvalMath's PDI-08 scorer only after external attestation. No model inference or sealed cohort exists in this commit.

The next runnable work order: custodian generates the actual manifest privately, freezes a symbolic policy plus the chosen compact reasoner, implements the native adapter for model proposals, measures family-level budgets, and scores the five arms on the sealed data. No architecture promotion on this pilot.
