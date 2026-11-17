# LS-SAE12 status

| Stage | State | Where |
|---|---|---|
| Registration | committed `f73d40b` before any pool fetch or model run | `active_reasoning_v1/PROTOCOL.md` |
| Unit tests | 12/12 pass locally (Windows, Python 3.12.10, PyTorch 2.7.1+cpu) | `active_reasoning_v1/test_active.py` |
| Source-only admission | **all 48 pool puzzles admitted** (`4e580db`); independent audit pass, including both solvers and disjointness from 36 earlier puzzles | `source_48.json`, `source-audit.local.json`, `source_stage.local.log` |
| Engineering pilot | ran locally on the 4 LS-SAE11 pilot puzzles in 1127 s; 3 eligible; every state passes the independent auditor | `engineering_pilot.local.summary.json`, `engineering_pilot.local.log`; full receipt on branch `evidence/ls-sae11-actions-44042014383` (`ls_sae12_engineering_pilot/`, SHA256 `848ed4fa…77fc`) |
| Registered screening + `active_a` + `active_b` | runs once, in the workflow triggered by pushing the commit that adds the push trigger (`mechinterp-ls-sae12-active-reasoning.yml`, job `registered-screen-and-two-families`); its artifacts are copied to branch `evidence/ls-sae12-run-<run_id>` | — |

The trigger change touches only when the workflow runs. Pool, rules, seeds, doses, outcome and gate are as registered in `f73d40b`. A `pull_request` run repeats only the unit tests and the source-only admission, so opening a PR does not create a second registered run.

Pool: 56–60 holes, singles stall after 0–5 deductions, Algorithm X nodes 377–9882 (median 1413), DFS nodes 3709–52236; pool fingerprint `f55292ae41abe96c26765c4373744fd82c66b72c4b0500de79db7a390ef51103`; `source_48.json` SHA256 `8c31659d3ccbc56f5aafa5c63511e98207a709fff98f8a4d82e48a57e212bede`. The CI source job must reproduce this file byte for byte.

Engineering pilot observations (no gate, nothing in the protocol changes because of them):
- The batch-61 clean row reproduced the batch-1 screening digits at all 12 continuation steps in every state on this machine, and both power iterations converged (last relative change ≈ 1e-7).
- The STRUCTURAL and LOGIT-SENSITIVE directions nearly coincide (cosine 0.996–0.999), so the secondary `D_SL` contrast will be close to degenerate. The primary `D_SI` is unaffected.
- Even at dose 0.02 every arm, isotropic included, moves many verified deductions (isotropic mean VDD 52–189 per state), and isotropic one-step hidden gain is 0.5–4 against 10–64 for STRUCTURAL.
- Per-state local run time was about 6 minutes; the 24 registered states plus screening should fit the job's 340-minute limit on a GitHub runner.
