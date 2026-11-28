# LOGRANK-BPD-04: assignment-constrained query codes

Adaptive development follow-up selected after early LOGRANK-BPD-03 results. Same four exposed root IDs, task permutations, splits, 500 updates, architecture, bit budget, learning rate and batch sequence. No extra query target labels. The training objective chooses an exact maximum-weight bipartite assignment between the eight local query indices and eight codewords using scipy.optimize.linear_sum_assignment on detached log p(c|j); add -0.5 times the mean selected log likelihood to task BCE. The assignment is recomputed per update, and gradients flow through selected log probabilities only. This is self-generated codebook supervision, with access to all eight local query identities at training time; it is not an entirely unsupervised task learner. Runtime still sends exactly four hard bits. Extra training compute is not matched to the unguided baseline.

Reproduce: `python experiments/logrank_bpd_04/pilot.py --steps 500 --out receipt.json`. For audit, copy the 03 audit.py next to the 04 pilot.py, then run it on receipt.json (it checks the neighboring pilot source hash). No native BPD or TRM training.

| Arm | heldout /4096 | full /8192 | query codes | full roots above one-way ceiling |
|---|---:|---:|---|---:|
| assignment | 4042 | 8116 | 8,8,8,8 | 4 |
| assignment_quadratic | 3788 | 7604 | 8,8,8,8 | 3 |

Linear assignment improves heldout by 911/4096 = 22.24pp over 03 joint linear, 98.68% heldout and 99.07% full. All eight assignment-trained models produced injective hard query codes, but root 607 quadratic produced only 1507/2048 full correct despite this. Keep that failure: address injectivity is necessary for exact success, not sufficient; the response and decoder can still fail. No complete matrix exact success. The independent transcript audit checked 32,768 saved records (16,384 full cases) with zero violations. Raw receipt digest and all outcomes are in summary_receipt.json; raw JSON is retained outside git.

Next: prospective fresh families and roots, compare frozen assignment, soft joint MI, fixed codebook and matched compute. In larger address spaces use an assignment over a restricted codebook or prefix hierarchy; do not claim unrestricted 2^q path enumeration scales. Test different query distributions, masked parity verifier queries, and sparse routing with native BPD factors. Separate causal interventions on address and response, matched-dose sham, recovery/corruption, full-head fidelity. Calibrate actual independent-root effects and variation via EvalMath before any confirmation; this adaptive pilot does not confer readiness.
