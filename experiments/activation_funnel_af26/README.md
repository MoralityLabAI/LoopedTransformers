# AF-26: frozen STOP-gate × legal-rule-ranking factorial

AF-25's source-bound frozen eight-seed study reported **432/640 Direct versus 195/640 VI-2** on positive depth-5 cases. On the 48-case hard subfamily repeated across eight original seeds, Direct solved **240/384** and VI-2 solved **131/384**. Both policies share exactly the same source-trained 6,036-parameter recurrent model. AF-26 asks which inference decision changes that gap.

[PROTOCOL.json](PROTOCOL.json) was committed before any AF-26 counterfactual model evaluation, fixing all six arms, the original 109-proof hard gap, a 55-proof (half-gap) attribution threshold, and independent audit requirements.

## Closed-loop interventions

For every visited state, the original AF-25 planner computes all original Direct, VI-1 and VI-2 scores over the same host-verified two-step successor graph. The new factor selector reads **only the Direct and VI-2 score vectors** and the host's legal rule mask.

- **DD:** native Direct STOP choice, Direct legal-rule ranking; exact replay of original Direct.
- **DV:** native Direct STOP choice, VI-2 legal-rule ranking.
- **VD:** native VI-2 STOP choice, Direct legal-rule ranking.
- **VV:** native VI-2 STOP choice, VI-2 legal-rule ranking; exact replay of original VI-2.
- **ND:** continue whenever at least one legal rule remains, following Direct rule ranking.
- **NV:** continue whenever at least one legal rule remains, following VI-2 rule ranking.

STOP is a *binary native decision* from the selected source scorer's complete legal-rule/STOP vector. The rule comes from the independent registered ranking source. The two source scores are **never added together across logit scales**. Terminal goals and states with no legal rule are host-governed.

The four **crossed intervention arms** (DV, VD, ND, NV) follow genuinely fresh, closed-loop trajectories, rerunning the original frozen checkpoint at each newly reached state. DD and VV are **immutable historical controls**, whose entire original AF-25 proof episodes, score vectors, action order and host proof evidence are read from the original per-seed SHA-256-verified CI artifacts. The controls are numerically rescored from their saved per-node head outputs and fully independently reaudited, including genuine sampled checkpoint-forward replays. They are not represented as fresh neural rerolls.

This source-parity procedure was introduced after two blocked attempts: [CI #43864977372](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43864977372) found VI-2 completion-count divergence in seeds 103 and 131 under new joint batching, while [CI #43866180821](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43866180821) found another VI-2 divergence in seed 29 despite retaining original three-mode grouping. The original seed-29 archive records **263 decisions with a top-two score margin below 1e-5** among 2,003 Direct/VI-2 action steps. The historical, independently verified source trajectories are thus the sound baseline for the completed-factorial comparisons. The counterfactual arms still recompute neural states and scores after every chosen action, using a frozen network.

Every original control action and full neural node ledger is checked against the official original AF-25 seed archive (IDs and ZIP checksums pinned in source) and then independently replayed from public physical Horn incidence. Numeric sampled Torch replays admit only narrowly documented <2e-5 action-boundary ambiguities after full score-vector agreement within 1e-5; these are counted rather than called exact action replays. The scientific protocol, outcome thresholds and original AF-25 science are unchanged, and both blocked attempts remain visible.
## Original-source parity and primary endpoint

The original [AF-25 run #43829157457](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43829157457) supplies the exact source-bound aggregate `af25_results.json` (member SHA-256 `5624d2bfe913c870dada6be00a722eceb8157be26824a875d36d0723eec99163`), pinned to artifact ZIP ID `14208435623`. Every original AF-23 frozen-checkpoint ZIP must independently pass its original checksums and Torch/NumPy state-dict parity.

**Before the experimental attribution is interpreted**, DD and VV must reproduce each original AF-25 seed's Direct and VI-2 proof-completion count at **both depths**. The pooled totals and original hard-family counts (DD=240, VV=131) must match byte-for-byte source-bound aggregate fields.

For the 384 repeated hard positive depth-5 episodes:

```text
Direct gap                 G = solved(DD) - solved(VV) = 109
STOP-gate recovery          = solved(DV) - solved(VV)
Rule-rank recovery          = solved(VD) - solved(VV)
Interaction                 = solved(DD) - solved(DV) - solved(VD) + solved(VV)
Half-gap threshold          = 55 solved proofs
```

A descriptive `STOP_ONLY`, `RULE_ONLY`, `BOTH` or `NEITHER` outcome is assigned from the **two independent threshold comparisons**. Source or replay failure yields `INCONCLUSIVE`. This does **not** identify a unique additive mediation in an interactive policy, establish equivalent inference cost, or provide evidence of general-purpose theorem proving.

## Reported controls

In addition to the hard depth-5 endpoint, every arm reports all 80 provable and 112 unprovable AF-25 cases at both held-out depths, with separate 48-case hard and 32-case balanced families. **First-step STOP, later premature STOP, correct initial STOP on unprovable goals**, proof completion and model forward-node work are all recorded.

The never-STOP controls can trade fewer premature stops on provable cases for worse correct initial STOP on unprovable cases. They are diagnostic arms and cannot be interpreted as successful calibrated policies on that metric.

Each seed's factorial comparisons are paired by identical case ID and fixed original checkpoint, reporting discordant paired successes and failures. The repeated 48 graphs are shared across seeds; 384 responses are seeded model outcomes rather than 384 independent proof instances.

## Reproduce

Source-only:

```bash
python -m unittest discover -s experiments/activation_funnel_af26/tests -p 'test_*.py' -v
```

With Python 3.12, Torch 2.5.1 CPU, both original SHA-pinned seed ZIPs named `original/af23-branch-seed-17.zip` and `original/af25-seed-17.zip`, and the verified original AF-25 `af25_results.json`:

```bash
python -m experiments.activation_funnel_af26.evaluate \
  --seed 17 --original-zip-dir original --af25-seed-zip-dir original \
  --af25-aggregate af25_results.json --out out/af26-seed-17

python -m experiments.activation_funnel_af26.audit \
  --seed 17 --original-zip-dir original --af25-seed-zip-dir original \
  --af25-aggregate af25_results.json --output-dir out/af26-seed-17
```

Source-locked CI runs smoke, all eight full seed evaluations, independent physical and Torch audits, and an aggregate that verifies the registered attribution thresholds.

**All original AF-22 through AF-25 scientific verdicts remain unchanged.** Keep the new PR draft and unmerged for review.
