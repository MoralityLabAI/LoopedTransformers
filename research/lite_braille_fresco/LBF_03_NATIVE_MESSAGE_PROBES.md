# LBF-03: frozen PPT-B message-probe / native-fresco receipt

**Date:** 2026-11-23  
**Evidence class:** native Lean source + independently replayed proof episodes, frozen checkpoint originally trained on synthetic fixture (two optimizer steps); scientifically ineligible for learned architecture claims.  
**Workflow:** [LBF native run 45179430465](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45179430465), completed successfully.  
**Artifact:** [lbf02-native-fresco](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45179430465/artifacts/14819559907) (SVG, encoded fresco records, observation manifests, frozen-model message-probe JSONL, native run and replay receipts).

## Registered mechanics

- Run native Lean 4.19 against two fresh generated Horn theorem roots with the already available AB/BA/auto8 stepper, then independently replay all six episodes.
- Freeze a Plan B model trained for two steps on synthetic proof fixture inputs. Re-forward public predecision input packets from neural AB/BA episodes, leaving source model and Lean decisions unchanged.
- Render four neural-arm observations, each with up to three recurrence passes per tactic decision, as strict `lbf.observation.v1` -> `lbf.encoded.v1` -> SVG.
- At each decision, re-run *the same frozen model* with previous-pass message modes `normal`, `zero`, `roll`, matching recurrent depth, candidate menu, parameter tensors, predecision packet, input view, and remaining budget.
- Record logit displacement, coda-state RMS displacement, and whether the maximum-logit candidate changes. First pass must be invariant because previous-pass memory starts at zero. No messages contain future proof outcomes.

## CI outcome

| Check | Outcome |
|---|---|
| Native Lean episode stepper | Passed |
| Independent native replay | Passed |
| Manifest / replay SHA checks | Passed |
| Frozen checkpoint | SHA-256 `a3b83460717016495cb7e9ccb27cf6514fa24a8659e331cddec94bdf0558936e` |
| LBF neural-arm episodes | 4 |
| LBF message-probe rows | 84 |
| Message modes | zero and roll relative to normal |
| First-pass zero-bank invariance | Passed, all within 1e-5 |
| Peak absolute candidate-logit difference | **0.0009042918682098389** |
| Renderer Lean calls | 0 |
| Private proof labels exported | 0 |
| Native source attestation | Checked |
| Scientific eligibility | False |

The test catches a real, bounded *model-output* sensitivity to cross-pass message transport. The effect was **small** on this extremely weak synthetic-fixture-trained checkpoint. It supports instrument correctness; it establishes neither improved candidate selection nor end-to-end Lean proof completion.

## Interpretation and next gate

LBF exposes nominal eight-slot **coda outputs** and candidate policy projections. A real cross-pass message intervention was run in the underlying model, but the fresco's `message` row is still a nominal slot, not an independent recording of every gated middle-block read/write operation. To make an explanatory claim, add explicit saved previous-pass and middle A/B gate hooks, test donor-message and matched random-norm sham patches, and measure verified native closure at fixed attempts and forwards.

Before promotion, train a truly source-verified PPT-B checkpoint with lineage-disjoint Lean proof families and a prespecified calibration competence gate. Then evaluate multiple seeds with root-paired uncertainty and a prospective sham. Historical residual-transfer policy weights and PPT-B remain incompatible models; their scores are kept separate.

**Run artifact SHA-256:** `08b3937ab60a2ce9ea12d9e8f215409d07f592209c41986ee1a9ef1a1ef6dacf`.
