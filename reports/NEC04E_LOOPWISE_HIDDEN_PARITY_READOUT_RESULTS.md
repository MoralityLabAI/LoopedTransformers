# NEC-04E: five-bit parity has no qualified source-linear readout in original 12-step Looper memory

**Completed 22 November 2026.** Evidence class: new hidden-state inference and source-fitted linear readout on the **previously trained** NEC04C six CALIBRATION world roots6–11, with zero new original task-model optimizer updates, new developmental/confirmation WORLD scoring, or neuron-community/SAE causality. Issue [#302](https://github.com/MoralityLabAI/LoopedTransformers/issues/302). Prospective design/controls are preserved in [NEC04E_PROTOCOL.md](../mechinterp/neuron_communities/NEC04E_PROTOCOL.md) and [NEC04E_PROBE_LOCK.json](../mechinterp/neuron_communities/NEC04E_PROBE_LOCK.json).

## Hosted evidence and correction log

The first pre-inference locked experiment [Actions run 44995988351](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44995988351) independently rechecked the entire prior NEC04C source/calibration-world validation history and **all 24 original model+optimizer checkpoint byte SHA256s**, and passed 10 preflight tests. First readout workers recomputed all 12 original recurrent states and saved logits; the separately coded linear-probe auditor then failed because numerical variations near a linear regression score of 0.5 changed a few binary predictions. **No result from this first run is accepted as independently replayed.**

The correction froze a **1e−8 tie tolerance above the unchanged 0.5 score cutoff**, treating near-half numerical ties as class 0 in both the primary and independently coded probe scorers. This was a source-replay stability fix after the failed independent audit, and is recorded in the revised lock, protocol and a new exact balanced-null test. The previous six calibration roots, source/owned split, 24 native checkpoints, initial weights, ridge lambda, label sources, control set and 70%/+10pp/four-world diagnostic gate were unchanged. The original failed run and correction are preserved in the PR discussion.

The corrected experiment ran from code/source commit **8d85de239abef9bba698cdd0416946e31ded3734**: [**successful GitHub Actions run 44996254773**](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44996254773). It passed **11/11 source-only control/lock tests**, the independent old-world archive and native checkpoint preflight, **6/6 root workers** and the fixed six-root readout aggregate. All **24/24 native checkpoints × all 12 original recurrent states** were independently re-executed, their 12th memory state and model-head logits checked against the previously scored NEC04C output, and **4,912 source-fitted ridge readouts** independently recomputed for every relevant visit, target and control. Independent readout checks used separately constructed prefix/future/teacher labels and compared each saved validation prediction SHA256, source/validation class counts and balanced accuracy.

Original model data came only from NEC04C training [run 44993859839](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44993859839), source artifact **14742886436** and native root6–11 checkpoint artifacts **14742124547,14743106877,14743221555,14742897690,14743391255,14742099591**. Original NEC04A source SHA256 **82c80643d3c99871913f5228f1a1cde1eec4484787c1784872b508349793e9e0** and 64-root registry SHA256 **88bf58d32fdf15a1270986c238afb3def1a7e6371d3eee64c4edd264c9cf041e** were preserved.

**Accepted final aggregate artifact:** ID **14744390471**, named **nec04e-six-root-independent-twelve-visit-hidden-parity-probe-aggregate**, ZIP SHA256 **5532e3d8e19b644bbb44b1e7cb82f71effe873a0206c2348941222d75fa0db6c**. Six individually audited root readouts: root6 **14743644990**, root7 **14742944680**, root8 **14743906328**, root9 **14743853271**, root10 **14744345617**, root11 **14744370543**. Original pre-inference lock artifact **14743524999**. The JSON preserves each original root/arm/240/900-step checkpoint/visit/label/control result, source and validation score, model and hidden-state SHA, target class-balance and independent replay receipt.

## Frozen task and readout

Each original 32D Looper consumes the same 12 chronological input bits with the original recurrent block. We inspected *exactly* the 12 successive 32D memory states, h1..h12, without modifying native weights or applying an intervention. The original 80 complete 32-focal-mask source context blocks per root (2,560 cases) supplied a source-only ridge fit for every fixed probe; the 24 original owned validation contexts per *same calibration WORLD root* (768 cases) supplied the prior allocation for scoring. Readout fits and scores used only the original context MUX=1 **parity-selector** blocks from those panels, with 50% class balance in every complete 32-focal-mask context.

Fixed readouts: trained hidden state, the exact same-seed random-initialized Looper hidden state, linear raw **already-seen** input bits, and hidden state trained on SHA-shuffled **source** labels. Fixed ridge lambda 1, source-only feature standardization (std floor 1e−4), unpenalized intercept and class cutoff 0.5 with the documented near-half tie policy.

The prospective diagnostic was **five-focal prefix XOR** at the *first visit when all five focal bits have arrived*, determined only from each original root's focal position order. The primary checkpoint/arm is the original **900-step parity_curriculum** Looper, with at least four original roots scoring **≥70%** validation and **≥10 percentage points** above their stronger random/visible-bit baseline, plus 70% and +10pp six-root macro conditions. Independent statistical unit = one original world root, n=6.

## Primary 900-step parity readout

Six original root validation accuracy values, source-trained linear readout of *unadjusted five-focal XOR* at the first fifth-focal-complete recurrent state:

| Original calibration world | First fifth-focal complete visit | Curriculum trained hidden | Same-seed random hidden | Raw visible bits | Source-shuffled hidden |
|---|---:|---:|---:|---:|---:|
| cal-006 | 12 | **50.67%** | 51.12% | 50.00% | 50.45% |
| cal-007 | 8 | **56.53%** | 51.70% | 50.00% | 49.72% |
| cal-008 | 11 | **50.28%** | 50.28% | 50.00% | 51.14% |
| cal-009 | 11 | **49.22%** | 49.74% | 50.00% | 51.56% |
| cal-010 | 11 | **48.96%** | 49.65% | 50.00% | 50.35% |
| cal-011 | 12 | **50.00%** | 49.38% | 50.00% | 48.75% |
| **Six-root macro** | Root-specific | **50.94%** | **50.31%** | **50.00%** | **50.33%** |

Mean hidden advantage over the **stronger random/raw baseline within each root** was only **+0.43 percentage points** (across-root descriptive SD **2.20 pp**, SE **0.90 pp**). Root cal-007 showed the highest curriculum readout at **56.53%**, still well short of the 70% requirement. **Zero of six roots qualified**. The accepted aggregate's machine decision is **NO_QUALIFIED_HIDDEN_PREFIX5_LINEAR_READOUT_SIGNAL**.

At the original 900-step vanilla checkpoint, the corresponding five-focal source-only trained hidden readout was **51.36%** six-root macro (random 50.31%, raw 50%, shuffled 50.14%), with **0/6 qualified roots**. Earlier 240-step corresponding trained hidden macro values were **50.70% vanilla** and **49.25% curriculum**, with zero qualified roots. The original 900-step curriculum probe also scored only **51.56% on the training-source parity contexts**, closely matching the weak **50.94%** calibration-world owned-validation result. The observed failure is already present in the source readout under this frozen linear probe.

## Lower-order and future-information controls

The lower-order positive controls behaved differently from the full five-focal parity target:

| Frozen control at its own first eligible recurrent visit | Vanilla 900-step | Curriculum 900-step |
|---|---:|---:|
| First focal bit, linear readout from native memory | **100.00%** | **100.00%** |
| XOR of first two observed focal bits, linear memory readout | **90.42%** | **95.00%** |
| Original final composed parity teacher label at visit 12 | **49.67%** | **50.45%** |
| XOR of last still-unseen focal input bit, future-information control | **50.00%** | **50.00%** |

The curriculum two-bit parity positive control was perfect in five original roots and **70.00%** on cal-011, giving 95.00% macro. This supports recoverable *low-order* XOR in the model's linear recurrent memory while exposing a sharp loss of recoverability at five focal inputs for the specific source-trained linear readout. It does not identify the exact visit where higher-order parity is first lost, although all 12 original visits and progression cases are present in the aggregate artifact. The fixed future-bit negative control remains exactly at chance, consistent with the original complete focal-mask closure and chronological input restriction. The final original composed parity teacher readout at visit 12 remains near chance.

These are calibration-only linear readout measurements. Nonlinear features or parity representations distributed across hidden units could remain detectable by different, independently qualified probes. Comparing models trained on the same Boolean worlds at two checkpoint budgets and multiple probes does not yield 24 independent experimental units. Original world roots, n=6, are the scientific unit.

## Consequence and protected state

NEC04B's native model-headroom gate failed; NEC04C's equal-budget source-parity curriculum failed the preregistered class-level parity improvement gate; NEC04D measured very small fifth-order final-output logit power; the present NEC04E finds **unqualified source-linear recovery of complete five-bit parity from intermediate recurrent memory despite strong one- and two-bit controls**. Together, these calibration-root observations motivate prospective *constructive intermediate XOR-state supervision* or a separately trained nonlinear hidden-state probe that can distinguish an optimization bottleneck from limits of the linear readout.

A future native-model training protocol would need entirely **untrained original NEC04A CALIBRATION roots 12–17**, predeclare source-only intermediate targets, matched native compute and initial weights, validation gates and independent replay before model training. An equivalent-budget, fixed-control design and explicit floor are necessary to preserve statistical meaning. No new model training or causal mediation is asserted by this report.

**Protection:** all 24 original NEC04A developmental WORLD roots and 16 confirmation WORLD roots, 24 reserved context blocks within every calibration world, old NEC03Z source groups 56–63, old NEC03P recruited-world causal-sham results, earlier embargoed and confirmation groups remain sealed. Original NEC03P's 69/88 source-qualified recruited worlds remain at **zero independently measured causal-sham outcomes**. No SAE fitting, functional/physical sham, or neuron-community necessity was scored in NEC04E. The original requirement for at least 59 eligible independent causal-sham world outcomes and a physically/functionally inert null remains deferred. **FSI-01: REDESIGN_OR_PILOT_ONLY.**
