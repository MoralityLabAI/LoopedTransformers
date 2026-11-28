# AUX-V2-MMS-03: native donor swaps, TRM input adapter, competence screening

Parent: PR #400, merged c25fe023b40eb4a802f4bb733f777d6d68067c88. All four final engineering checks passed. No performance claims.

## Actual learned factor interventions

The new native causal evaluator loads the exact frozen LOOP-T teacher and its learned V/U factors from the pinned BPDTrainer fit. It invokes BPDModel.intervene(mode="swap_acts") at the first divergent symbolic operation's visit. All non-answer positions retain anchor reads. Selected factor indices are determined by product norm ||v_c|| ||u_c||, invariant under reciprocal scaling; no validation or locked labels are used for selection. The independent additive implementation checks native API parity on every evaluated pair.

Intervention on selected components is exactly (x_donor-x_anchor) V_selected U_selected. Two controls are matched to the same physical write norm: SVD-subspace projected full-write difference, and random direction within the selected learned output span. Sphere rescaling is deliberately disabled for this additive engineering experiment. This avoids conflating direction with dose; a pass-neutral sphere-rescaled experiment must be separate. The default tests skip doses <=1e-7 and unmatchable directions. A rank-deficient native span is a hard failure. Pass calibration tolerance is carried in the receipt, not used to claim tolerance-normalized significance.

The same-trace control visit is sampled cyclically from changed pairs at the same depth. It is a depth-matched control only; edit-type and dose-distribution matching remain needed for a definitive claim. All model inputs are original token sequences. The scorer measures final donor-target recovery, local symbolic donor state, and clean target preservation for same-trace controls, stratifying by depth. The primary causal cohort requires both clean inputs correct and matched physical dose; the denominator and nonadmitted rows are recorded. No competent pairs means inconclusive. The frozen teacher's factor provenance precludes silently applying old factors to updated MMS weights.

## Actual attention-TRM block

AuxTokenTRM reuses LOOPSCOPE-02's TRMBlock (RoPE, attention, SwiGLU, RMS norm). It adapts the input vocabulary to 2048 RMP tokens, omits Sudoku-specific puzzle embeddings, uses separate z_L/z_H states, and shares one physical block across low updates and high commits. It trains from scratch with full BPTT and tied answer lens. It is a new token-input adaptation, not a pretrained Sudoku TRM evaluation or an exact reproduction of the full ACT model. Low states remain accessible; high commits receive trace supervision. No adaptive halting.

At one low cycle and four high commits, there are eight physical block applications per input, matching LOOP-T's call count. Four commits cannot represent all eight symbolic steps one per commit; final CE still trains the terminal answer. Compare depth <=4 and >4 separately. An eight-commit TRM uses 16 applications and must be compared separately to a 16-visit LOOP-T. TRM uses bidirectional attention whereas LOOP-T uses causal attention. Block FLOPs, parameter counts and inference latency must be measured; equal calls alone do not establish FLOP equality.

## Development-only screening

Two independent seeds 1,2 per family, fixed 256 optimizer updates, batch 16, no adaptive stop, identical development dataset seed 2026112602, train quota 64 and validation quota 8. Four initial arms: LOOP-T final, repeated_final, symbolic_trace, and four-commit attention TRM symbolic_trace. Each root shares its data and seed. All evaluation is validation-only. Report depth-conditioned macro recall and masked node exact versus calibration constant; compare the three LOOP-T modes first. This small screening is not the preregistered eight-root confirmation. If competence remains poor, inspect depth-one arithmetic and optimize curriculum before launching the 128-cell suite. No locked predictions. Native factor causal engineering smoke uses a separate synthetic seed and a tiny teacher; no competence claim.

## Run

    python -m research_gym.interp.aux_gym_v2_native_causal --bpd-checkout /tmp/bpd-native --run /tmp/teacher --factors /tmp/native-factors --pairs /tmp/aux-v2-pairs --split validation --out /tmp/aux-native-causal --rank 4
    python -m research_gym.interp.aux_gym_v2_trm --data /tmp/aux-v2 --out /tmp/aux-trm --family masked_pointer_chase --mode symbolic_trace --seed 1 --steps 256 --batch 16 --commits 4

Future: fresh natural-prevalence evaluation, contamination checks against V1, robust root-level confidence intervals, measured FLOPs, learned-factor refresh with identity alignment, native BPD fitting to low/high TRM roles, and a larger paired MMS study contingent on competence.
