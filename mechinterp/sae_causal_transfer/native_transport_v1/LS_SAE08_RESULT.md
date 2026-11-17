# LS-SAE08 completed result: 52/54-hole native Sudoku causal transfer

**Status: registered FAIL.** This report was written after the preregistered LS-SAE08 full run finished; it is a post-result report, not a protocol amendment.

## Immutable source and evidence

- Protocol frozen at `6b38c3dcf65dbc3a812e0f7c63b79dc14c6893fe`, independently first-admitted 52/54-hole roster with 8 held-out puzzles per family, original checkpoint and LS-SAE04 pretrained source SAE/PCA SHA locks.
- Full test [Actions run 43851165980](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43851165980), [evidence artifact 14220373987](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43851165980/artifacts/14220373987); original full.json SHA256 `69c4cd1a2e09760ed2d463e598550bdd1da42fb1bc0ad83a8a95825c47bea7f9`. Source, pilot and full jobs finished successfully.
- Independent stdlib audit checked all original first-admission rosters, checkpoint/fit hash declarations, frozen four-update geometry transcripts, ACT3/6/12 saved clean and signed logits, exact Sudoku digit correctness and originally registered 2000-resample puzzle-cluster bootstrap. The independent audit does not re-run the model or certify floating-point intervals.

## Precommitted primary endpoint

For each held-out puzzle `D = 0.5 * [F(STATE_POWER,+8) + F(STATE_POWER,-8)] - 0.5 * [F(RANDOM,+8) + F(RANDOM,-8)]` where F is the number of ACT6 originally blank-cell predicted digits changed from clean. Each 8-puzzle family required at least 6 puzzles with `D >= 1` and a strictly positive lower bootstrap-95% bound.

| Full cohort | n | Clean ACT6 correct / blanks | D per puzzle | Puzzles with D >= 1 | Bootstrap 95% CI | Original gate |
| --- | ---: | ---: | --- | ---: | --- | --- |
| 52-hole | 8 | 416/416 | [0,0,0,0,0,0,0,0] | 0/8 | [0,0] | FAIL |
| 54-hole | 8 | 432/432 | [0,0,0,0,0,0,0,0] | 0/8 | [0,0] | FAIL |

Every full-test puzzle had zero STATE_POWER blank-digit flips at both +8.00 and -8.00; matched RANDOM also had zero. The observed paired correctness effect was zero in both families. Engineering pilots 2+2 were excluded from the preregistered gate.

The model-blind visibility-only difficulty audit found that 10/16 full-test boards solve by naked+hidden singles alone. The other six require a stronger solver than that propagation baseline. Every board, including the six, was exactly solved by the pretrained network at ACT6.

## Inference and next measurement

These results establish native-output robustness to the specified single-token physical perturbations on the tested 16 puzzles. They leave open the dynamical explanation: H-state contraction, weak transport into the digit-head rowspace, or a large enough clean decision margin can each coexist with zero flips. LS-SAE09 is separately registered to measure the actual ACT3/6/12 hidden-state transport and margin pressure, using the original physical directions, rosters and exact model, and refusing to reoptimize them on the solved cohort.

An [external same-checkpoint study](https://arxiv.org/abs/2609.26487) already examines full-state Jacobians and perturbation absorption. LS-SAE08 and LS-SAE09 should credit that prior art and limit novelty to the single-site, SAE/logit equal-budget comparisons and finite Sudoku-specific readout diagnostics.

## Independent unrelated CI failure

The LS-SAE03 `pinned-checkpoint-pilot` job on [43851165837](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43851165837) fails in `native_jvp_v1/fit_lock.py` with `ValueError: full source LS-SAE02 calibration fit fingerprint mismatched`. No checksum is relaxed here.

## Post-result source-locked secondary findings (descriptive)

The original 64,074,609-byte `full.json` has all 16 clean boards solved **exactly** by ACT6, including original givens consistency, and all 16 still exact at ACT12. At ACT3, the 52-hole family had 8/8 exact solves; the 54-hole family had 7/8. The remaining 54-hole board, seed `202628040300`, had 30/54 originally blank cells correct at ACT3 and all 54 correct by ACT6.

At ACT3, STATE_POWER and LOGIT_POWER each repaired a symmetric mean of **3.0 blank digits** at physical dose 2.00 on seed `202628040300`. The SAE 8.00 perturbation repaired a symmetric mean of **13.5** there. On another 54-hole board, seed `202628040600`, STATE_POWER at ACT3 dose 8.00 produced a symmetric **5.0** additional errors, and LOGIT_POWER **6.0**. These retrospective comparisons were excluded from the ACT6 registered gate.

On the previously difficult `202628040300` board, the SAE physical dose 0.20 caused an ACT6 symmetric **10.0 blank-cell corruptions**, all from its negative-sign arm (20 digit flips), whereas SAE at doses 2.00 and 8.00 caused zero ACT6 flips. This nonmonotone finite-dose behavior is descriptive and should be investigated as such.

Across all 16 original source directions, absolute cosine similarity between STATE_POWER and LOGIT_POWER was **0.95047–0.99999**. Their strong near-alignment is evidence that the fixed search budget selected similar input directions on this checkpoint; it limits how independently the two directions can be interpreted. ACT6 clean blank-cell top-two margins had a **minimum of 59.4558** logits on the 54-hole family and **62.7680** on the 52-hole family, offering a concrete reason for high flip resistance.

The original ACT1 edited source `z_H` norm is approximately **22.6274** in all 16 held-out cases, so physical doses 0.20, 2.00 and 8.00 correspond to about 0.9%, 8.8% and 35.4% of its norm. These source and margin numbers come from the frozen original LS-SAE08 receipts. LS-SAE09 measures their subsequent hidden-state transport rather than selecting an intervention sign or changing the original registered scientific outcome.
