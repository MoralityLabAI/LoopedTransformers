# mechinterp

Mechanistic-interpretability work on looped (weight-tied, recurrent-depth) transformers.

| folder | what it is |
|---|---|
| [loopscope01/](loopscope01/) | LOOPSCOPE-01: a fragility-vs-specificity instrument (TCA) for looped models, and a preregistered series (Q1–Q11) asking which loop-level statistics predict length extrapolation in toy loopers |
| [loopscope02/](loopscope02/) | LOOPSCOPE-02: pretrained 7M Attention-TRM vs. looped 3B and untied 4B LMs on stratified 9x9 Sudoku (`P1`–`P5`) |
| [loopscope03/](loopscope03/) | LOOPSCOPE-03: three-ledger iteration pricing ($S_t = V_t \oplus C_t \oplus R_t$), conditional refill, and enforced path independence |
| [bpd/](bpd/) | **BPD (Behavioral Parameter Decomposition)**: parameter decomposition for weight-tied computation (generalizing Goodfire's VPD to recurrent loops, reducing to VPD at $T = 1$) |

Related, elsewhere in the repo: `papers/rmp_looped_mechinterp/` (paper) and `reports/technical_note_looped_mechinterp.md`.
