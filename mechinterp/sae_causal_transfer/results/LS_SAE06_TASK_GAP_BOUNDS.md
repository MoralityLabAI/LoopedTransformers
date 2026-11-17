# LS-SAE06: exact bounds connecting hidden-state gain to discrete Sudoku prediction changes

**Mathematical companion, November 16, 2026.** These finite-dimensional inequalities were written after the prospective [LS-SAE06 protocol](../native_task_transfer_v1/PROTOCOL.md) was committed. They are classical exact bounds used to interpret the full 81×9 logit receipts; they are not empirical results or Lean kernel proofs, and they cannot change the preregistered task-prediction gate.

## Winner-gap barrier under finite centered digit-logit edits

For cell i in an 81-cell Sudoku, let its clean nine-logit vector be L_i∈R^9. Let w_i be a clean top-logit digit with **strict** clean runner gap

    Δ_i = L_{i,w_i} − max_{d ≠ w_i} L_{i,d} > 0.

Let E_i be any finite change of that cell's nine logits from a physical hidden-state intervention. Its centered value is E_i°=E_i−(∑_d E_{i,d}/9)1; this removes the per-cell additive gauge that leaves every argmax prediction invariant.

**Proposition 1, exact finite winner-gap barrier.** If a different digit v_i becomes top logit after the edit (with ties treated as possible changes), then

    Δ_i ≤ E_{i,v_i} − E_{i,w_i}
        ≤ sqrt(2) ||E_i°||_2.

Proof: Since v_i beats the original winner, L_iv+E_iv≥L_iw+E_iw. Thus E_iv−E_iw≥L_iw−L_iv≥Δ_i. The difference of two coordinates of E_i° is the Euclidean inner product of E_i° with e_v−e_w, whose norm is sqrt(2). Cauchy–Schwarz gives the stated result. The bound is sharp in a two-coordinate perturbation with E_iv=+a and E_iw=−a and clean gap 2a, with all other logits sufficiently low.

**Corollary 1, global budget of robust-cell flips.** For any γ>0, let B_γ be the originally blank Sudoku cells with clean winner gap at least γ, and let N_γ be how many change predicted digit after the edit. Then

    N_γ ≤ (2/γ²) Σ_{i∈B_γ} ||E_i°||²
        ≤ (2/γ²) ||center(E)||_F².

Every changed gap≥γ cell consumes at least γ²/2 squared centered-logit energy. Summing gives the bound. Therefore a large full **hidden-state** Jacobian gain is insufficient to force *any* discrete Sudoku prediction change; the perturbation must reach a sufficient number of small-gap output cells with enough centered-logit energy.

## Sign symmetry and verified correctness

Let P_i be the clean predicted digit and T_i the unique correct digit. If edited prediction P_i' differs in whether it is correct, then necessarily P_i'≠P_i. Hence

    |Σ_{i∈blank}[P_i'=T_i] − Σ_{i∈blank}[P_i=T_i]|
    ≤ #{i∈blank:P_i'≠P_i}.

For physical plus/minus edits ±r u, define the **registered** symmetric blank prediction flip score F(u)=½(Flips_+ + Flips_−). The signed symmetric mean correctness gain C(u)=½(Correct_+ + Correct_−)−Correct_clean satisfies the exact finite inequality

    |C(u)| ≤ F(u).

A positive flip score therefore demonstrates causal effect on **discrete predictions**. Accuracy improvement requires an additional separate check on the true solution. This is why LS-SAE06 registers the flip contrast as its primary test and reports blank-cell correctness gain and verified Sudoku validity independently.

For any 9×9 grid predicted by argmax, a constructive witness that it is a valid completion consists of all 81 digits in {1,...,9}, equality to the original nonzero clues, and each of 27 Sudoku row/column/3×3-box units containing digits {1,...,9}. If the original puzzle has exactly one solution, this validation implies equality with the registered target solution. The LS-SAE06 audit checks those integer conditions and equality directly, independently of floating-point margin estimates.

## First-order task readout and finite-error budget

Let f:R^512→R^(81×9) be the native recurrent logit map through two additional ACT steps after the frozen source site; let C denote the per-cell orthogonal centering projection. For source edit h→h+r u, its derivative is J=D(Cf)(h).

Suppose an independently established bound on the operator norm of the Hessian of Cf on the segment h+tru, 0≤t≤1, is M, with ||u||=1. The integral Taylor remainder gives

    ||C(f(h+r u)−f(h))−rJu||_F ≤ Mr²/2.

Consequently at any originally blank cell i, a sufficient condition for its original predicted digit to remain unchanged is

    sqrt(2) * ( r||(Ju)_i||_2 + Mr²/2 ) < Δ_i.

This is a **conditional** theorem requiring a genuine Hessian bound M, not one supplied by LS-SAE05 or LS-SAE06. The actual finite ±0.05/±0.20 source interventions and centered-logit JVP secants measure the local discrepancy; they do not yield a certified global curvature bound from one finite difference.

## Why the two power baselines answer separate questions

The original LS-SAE05 structural power search iterates the positive operator J_stateᵀ J_state, where J_state:R^512→R^(97×512) is the full hidden-state transport Jacobian after two ACT steps. LS-SAE06's task-logit power search iterates J_logitᵀ J_logit, where J_logit:R^512→R^(81×9) is the *centered* all-cell predicted digit-logit Jacobian at the same horizon. Each experiment computes four forward JVP and four reverse VJP updates from an independently seeded unit source direction, keeping task answers entirely outside the selection loop.

Because a head-map can annihilate or attenuate a high-norm hidden-state output direction, the ordering of gains under these two quadratic forms need not coincide. A large ||J_state u|| can coexist with a small ||J_logit u||; a large centered-logit norm can still fail to cross any clean winner gap. The registered output-array and integer-Sudoku receipts explicitly distinguish all three levels: physical state transmission, gauge-invariant digit-logit perturbation and changed/verified predictions.

**Scope:** exact algebra under stipulated numerical matrices and a conditional Taylor bound. Native FP32 Jacobians, predicted logits, source SAE selections and finite hidden interventions are empirical and must be verified through the separate source-locked CI receipts.
