# NRF-02D: fully state-dependent soft routing with a global certificate

A worked extension of the graph-stability gym, paired with MathCan NRF-02D (PR #46). The new mathematical result uses OpenAI family #328 to obtain a barycentric nonexpansive network equilibrium in reflexive Banach spaces, under that paper's asserted theorem. The Crouzeix estimate (family #325) supplies nonnormal polynomial-update Lipschitz rates for a regularized network.

A fixed mixture of two loopers with certified gains 3/2 and 1/4, chosen with probabilities 1/4 and 3/4, has a global Lipschitz bound 9/16 even though the first branch alone can expand.

A state-dependent soft gate pi_high(x)=1/4+x[0]/16 has range [3/16,5/16] in C=[-1,1]^2. Its two-component routing weights obey ||pi(x)-pi(y)||_1<=||x-y||/8. As diam(C)=2sqrt(2)<=3, the contraction bound including gate variation is

    (5/16)*(3/2) + (11/16)*(1/4) + (3/2)*(1/8)
        = 53/64 < 1.

By the Banach contraction theorem, the resulting soft operator has a unique globally attracting fixed point, with a posteriori residual-distance certificates. The exact rational implementation uses the same Jordan-polynomial updates as NRF-01, with coordinatewise metric projection and the varying affine gate.

Reproduce with:

    python -m unittest tests.test_soft_numerical_range -v
    python -m research_gym.scripts.bench_soft_numerical_range

Six tests include 625 exact squared-Lipschitz state-pair checks, static-mixture comparisons, eight-step trajectories and a posteriori radii. The receipt deliberately uses five steps because repeated nonlinear rational iteration causes rapid growth of numerators and denominators.

This result is for the averaged **soft mixture map**. It does not certify independent stochastic branch samples or hard switching across a discontinuous guard. It remains an algebraic toy with no tested neural model weights. Prior art on positive-matrix contraction and graph switching is acknowledged in the MathCan proof.
