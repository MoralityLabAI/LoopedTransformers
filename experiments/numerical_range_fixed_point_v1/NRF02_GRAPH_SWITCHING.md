# NRF-02: graph-constrained numerical-range loop certification

November 15, 2026. Follow-up to NRF-01 in PR #56, based on OpenAI math families 325 (complete Crouzeix) and 328 (nonexpansive fixed points). Pure-math proof and independent oracle: patrickdugan/MathCan PR #46, research/compactification/GRAPH_DIRECTED_NUMERICAL_RANGE.md.

## Mathematical contract

For an edge e:u->v, an externally certified polynomial norm bound q_e gives r_e=1-eta_e+2 eta_e q_e. This module uses exact Jordan-disk values and box-preserving orthogonal backbone gates. With rational 0<alpha<1, seek w_v>0 satisfying r_e*w_u <= alpha*w_v. An exact max-product Bellman-Ford procedure constructs the weights if all directed cycles meet the matching product constraint. For any *shared admissible route*, the accumulated rate is <= alpha^t*w_last/w_first. A single edge may have r_e>1.

For state-dependent routing, a graph certificate alone does not guarantee continuity. At every nominal guard s(x)=normal dot x - threshold, require (s(x))^2 > ||normal||^2*(error_radius)^2. When this strict separation holds along the nominal trace, every disturbance-limited neighboring trajectory keeps the same route; the exact per-step error bound grows by epsilon_next=r_e*epsilon+delta. A tie or inadequate margin is rejected.

## Reproduce

    python -m unittest tests.test_graph_numerical_range -v
    python -m research_gym.scripts.bench_graph_numerical_range

The dependency-free gym emits a JSON receipt with alpha=3/4, vertex weights idle=1, probe=2, audit=4/3 and six edge rates. It checks 256 two-node cycle configurations, a bad positive cycle, acyclic expansion, path certificates through depth seven, and strict guard failures. A 12-step nominal trace and perturbed replay select matching edges. The opposite side of the idle guard produces a one-step squared amplification of exactly 20527009/144; therefore policy switching invalidates an unchecked global Lipschitz claim.

## Evidence boundary

No trained transformer, neural logits, or real control-harness transitions are evaluated. The benchmark proves only exact rational toy claims under specified maps, guards and disturbances. The general matrix polynomial case uses the claimed complete Crouzeix bound as an input. Its two-by-two Jordan specialization has a direct independent proof in the MathCan note. Graph-cycle potentials and graph-directed dimension methods have prior literature: Mauldin–Williams (1988) and path-complete Lyapunov research. No priority claim is made.

For native LoopScope measurements, first export actual mode transition edges, guarded route scores and certified upper bounds on the operator numerical ranges. Test the policy-margin rejection frequency before interpreting these certificates as stability signals.
