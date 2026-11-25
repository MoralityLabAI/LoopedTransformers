import random, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lbf19_tree_automata as T


def test_rewriting_commutativity():
    V = lambda i: ('V', i)
    rules = [(('B', 0, V(0), V(1)), ('B', 0, V(1), V(0)))]
    t = ('B', 0, ('L', 0), ('L', 1))
    assert T.succs(t, rules) == {('B', 0, ('L', 1), ('L', 0))}


def test_dag_automaton_language_exact():
    rng = random.Random(1)
    seeds = [T.rand_term(rng, rng.randint(5, 20)) for _ in range(5)]
    A = T.dag_automaton(seeds)
    A.prepare()
    assert all(A.accepts(t) for t in seeds)
    others = [T.rand_term(rng, 10) for _ in range(200)]
    assert all(A.accepts(t) == (t in seeds) for t in others)


def test_bare_var_rhs_uses_epsilon():
    V = lambda i: ('V', i)
    rules = [(('U', 0, V(0)), V(0))]   # s(x) -> x
    seed = ('U', 0, ('U', 0, ('L', 1)))
    A = T.dag_automaton([seed])
    cps = T.run_ta(A, rules, 2)
    B = cps[2][1]; B.prepare()
    assert B.accepts(('U', 0, ('L', 1))) and B.accepts(('L', 1))


def test_soundness_named_and_random_systems():
    for name, rules, seed in [('named', T.named_system(), 7)] + \
            [('r%d' % i, T.random_system(19300 + i, 'mixed'), 100 + i) for i in range(3)] + \
            [('p%d' % i, T.random_system(19100 + i, 'preserve'), 200 + i) for i in range(2)]:
        rng = random.Random(seed)
        seeds = [T.rand_term(rng, rng.randint(5, 15)) for _ in range(3)]
        layers, _ = T.ball_layers(seeds, rules, 2, cap=20000)
        A0 = T.dag_automaton(seeds)
        for widen in (None, 2, 1):
            try:
                cps = T.run_ta(A0, rules, 2, widen=widen, limit=20000, tlimit=30)
            except T.Blowup:
                continue
            for k in (1, 2):
                A = cps[k][1]; A.prepare()
                assert all(A.accepts(t) for t in layers[k]), (name, widen, k)


def test_widening_enlarges_language():
    rules = T.named_system()
    rng = random.Random(5)
    seeds = [T.rand_term(rng, 12) for _ in range(3)]
    A0 = T.dag_automaton(seeds)
    exact = T.run_ta(A0, rules, 1)[1][1]; exact.prepare()
    wide = T.run_ta(A0, rules, 1, widen=1)[1][1]; wide.prepare()
    probes = [T.rand_term(rng, rng.randint(3, 14)) for _ in range(300)]
    layers, _ = T.ball_layers(seeds, rules, 1)
    probes += list(layers[1])
    assert all(wide.accepts(t) for t in probes if exact.accepts(t))
    assert wide.n_active_states() <= exact.n_active_states()


def test_chain_embedding_and_soundness():
    r = T.chain_system(19500, ks=(1, 2))
    assert r['embedding_ok']
    for c in r['cells']:
        assert c['ta_pos_false_negatives'] == 0 and c['ta_dag_false_negatives'] == 0
        assert c['env_false_negatives'] == 0


def test_baseline_alpha_contains_ball():
    rules = T.named_system()
    rng = random.Random(3)
    seeds = [T.rand_term(rng, 10) for _ in range(3)]
    layers, _ = T.ball_layers(seeds, rules, 2)
    al = T.tree_alpha(layers[2])
    assert all(T.alpha_member(al, t) for t in layers[2])
