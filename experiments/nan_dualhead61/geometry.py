"""Find continuous witnesses and verify conservative vocabulary dual bounds."""
from fractions import Fraction

from experiments.nan_headnull60.optimize import plan as plan_row_null
from experiments.nan_sham56.minimax import optimize


WEIGHTS = (0., 1e-6, 4e-6, 1.6e-5, 6.4e-5, .000256, .001024, .004096, .016384, .065536, .262144)
STEPS = (.16, .08, .04, .02, .01)


def gram_of(atlas, np):
    matrix = atlas.astype(np.float64)
    return matrix.T @ matrix / len(matrix)


def positive_definite(matrix):
    """Exact symmetric LDL pivots using a rational Schur complement."""
    for i in range(len(matrix)):
        pivot = matrix[i][i]
        if pivot <= 0:
            return False
        for j in range(i+1, len(matrix)):
            for k in range(j, len(matrix)):
                matrix[k][j] -= matrix[k][i] * matrix[j][i] / pivot
    return True


def certificate(gram, rows, factors, max_entry, peak, rms):
    p, r = Fraction(str(peak)), Fraction(str(rms))
    weights = list(map(lambda v: Fraction(float(v)), factors))
    if any(v < 0 for v in weights):
        raise ValueError("Dual multipliers must be nonnegative")
    a = [[Fraction(float(entry)) for entry in row] for row in rows]
    n = len(gram)
    error = n * Fraction(float(max_entry))**2 / 2**32
    threshold = r*r + p*p * sum(weights) + error
    matrix = [[Fraction(float(gram[i,j])) + sum(w*v[i]*v[j] for w,v in zip(weights, a, strict=True))
               - (threshold if i==j else 0) for j in range(n)] for i in range(n)]
    return positive_definite(matrix)


def dual(atlas, gram, seeds, peak, cfg, guard, np):
    vectors = {}
    current = gram.copy()
    mass = 0.
    weights = {}
    number = cfg["coordinate_sweeps"] + cfg["extra_dual_rows"]
    for j in range(number):
        guard()
        if j >= cfg["coordinate_sweeps"]:
            _, e = np.linalg.eigh(current)
            token = int(np.argmax(abs(atlas @ e[:, 0].astype(np.float32))))
            local = [token]
        else:
            local = seeds
        for token in local:
            if token not in vectors:
                a = atlas[token].astype(np.float64)
                vectors[token] = a, np.outer(a,a)
                weights[token] = 0.
            _, block = vectors[token]
            old = weights[token]
            base = current-old*block
            rest = mass-old

            def value(x):
                return float(np.linalg.eigvalsh(base+x*block)[0] - peak**2*(rest+x))

            chosen = max((*WEIGHTS, old), key=value)
            weights[token] = chosen
            current = base+chosen*block
            mass = rest+chosen
    eigen, mode = np.linalg.eigh(current)
    effective = {i: w for i,w in weights.items() if w>0}
    return mode[:, 0], float(eigen[0]-peak**2*mass), effective


def primal(atlas, gram, starts, peak, lower, upper, iterations, guard, np):
    def metrics(q):
        q = q / np.linalg.norm(q)
        z = atlas @ q.astype(np.float32)
        ix = int(np.argmax(abs(z)))
        p = float(abs(z[ix]))
        rms = float(np.sqrt(max(float(q @ gram @ q), 1e-30)))
        constraints = (p/peak, rms/upper, lower/rms)
        return max(constraints), p, rms, q, int(np.argmax(constraints)), z, ix

    best = None
    for start in starts:
        current = metrics(start.astype(np.float64))
        for _ in range(iterations):
            guard()
            violation, _, rms, q, kind, z, i = current
            if violation <= 1.:
                break
            if kind == 0:
                # Use the signed top-row gradient, which does not cancel
                # when the worst positive and negative responses are close.
                gradient = np.sign(z[i]) * atlas[i].astype(np.float64) / peak
            else:
                gradient = gram @ q
                gradient *= (1/(rms*upper)) if kind==1 else (-lower/rms**3)
            tangent = gradient - float(gradient @ q)*q
            length = np.linalg.norm(tangent)
            if length <= 1e-14:
                break
            for step in STEPS:
                trial = metrics(q - step*tangent/length)
                if trial[0] < violation:
                    current = trial
                    break
            else:
                break
        if best is None or current[0] < best[0]:
            best = current
    return best[3], {"maximum_constraint_ratio": best[0], "predicted_peak": best[1], "predicted_rms": best[2]}


def solve(atlas, seed_tokens, target, cfg, guard, np):
    if atlas.dtype != np.float32 or not 1 <= len(atlas) < 2**18:
        raise ValueError("The Gram roundoff enclosure requires the pinned float32 atlas dimensions")
    gram = gram_of(atlas, np)
    p = cfg["peak_target"]
    r_low = max(0., target-cfg["rms_gap_limit"])
    r_high = target+cfg["rms_gap_limit"]
    q, bound, multipliers = dual(atlas, gram, seed_tokens, p, cfg, guard, np)
    lmodes = np.linalg.eigh(gram)[1]
    ordinary = optimize(atlas, np)[0]
    previous, _ = plan_row_null(atlas, target, {"maximum_outlier_constraints": 10, "predicted_peak_limit": .12}, np)
    unit, metrics = primal(atlas, gram, [q, lmodes[:,0], previous, ordinary], p, r_low, r_high,
                           cfg["primal_iterations"], guard, np)
    certified = certificate(gram, [atlas[i] for i in multipliers], list(multipliers.values()),
                             abs(atlas).max(), p, r_high) if bound > r_high**2 else False
    if certified and metrics["maximum_constraint_ratio"] < 1.-1e-6:
        raise ValueError("Continuous witness conflicts with the positive-definite obstruction")
    return unit, metrics | {"rms_lower": r_low, "rms_upper": r_high, "peak_limit": p,
                            "dual_numeric_lower_rms": float(np.sqrt(max(bound, 0.))),
                            "certified_continuous_infeasible": certified,
                            "dual_nonzero_rows": len(multipliers), "dual_witness_rows_and_weights": multipliers}


def verify(np):
    a = np.array([[1., 0.], [0., 2.]], dtype=np.float32)
    gram = gram_of(a, np)
    assert certificate(gram, [a[0]], [.5], 2., .1, .9)
    assert not certificate(gram, [a[0]], [0.], 2., .1, .9)
    q, data = primal(a, gram, [np.array([1., 0.])], 1.1, .5, .9, 4, lambda: None, np)
    assert data["maximum_constraint_ratio"] < 1. and np.array_equal(q, [1., 0.])
