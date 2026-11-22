"""Constrained Rayleigh search with explicit worst-vocabulary-row exclusions."""
from math import sqrt

from experiments.nan_sham56.minimax import optimize


def plan(matrix, target_rms, cfg, np):
    raw = matrix.astype(np.float64)
    gram = raw.T @ raw / len(raw)
    del raw
    dimension = gram.shape[0]
    ambient, _, _ = optimize(matrix, np)
    directions, excluded = [], []
    limit = cfg["predicted_peak_limit"]
    n_steps = cfg["maximum_outlier_constraints"]

    def evaluate(unit):
        unit = unit / np.linalg.norm(unit)
        fx = matrix @ unit.astype(np.float32)
        biggest = int(np.argmax(abs(fx)))
        value = sqrt(max(0., float(unit @ gram @ unit)))
        return unit, float(abs(fx[biggest])), value, biggest

    directional = np.eye(dimension)
    for j in range(n_steps+1):
        energy, modes = np.linalg.eigh(directional.T @ gram @ directional)
        if energy[0] < -1e-12:
            raise ValueError("Projected derivative Gram is not PSD")
        vector = directional @ modes[:, 0]
        chosen = evaluate(vector)
        directions.append(chosen)
        if chosen[1] <= limit or j == n_steps:
            break
        token = chosen[3]
        if token in excluded:
            raise ValueError("An already-nulled row remained the highest violation")
        plate = matrix[token].astype(np.float64)
        projected = directional.T @ plate
        if np.linalg.norm(projected) < 1e-12:
            raise ValueError("Missing gradient for the predicted peak")
        _, _, right = np.linalg.svd(projected[None, :], full_matrices=True)
        directional = directional @ right[1:].T
        excluded.append(token)
    directions.append(evaluate(ambient.astype(np.float64)))
    if chosen[1] <= limit:
        end = ambient.astype(np.float64)
        if float(end @ chosen[0]) < 0.:
            end = -end
        directions.extend(evaluate((1-t)*chosen[0] + t*end) for t in np.linspace(0, 1, 17)[1:-1])
    passing = [d for d in directions if d[1] <= limit]
    result = min(passing, key=lambda c: abs(c[2]-target_rms)) if passing else min(directions, key=lambda c: c[1])
    return result[0], {"initial_low_peak": directions[0][1], "initial_low_rms": directions[0][2],
                       "predicted_peak": result[1], "predicted_rms": result[2], "target_rms": target_rms,
                       "peak_met": bool(passing), "excluded_rows": excluded}


def verify(np):
    matrix = np.zeros((128, 3), dtype=np.float32)
    matrix[0, 0] = .9
    matrix[1:64, 1] = .1171875
    matrix[64, 2] = .2
    matrix[65:, 2] = .01
    q, info = plan(matrix, .05, {"maximum_outlier_constraints": 2, "predicted_peak_limit": .12}, np)
    assert abs(np.linalg.norm(q)-1.) < 1e-12
    assert info["peak_met"] and info["predicted_peak"] <= .12
    assert info["excluded_rows"] and info["predicted_peak"] < info["initial_low_peak"]
