"""Plan matched displacements on the BF16 state actually handed to pass 2."""
from hashlib import sha256

from experiments.nan_delay39.instrument import prepare
from experiments.nan_readpath37.run import bf16


COORDINATE = 2279
DOSE = .0075
DOSE_TOLERANCE = .02
NORM_TOLERANCE = .001
SHAM_COUNT = 4


def plan(clean, coordinate, root_key, np):
    """Return all registered arms, or refuse the complete matched comparison."""
    base = np.asarray(clean, dtype=np.float64)
    if (base.ndim != 1 or not np.isfinite(base).all() or not 0 <= coordinate < base.size
            or base.size <= SHAM_COUNT or not np.array_equal(base, bf16(base, np))):
        raise ValueError("Expected a finite BF16 reader input and a valid coordinate")
    radius = float(np.linalg.norm(base))
    wanted = DOSE * radius
    if radius == 0 or abs(base[coordinate]) < wanted:
        return {}, {"status": "insufficient_target_amplitude", "requested_l2": wanted}

    def axis(index, sign):
        direction = np.zeros_like(base)
        direction[index] = sign
        return direction

    target = axis(coordinate, -np.sign(base[coordinate]))
    gen = np.random.default_rng(int.from_bytes(sha256(
        f"NAN-READER71-v1|{root_key}".encode()).digest()[:8], "big"))
    available = np.delete(np.arange(base.size), coordinate)
    indices = gen.choice(available, SHAM_COUNT, replace=False)
    directions = {"target": target}
    for i, index in enumerate(indices):
        directions[f"coordinate_sham_{i + 1}"] = axis(index, gen.choice([-1., 1.]))
    vector = gen.normal(size=base.size)
    vector[coordinate] = 0.
    directions["vector_sham"] = vector

    changes, direction_cosines = {}, {}
    for name, direction in directions.items():
        # Distributed edits necessarily lose more direction to BF16 rounding.
        minimum_cosine = .95 if name == "vector_sham" else .99
        patch = prepare(base, direction, DOSE, DOSE_TOLERANCE, minimum_cosine, np)
        direction_cosines[name] = patch.cosine
        if patch.eligible:
            changes[name] = patch.after

    # Move toward zero along a great-circle arc, matching the chord length.
    radial = base / radius
    tangent = target - float(target @ radial) * radial
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm > 0:
        cos_angle = 1 - DOSE**2 / 2
        sin_angle = DOSE * np.sqrt(1 - DOSE**2 / 4)
        rotated = bf16(radius * (cos_angle * radial + sin_angle * tangent / tangent_norm), np)
        if abs(rotated[coordinate]) < abs(base[coordinate]) and np.sign(rotated[coordinate]) == np.sign(base[coordinate]):
            changes["norm_preserving_target"] = rotated

    stats = {}
    for name, after in changes.items():
        stats[name] = {
            "realized_l2": float(np.linalg.norm(after - base)),
            "relative_norm_drift": float(np.linalg.norm(after) / radius - 1),
            "target_coordinate_delta": float(after[coordinate] - base[coordinate]),
        }
    receipt = {"requested_l2": wanted, "arms": stats, "coordinate_shams": indices.tolist(),
               "direction_cosines": direction_cosines,
               "clean_coordinate_energy_share": float((base[coordinate] / radius)**2)}
    doses = [item["realized_l2"] for item in stats.values()]
    matched = (set(changes) == set(directions) | {"norm_preserving_target"}
               and abs(changes["target"][coordinate]) < abs(base[coordinate])
               and np.sign(changes["target"][coordinate]) == np.sign(base[coordinate])
               and abs(stats["norm_preserving_target"]["relative_norm_drift"]) <= NORM_TOLERANCE
               and all(abs(dose / wanted - 1) <= DOSE_TOLERANCE for dose in doses)
               and max(doses) - min(doses) <= wanted * DOSE_TOLERANCE)
    if not matched:
        return {}, receipt | {"status": "matched_dose_unqualified"}
    return changes, receipt | {"status": "matched"}
