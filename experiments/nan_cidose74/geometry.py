"""Build equal-BF16-dose alternatives for one observed native write."""
from hashlib import sha256

from experiments.nan_carry63.algebra import round_bf16
from experiments.nan_cigate73.account import route
from experiments.nan_delay39.instrument import prepare


EPS = 1e-6
DOSE = .02
DOSE_TOLERANCE = .02
COSINE = .98
ARMS = ("dynamic_on", "static_on", "dynamic_off", "static_off",
        "dynamic_tangent", "dynamic_radial", "random_tangent")
CONTRASTS = {
    "dynamic_minus_static_on": {"dynamic_on": 1, "static_on": -1},
    "dynamic_minus_static_off": {"dynamic_off": 1, "static_off": -1},
    "sphere_interaction": {"dynamic_on": 1, "static_on": -1, "dynamic_off": -1, "static_off": 1},
    "dynamic_on_minus_radial": {"dynamic_on": 1, "dynamic_radial": -1},
    "dynamic_tangent_minus_random": {"dynamic_tangent": 1, "random_tangent": -1},
}


def directions(x, native, dynamic, static, bank, phase, root, np):
    v, u = bank["V"], bank["U"]
    mean = bank["mean_comp_acts"][phase - 1]
    sphere_rms = bank["mean_write_rms"][phase - 1]
    centered = x @ v - mean
    choices = {}
    parity = 0
    for name, gate in (("dynamic", dynamic), ("static", static)):
        adjusted = native + ((gate - 1) * centered) @ u
        power = np.mean(adjusted**2, axis=1, keepdims=True)
        rms = np.sqrt(power + EPS)
        mix = (1 - gate).mean(axis=1, keepdims=True)
        desired = (1 - mix) * rms + mix * max(sphere_rms, EPS)
        projected = adjusted * np.where(power > EPS, desired / rms, 1.)
        for key, row, target in (("on", projected, sphere_rms), ("off", adjusted, None)):
            reference, _ = route(x, native, gate, v, u, mean, target, EPS, np)
            np.testing.assert_array_equal(round_bf16(row, np), reference)
            choices[f"{name}_{key}"] = row - native
            parity += 2

    lengths = np.linalg.norm(native, axis=1, keepdims=True)
    if (lengths == 0).any():
        return {}, parity
    axis = native / lengths
    raw = choices["dynamic_on"]
    radial = np.sum(raw * axis, axis=1, keepdims=True) * axis
    seed = sha256(f"NAN-CIDOSE-74-v1|{root}|{phase}".encode()).digest()[:8]
    random = np.random.default_rng(int.from_bytes(seed, "big")).normal(size=native.shape[1])
    random_axis = random - np.sum(random * axis, axis=1, keepdims=True) * axis
    return choices | {"dynamic_tangent": raw - radial, "dynamic_radial": radial,
                      "random_tangent": random_axis}, parity


def match(native, wanted, np):
    if not wanted:
        return {}, {"status": "zero_native_write", "arms": {}}
    all_writes = {}
    measurements = {}
    unit = native / np.linalg.norm(native, axis=1, keepdims=True)
    for arm in ARMS:
        edits = [prepare(base, direction, DOSE, DOSE_TOLERANCE, COSINE, np)
                 for base, direction in zip(native, wanted[arm], strict=True)]
        reports = []
        for base, patch, radial in zip(native, edits, unit, strict=True):
            delta = patch.after - base
            reports.append({"eligible": patch.eligible, "dose": patch.realized_l2,
                            "cosine": patch.cosine,
                            "signed_radial_share": float(delta @ radial / patch.realized_l2) if patch.realized_l2 else None,
                            "relative_norm_drift": float(np.linalg.norm(patch.after)/np.linalg.norm(base) - 1.)})
        measurements[arm] = reports
        if all(edit.eligible for edit in edits):
            all_writes[arm] = np.stack([edit.after for edit in edits])
    spreads = [max(doses) - min(doses) for doses in zip(*([row["dose"] for row in details]
                                                          for details in measurements.values()), strict=True)]
    nominal = DOSE * np.linalg.norm(native, axis=1)
    receipt = {"arms": measurements, "nominal_l2": nominal.tolist(), "absolute_dose_spread": spreads}
    if set(all_writes) != set(ARMS):
        return {}, receipt | {"status": "physical_arm_unqualified"}
    if any(spread > desired * DOSE_TOLERANCE for spread, desired in zip(spreads, nominal, strict=True)):
        return {}, receipt | {"status": "joint_dose_band_unqualified"}
    return all_writes, receipt | {"status": "matched"}


def verify(np):
    rng = np.random.default_rng(74023)
    native = round_bf16(rng.normal(size=(2, 144)), np).astype(np.float64)
    x = round_bf16(rng.normal(size=(2, 32)), np).astype(np.float64)
    gate = rng.uniform(low=.1, high=.9, size=(2, 6))
    static = np.broadcast_to(gate.mean(axis=0), gate.shape)
    bank = {"V": rng.normal(size=(32, 6)) * .1, "U": rng.normal(size=(6, 144)) * .1,
            "mean_comp_acts": np.zeros((2, 6)), "mean_write_rms": np.array([.5, .5])}
    choices, parity = directions(x, native, gate, static, bank, 1, "fixture", np)
    assert parity == 8
    np.testing.assert_allclose(np.sum(choices["dynamic_tangent"] * native, axis=1), np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(np.sum(choices["random_tangent"] * native, axis=1), np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(choices["dynamic_tangent"] + choices["dynamic_radial"],
                               choices["dynamic_on"], atol=1e-12, rtol=0)
    outputs, receipt = match(native, choices, np)
    assert receipt["status"] == "matched" and set(outputs) == set(ARMS)
    failed, refusal = match(native, choices | {"dynamic_radial": np.zeros_like(native)}, np)
    assert not failed and refusal["status"] == "physical_arm_unqualified"
    for reached in outputs.values():
        np.testing.assert_array_equal(reached, round_bf16(reached, np))
    return True
