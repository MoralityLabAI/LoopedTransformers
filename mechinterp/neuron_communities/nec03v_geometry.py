"""NEC-03V source-only geometry feasibility. Pure stdlib; zero neural inference.

Input is an EXTERNALLY supplied archive of previously recorded, label-blind
source-fit TopK16 activations, original source-training case activations and
frozen projected SAE decoder columns. This tool NEVER loads native/SAE
checkpoints, generates labels, constructs new model inputs or scores behavior.

Usage:
    python -m mechinterp.neuron_communities.nec03v_geometry preflight \
        --pack /path/to/EXISTING/authenticated-source-geometry \
        --out /path/to/feasibility.json

A pack cannot establish the authenticity of its source by self-reported
hashes. Independent source-checkpoint/fit-rank identity checks must precede
any future scientific promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

SCHEMA = "NEC-03V/label-blind-source-geometry/v1"
ARM_SCHEMA = "NEC-03V/source-geometry-arm/v1"
RESULT_SCHEMA = "NEC-03V/geometry-preflight-unscored/v1"
ROOTS = tuple(range(8))
ARMS = ("composed", "additive")
GROUPS = tuple(range(24, 32))
MASKS = tuple(range(32))
ORIGINAL_SOURCE_FIT_ROWS = 1280
FEATURES = 128
ACTIVE = 16
K = 4
DIM = 32
LAMBDA = (0.0, 0.25, 0.5, 1.0)
MIN_COHERENCE = 0.10
MAX_SHAM_COHERENCE_GAP = 0.35
MIN_DOSE_32D = 1e-4
DOSE_RELATIVE_TOL = 0.01


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_keys(value: dict, required: set[str], name: str):
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{name}: expected fields {sorted(required)}")


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("source geometry contains nonfinite or nonnumeric scalar")
    if abs(value) > 1e8:
        raise ValueError("source geometry exceeds frozen finite magnitude cap")
    return float(value)


def ids16(values):
    if (not isinstance(values, list) or len(values) != ACTIVE
            or any(type(v) is not int or not 0 <= v < FEATURES for v in values)
            or values != sorted(set(values))):
        raise ValueError("every source-fit/case row must contain 16 distinct sorted feature IDs")
    return values


def vec32(value):
    if not isinstance(value, list) or len(value) != DIM:
        raise ValueError("projected decoder vectors must have exactly 32 coordinates")
    return tuple(finite(x) for x in value)


def original_lock_digest():
    p = Path(__file__).with_name("NEC03T_SOURCE_LOCK.json")
    original = json.loads(p.read_text(encoding="utf-8"))
    if (original.get("original_world_root_indices") != list(ROOTS)
            or original.get("source_group_indices") != list(GROUPS)
            or original.get("source_histories_per_arm") != len(GROUPS)*len(MASKS)
            or original.get("paired_ITT_zero") != [3, 5]):
        raise ValueError("original NEC03T source registration changed")
    return file_hash(p)


def _shard_path(pack: Path, name: str):
    if (not isinstance(name, str) or name != Path(name).name
            or name.startswith(".") or not name.endswith(".json")):
        raise ValueError("source manifest names must be ordinary local JSON filenames")
    return pack / name


def read_pack(pack: Path):
    """Read ONLY explicitly registered feature-geometry blobs from a local path."""
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check_keys(manifest, {
        "schema", "world_index", "world_id", "source_panel", "original_lock_sha256",
        "evidence", "source_checkpoint_receipts", "arm_shards",
    }, "geometry manifest")
    index = manifest["world_index"]
    if type(index) is not int or index not in ROOTS or manifest["world_id"] != f"dev5-{index:03d}":
        raise ValueError("only original historical eight five-input roots are permitted")
    if manifest["schema"] != SCHEMA or manifest["source_panel"] != list(GROUPS):
        raise ValueError("geometry pack has wrong schema or source-only panel")
    if manifest["original_lock_sha256"] != original_lock_digest():
        raise ValueError("original historical source lock checksum changed")
    if manifest["evidence"] != "cached_source_fit_rank_geometry_unscored":
        raise ValueError("source geometry evidence must be explicitly unscored")
    check_keys(manifest["arm_shards"], set(ARMS), "registered arm shards")
    check_keys(manifest["source_checkpoint_receipts"], set(ARMS), "source checkpoint receipts")
    shards = {}
    for arm in ARMS:
        receipt = manifest["source_checkpoint_receipts"][arm]
        check_keys(receipt, {"native_sha256", "sae_sha256", "source_sites_sha256",
                             "origin_receipt_sha256"}, "checkpoint provenance")
        if any(not isinstance(v, str) or len(v) != 64
               or any(c not in "0123456789abcdef" for c in v) for v in receipt.values()):
            raise ValueError("missing 64-digit immutable upstream source provenance")
        item = manifest["arm_shards"][arm]
        check_keys(item, {"file", "sha256"}, "arm shard")
        path = _shard_path(pack, item["file"])
        if item["sha256"] != file_hash(path):
            raise ValueError(f"{arm}: SHA-256 source-geometry shard mismatch")
        data = json.loads(path.read_text(encoding="utf-8"))
        check_keys(data, {"schema", "arm", "world_index", "visits", "cases"}, "source arm")
        if data["schema"] != ARM_SCHEMA or data["arm"] != arm or data["world_index"] != index:
            raise ValueError("source-arm identity or role mismatch")
        if not isinstance(data["visits"], dict) or not 1 <= len(data["visits"]) <= 12:
            raise ValueError("one to twelve original registered source visits required")
        allowed_visits = sorted(data["visits"], key=int)
        if (any(not key.isdecimal() or int(key) not in range(12)
                or str(int(key)) != key for key in allowed_visits)):
            raise ValueError("invalid original 12-pass recurrence visit")
        for visit in allowed_visits:
            spec = data["visits"][visit]
            check_keys(spec, {"decoder32", "source_fit_active"}, "visit geometry")
            vectors = spec["decoder32"]
            if not isinstance(vectors, list) or len(vectors) != FEATURES:
                raise ValueError("frozen SAE geometry must have 128 decoder columns")
            spec["decoder32"] = [vec32(v) for v in vectors]
            fits = spec["source_fit_active"]
            if (not isinstance(fits, list)
                    or len(fits) != ORIGINAL_SOURCE_FIT_ROWS):
                raise ValueError("expected exactly 1280 label-blind source-fit rows per visit")
            for ids in fits:
                ids16(ids)
        cases = data["cases"]
        if not isinstance(cases, list) or len(cases) != len(GROUPS) * len(MASKS):
            raise ValueError("expected exactly 8 source groups by 32 masks per original arm")
        for expected, case in enumerate(cases):
            check_keys(case, {"group", "mask", "visits"}, "source case")
            if (type(case["group"]) is not int or case["group"] != GROUPS[expected // 32]
                    or type(case["mask"]) is not int or case["mask"] != expected % 32):
                raise ValueError("source cohort must be exactly ordered original 24..31 x 32 masks")
            if not isinstance(case["visits"], dict) or set(case["visits"]) != set(allowed_visits):
                raise ValueError("incomplete registered per-case original recurrence visits")
            for visit in allowed_visits:
                active = case["visits"][visit]
                if not isinstance(active, list) or len(active) != ACTIVE:
                    raise ValueError("each visit needs 16 active positive TopK codes")
                if any(not isinstance(pair, list) or len(pair) != 2
                       or type(pair[0]) is not int or type(pair[1]) not in (int, float)
                       for pair in active):
                    raise ValueError("only ID and code coefficient allowed in case geometry")
                ids16([pair[0] for pair in active])
                if any(not finite(pair[1]) > 1e-7 for pair in active):
                    raise ValueError("active coefficients must be finite positive TopK codes")
        shards[arm] = data
    return manifest, shards, file_hash(manifest_path)


def dot(a, b):
    return math.fsum(x*y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(math.fsum(x*x for x in a))


def graph(fit_active: list[list[int]], decoder: list[tuple[float, ...]]):
    count = [0] * FEATURES
    both: dict[tuple[int, int], int] = {}
    for ids in fit_active:
        for feature in ids:
            count[feature] += 1
        for pair in itertools.combinations(ids, 2):
            both[pair] = both.get(pair, 0) + 1
    lengths = [norm(v) for v in decoder]
    edges = {}
    for i in range(FEATURES):
        for j in range(i + 1, FEATURES):
            union = count[i] + count[j] - both.get((i, j), 0)
            overlap = both.get((i, j), 0) / union if union else 0.0
            cosine = (max(0.0, dot(decoder[i], decoder[j]) / (lengths[i]*lengths[j]))
                      if lengths[i] > 1e-12 and lengths[j] > 1e-12 else 0.0)
            edges[(i, j)] = 0.5 * (overlap + cosine)
    return edges


def group_score(group: tuple[int, ...], edges: dict):
    return math.fsum(edges[(a, b)] for a, b in itertools.combinations(group, 2)) / 6


def connected(group: tuple[int, ...], edges: dict):
    visited = {group[0]}
    while True:
        new = visited | {b for a in visited for b in group
                         if a != b and edges[tuple(sorted((a, b)))] > 0.01}
        if new == visited:
            return len(visited) == len(group)
        visited = new


def removal(ids: tuple[int, ...], active: dict[int, float], decoder: list):
    return tuple(math.fsum(active[i]*decoder[i][coordinate] for i in ids)
                 for coordinate in range(DIM))


def choose_groups(active_codes: list, cols: list, edges: dict):
    active = {item[0]: item[1] for item in active_codes}
    ids = tuple(sorted(active))
    candidates = [tuple(c) for c in itertools.combinations(ids, K)
                  if connected(c, edges) and group_score(c, edges) >= MIN_COHERENCE]
    if not candidates:
        return {"ok": False, "reason": "missing_coherent_source_group", "candidate_count": 0}
    source = min(candidates, key=lambda g: (-group_score(g, edges), g))
    source_score = group_score(source, edges)
    raw_source = removal(source, active, cols)
    source_norm = norm(raw_source)
    sham_candidates = [h for h in candidates if set(h).isdisjoint(source)]
    if not sham_candidates:
        return {"ok": False, "reason": "missing_disjoint_structural_sham",
                "candidate_count": 0, "source": list(source)}
    def sham_rank(h):
        x = abs(source_score - group_score(h, edges))
        y = abs(norm(removal(h, active, cols)) - source_norm) / max(source_norm, MIN_DOSE_32D)
        return (round(x, 12), round(y, 12), h)
    sham = min(sham_candidates, key=sham_rank)
    sham_score = group_score(sham, edges)
    raw_sham = removal(sham, active, cols)
    sham_norm = norm(raw_sham)
    gap = abs(source_score - sham_score)
    dose = min(source_norm, sham_norm)
    valid = gap <= MAX_SHAM_COHERENCE_GAP and dose >= MIN_DOSE_32D
    if not valid:
        return {"ok": False, "reason": (
            "sham_coherence_gap" if gap > MAX_SHAM_COHERENCE_GAP else "weak_shared_32d_dose"),
            "source": list(source), "sham": list(sham),
            "candidate_count": len(sham_candidates), "source_coherence": round(source_score, 8),
            "sham_coherence": round(sham_score, 8), "coherence_gap": round(gap, 8),
            "source_norm": round(source_norm, 8), "sham_norm": round(sham_norm, 8)}
    source_gain = dose / source_norm
    sham_gain = dose / sham_norm
    source_vec = tuple(source_gain*x for x in raw_source)
    sham_vec = tuple(sham_gain*x for x in raw_sham)
    source_actual = norm(source_vec)
    sham_actual = norm(sham_vec)
    error = abs(source_actual-sham_actual) / max(dose, MIN_DOSE_32D)
    if (error > DOSE_RELATIVE_TOL or source_gain > 1 + 1e-12
            or sham_gain > 1 + 1e-12):
        raise AssertionError("nonamplification or matched physical 32D dose gate failed")
    return {"ok": True, "source": list(source), "sham": list(sham),
            "candidate_count": len(sham_candidates),
            "source_coherence": round(source_score, 8),
            "sham_coherence": round(sham_score, 8), "coherence_gap": round(gap, 8),
            "source_norm": round(source_norm, 8), "sham_norm": round(sham_norm, 8),
            "shared_dose": round(dose, 8),
            "source_gain": round(source_gain, 8), "sham_gain": round(sham_gain, 8),
            "actual_dose_source": round(source_actual, 8),
            "actual_dose_sham": round(sham_actual, 8),
            "relative_match_error": round(error, 8),
            "scaled_source32": [round(x, 8) for x in source_vec],
            "scaled_sham32": [round(x, 8) for x in sham_vec]}


def preflight(pack: Path):
    manifest, shards, source_sha = read_pack(pack)
    output = []
    groups_by_arm = {}
    for arm in ARMS:
        shard = shards[arm]
        graphs = {visit: graph(spec["source_fit_active"], spec["decoder32"])
                  for visit, spec in shard["visits"].items()}
        groups_by_arm[arm] = {visit: digest({str(k): round(v, 12) for k,v in weights.items()})
                              for visit, weights in graphs.items()}
        for case in shard["cases"]:
            visits = {
                visit: choose_groups(case["visits"][visit],
                                      shard["visits"][visit]["decoder32"],
                                      graphs[visit])
                for visit in sorted(shard["visits"], key=int)
            }
            eligible = all(v["ok"] for v in visits.values())
            if not eligible:
                for v in visits.values():
                    v["effective_zero_pair"] = True
                    if v["ok"]:
                        v["reason"] = "other_visit_invalid_case_zero"
                    v["effective_source32"] = [0.0] * DIM
                    v["effective_sham32"] = [0.0] * DIM
            else:
                for v in visits.values():
                    v["effective_zero_pair"] = False
                    v["reason"] = "valid"
                    v["effective_source32"] = v["scaled_source32"]
                    v["effective_sham32"] = v["scaled_sham32"]
            output.append({"arm": arm, "group": case["group"], "mask": case["mask"],
                           "eligible_all_visits": eligible, "visits": visits})
    summary = {}
    for arm in ARMS:
        rows = [r for r in output if r["arm"] == arm]
        ok = [r for r in rows if r["eligible_all_visits"]]
        summary[arm] = {
            "cases": len(rows), "eligible_cases": len(ok),
            "invalid_as_intention_to_treat_zero_pair": len(rows)-len(ok),
            "coverage": len(ok)/len(rows),
            "valid_pair_min_sham_candidates": min(
                (v["candidate_count"] for row in ok for v in row["visits"].values()), default=0),
            "valid_pair_min_dose_32d": min(
                (v["shared_dose"] for row in ok for v in row["visits"].values()), default=0),
            "visit_graph_sha256": groups_by_arm[arm]}
    return {
        "schema": RESULT_SCHEMA, "world_id": manifest["world_id"],
        "world_index": manifest["world_index"], "origin_pack_manifest_sha256": source_sha,
        "original_nec03t_lock_sha256": manifest["original_lock_sha256"],
        "evidence_kind": manifest["evidence"], "source_scope": list(GROUPS),
        "source_checkpoint_receipts": manifest["source_checkpoint_receipts"],
        "arms": summary, "case_receipts": output,
        "registered_primary_scoring_enabled": False,
        "labels_accessed": 0, "logits_accessed": 0, "model_forwards": 0,
        "new_neural_or_SAE_training_updates": 0,
        "new_recruited_root_reads": 0, "embargoed_or_confirmation_reads": 0,
        "next_gate": "independent-geometry-and-source-identity-audit",
        "scientific_gate": "REDESIGN_OR_PILOT_ONLY",
        "behavioral_effect": None, "source_only_pilot_scored": False,
        "authenticity_warning": "SHA does not authenticate the geometry producer; upstream source checkpoint identity audit required"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    run = commands.add_parser("preflight")
    run.add_argument("--pack", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.command != "preflight" or args.out.exists():
        raise ValueError("preflight only, output must not exist")
    result = preflight(args.pack)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(canonical(result) + "\n", encoding="utf-8")
    print(canonical({k: result[k] for k in ("world_id", "arms", "model_forwards",
                                           "next_gate", "scientific_gate")}))
    return result


if __name__ == "__main__":
    main()
