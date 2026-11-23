"""Lite-Braille Fresco: deterministic glyph codec and bounded trace-decision gym.

Standard-library-only. The encoder observes predecision quantities; any verifier
outcome belongs to a separate label file. Visual ablations never alter a model.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
from pathlib import Path
from typing import Any

SCHEMA_OBS = "lbf.observation.v1"
SCHEMA_ENCODED = "lbf.encoded.v1"
SCHEMA_LABEL = "lbf.labels.v1"
ROLES = ("goal", "context", "lemma", "plan", "value", "message", "halt", "delegate")
FRAMES = ("state", "action", "gate", "memory", "closure", "error")
SPLITS = ("train", "calibration", "ordinary_test", "ood_family")
OUTCOMES = ("success", "failure", "censored")
POLARITIES = (-1, 0, 1)
MAX_VISITS = 64
MAX_TILES_PER_VISIT = 512
MAX_CANDIDATES = 64


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(canonical(row) + "\n" for row in records), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{n}: expected an object")
            items.append(record)
    return items


def exact_keys(obj: dict, keys: set[str], context: str) -> None:
    if not isinstance(obj, dict) or set(obj) != keys:
        raise ValueError(f"{context}: expected exactly {sorted(keys)}")


def identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 160 or any(ord(x) < 32 for x in value):
        raise ValueError(f"{where}: invalid identifier")
    return value


def glyph(role: str, frame: str, polarity: int = 0) -> str:
    """Injective Unicode Braille-cell code, distinct from language Braille."""
    if role not in ROLES or frame not in FRAMES or type(polarity) is not int or polarity not in POLARITIES:
        raise ValueError("unknown role/frame/polarity")
    r = ROLES.index(role)                 # braille dots 1,2,3; 0..7
    f = FRAMES.index(frame) + 1           # braille dots 4,5,6; 1..6
    mask = r | (f << 3) | (64 if polarity == 1 else 128 if polarity == -1 else 0)
    return chr(0x2800 + mask)


def unglyph(char: str) -> tuple[str, str, int]:
    if not isinstance(char, str) or len(char) != 1 or not 0x2800 <= ord(char) <= 0x28ff:
        raise ValueError("one Unicode Braille cell required")
    mask = ord(char) - 0x2800
    f = ((mask >> 3) & 7) - 1
    if f not in range(len(FRAMES)) or (mask & 192) == 192:
        raise ValueError("noncanonical glyph")
    p = 1 if mask & 64 else -1 if mask & 128 else 0
    return ROLES[mask & 7], FRAMES[f], p


def light_level(value: float) -> int:
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("activation must be a finite number in [0,1]")
    return min(3, int(4 * value))


def validate_observation(row: dict) -> None:
    exact_keys(row, {"schema", "root_id", "lineage", "split", "source", "candidates", "visits"}, "observation")
    if row["schema"] != SCHEMA_OBS or row["split"] not in SPLITS:
        raise ValueError("invalid observation version/split")
    identifier(row["root_id"], "root_id")
    identifier(row["lineage"], "lineage")
    exact_keys(row["source"], {"kind", "source_sha256", "checkpoint_sha256"}, "source")
    if row["source"]["kind"] not in ("synthetic_fixture", "ppt_replay"):
        raise ValueError("source kind must be synthetic_fixture or ppt_replay")
    for k in ("source_sha256", "checkpoint_sha256"):
        v = row["source"][k]
        if v is not None and (not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)):
            raise ValueError(f"{k}: SHA-256 lowercase hex or null expected")
    cs = row["candidates"]
    if not isinstance(cs, list) or not 1 <= len(cs) <= MAX_CANDIDATES:
        raise ValueError("1..64 candidates required")
    for c in cs:
        identifier(c, "candidate id")
    if len(cs) != len(set(cs)):
        raise ValueError("duplicate candidate")
    visits = row["visits"]
    if not isinstance(visits, list) or not 1 <= len(visits) <= MAX_VISITS:
        raise ValueError("1..64 visits required")
    previous = (0, 0)
    for v in visits:
        exact_keys(v, {"attempt", "pass", "tiles"}, "visit")
        if type(v["attempt"]) is not int or type(v["pass"]) is not int:
            raise ValueError("integer clock required")
        clock = (v["attempt"], v["pass"])
        if not 1 <= v["attempt"] <= 32 or not 1 <= v["pass"] <= 16 or clock <= previous:
            raise ValueError("visits must have increasing positive (attempt,pass) clocks")
        previous = clock
        if not isinstance(v["tiles"], list) or not 1 <= len(v["tiles"]) <= MAX_TILES_PER_VISIT:
            raise ValueError("1..512 tiles per visit required")
        found = set()
        for t in v["tiles"]:
            exact_keys(t, {"candidate_id", "role", "frame", "activation", "polarity"}, "tile")
            if t["candidate_id"] != "shared" and t["candidate_id"] not in cs:
                raise ValueError("tile candidate absent from menu")
            glyph(t["role"], t["frame"], t["polarity"])
            light_level(t["activation"])
            key = (t["candidate_id"], t["role"])
            if key in found:
                raise ValueError("duplicate candidate/role in one visit")
            found.add(key)


def encode(row: dict) -> dict:
    """Strict output allowlist: verdicts and reference proofs cannot cross."""
    validate_observation(row)
    return {
        "schema": SCHEMA_ENCODED,
        "root_id": row["root_id"], "lineage": row["lineage"],
        "split": row["split"], "source": dict(row["source"]),
        "candidates": list(row["candidates"]), "observation_sha256": digest(row),
        "visits": [{"attempt": v["attempt"], "pass": v["pass"],
                    "tiles": [{"candidate_id": t["candidate_id"],
                               "glyph": glyph(t["role"], t["frame"], t["polarity"]),
                               "light": light_level(t["activation"])}
                              for t in v["tiles"]]} for v in row["visits"]],
    }


def validate_encoded(row: dict) -> None:
    exact_keys(row, {"schema", "root_id", "lineage", "split", "source", "candidates", "observation_sha256", "visits"}, "encoded")
    if row["schema"] != SCHEMA_ENCODED or row["split"] not in SPLITS:
        raise ValueError("encoded version/split invalid")
    identifier(row["root_id"], "root_id")
    identifier(row["lineage"], "lineage")
    if len(row["observation_sha256"]) != 64:
        raise ValueError("missing raw observation digest")
    if not isinstance(row["candidates"], list) or not 1 <= len(row["candidates"]) <= MAX_CANDIDATES:
        raise ValueError("invalid candidate roster")
    if len(set(row["candidates"])) != len(row["candidates"]):
        raise ValueError("repeated candidate")
    if not isinstance(row["visits"], list) or not 1 <= len(row["visits"]) <= MAX_VISITS:
        raise ValueError("invalid number of visits")
    prev = (0, 0)
    for visit in row["visits"]:
        exact_keys(visit, {"attempt", "pass", "tiles"}, "encoded visit")
        clock = (visit["attempt"], visit["pass"])
        if clock <= prev or not 1 <= clock[0] <= 32 or not 1 <= clock[1] <= 16:
            raise ValueError("clock ordering invalid")
        prev = clock
        for t in visit["tiles"]:
            exact_keys(t, {"candidate_id", "glyph", "light"}, "encoded tile")
            if t["candidate_id"] not in row["candidates"] + ["shared"]:
                raise ValueError("unknown tile candidate")
            unglyph(t["glyph"])
            if type(t["light"]) is not int or not 0 <= t["light"] <= 3:
                raise ValueError("bad LED level")


def validate_labels(label: dict, obs: dict) -> None:
    exact_keys(label, {"schema", "root_id", "lineage", "evidence_class", "outcomes"}, "label")
    if label["schema"] != SCHEMA_LABEL or label["root_id"] != obs["root_id"] or label["lineage"] != obs["lineage"]:
        raise ValueError("label/observation identity mismatch")
    if label["evidence_class"] not in ("synthetic_fixture", "externally_attested_native"):
        raise ValueError("unknown provenance class")
    if ((obs["source"]["kind"] == "synthetic_fixture") !=
            (label["evidence_class"] == "synthetic_fixture")):
        raise ValueError("source/label provenance conflict")
    if not isinstance(label["outcomes"], dict) or set(label["outcomes"]) != set(obs["candidates"]):
        raise ValueError("one status per candidate required")
    if any(v not in OUTCOMES for v in label["outcomes"].values()):
        raise ValueError("invalid outcome class")


class FrescoEnv:
    """No access to hidden labels via observation(); actions: continue/select/abstain."""
    def __init__(self, encoded: dict, labels: dict, continue_cost: float = 0.02):
        validate_encoded(encoded)
        validate_labels(labels, encoded)
        if not 0 <= continue_cost <= 1:
            raise ValueError("invalid continuation cost")
        self.trace = encoded
        self._labels = labels["outcomes"]
        self.continue_cost = continue_cost
        self.pos = 0
        self.charges = 0.0
        self.done = False

    def observation(self) -> dict:
        return {"schema": SCHEMA_ENCODED, "root_id": self.trace["root_id"],
                "candidates": list(self.trace["candidates"]),
                "visits": self.trace["visits"][:self.pos+1],
                "remaining_visits": len(self.trace["visits"]) - self.pos - 1}

    def step(self, action: str, candidate_id: str | None = None) -> dict:
        if self.done:
            raise ValueError("episode already ended")
        if action == "continue":
            if self.pos == len(self.trace["visits"]) - 1:
                raise ValueError("no future pass available")
            self.pos += 1
            self.charges += self.continue_cost
            return {"done": False, "reward": -self.continue_cost, "total_reward": -self.charges, "status": "continued"}
        if action == "abstain":
            self.done = True
            return {"done": True, "reward": 0.0, "total_reward": -self.charges, "status": "abstained"}
        if action != "select" or candidate_id not in self.trace["candidates"]:
            raise ValueError("select requires a listed candidate_id")
        self.done = True
        status = self._labels[candidate_id]
        return {"done": True, "reward": 1.0 if status == "success" else 0.0,
                "total_reward": (1.0 if status == "success" else 0.0) - self.charges,
                "status": status, "scientific_eligible": False}


def intervene(encoded: dict, mode: str) -> dict:
    """Representation-only controls; cannot imply neural causal effects."""
    if mode not in ("intact", "zero_value", "roll_roles", "zero_message"):
        raise ValueError("unsupported control")
    validate_encoded(encoded)
    data = json.loads(canonical(encoded))
    for visit in data["visits"]:
        for tile in visit["tiles"]:
            role, frame, polarity = unglyph(tile["glyph"])
            if mode == "zero_value" and role == "value":
                tile["light"] = 0
            elif mode == "zero_message" and role == "message":
                tile["light"] = 0
            elif mode == "roll_roles":
                tile["glyph"] = glyph(ROLES[(ROLES.index(role) + 1) % len(ROLES)], frame, polarity)
    return data


def choose_candidate(encoded: dict, policy: str, seed: int = 0) -> str:
    if policy not in ("glyph", "led", "random"):
        raise ValueError("unknown policy")
    cs = encoded["candidates"]
    if policy == "random":
        h = hashlib.sha256((str(seed) + ":" + encoded["root_id"]).encode()).digest()
        return cs[int.from_bytes(h[:8], "big") % len(cs)]
    scores = {c: -1 for c in cs}
    for tile in encoded["visits"][-1]["tiles"]:
        c = tile["candidate_id"]
        if c == "shared":
            continue
        if policy == "glyph":
            role, frame, _ = unglyph(tile["glyph"])
            if role != "value" or frame != "action":
                continue
        scores[c] = max(scores[c], tile["light"])
    return max(cs, key=lambda c: (scores[c], -cs.index(c)))


def score(encoded_rows: list[dict], label_rows: list[dict], seed: int = 17) -> dict:
    label_map = {r["root_id"]: r for r in label_rows}
    if len(label_map) != len(label_rows) or len(encoded_rows) != len({r["root_id"] for r in encoded_rows}):
        raise ValueError("duplicate root identity")
    controls = ("intact", "zero_value", "roll_roles", "zero_message")
    policies = ("glyph", "led", "random")
    outcomes = {c: {p: [] for p in policies} for c in controls}
    skipped = 0
    eligible_roots = []
    for row in encoded_rows:
        validate_encoded(row)
        if row["root_id"] not in label_map:
            raise ValueError("missing privately held labels")
        label = label_map[row["root_id"]]
        validate_labels(label, row)
        if "censored" in label["outcomes"].values():
            skipped += 1
            continue
        eligible_roots.append(row["root_id"])
        for mode in controls:
            x = intervene(row, mode)
            for p in policies:
                selected = choose_candidate(x, p, seed)
                outcomes[mode][p].append(int(label["outcomes"][selected] == "success"))
    return {"schema": "lbf.score.v1", "roots_seen": len(encoded_rows),
            "roots_eligible": len(eligible_roots), "roots_censored": skipped,
            "evidence_classes": sorted({r["evidence_class"] for r in label_rows}),
            "scientific_eligible": False, "certification_replayed": False,
            "results": {c: {p: {"correct": sum(values), "n": len(values),
                                   "rate": (sum(values)/len(values) if values else None)}
                            for p, values in arms.items()} for c, arms in outcomes.items()},
            "mean_paired_glyph_minus_led": (
                sum(a-b for a,b in zip(outcomes["intact"]["glyph"], outcomes["intact"]["led"])) / len(eligible_roots)
                if eligible_roots else None)}


def fixture(n: int, seed: int) -> tuple[list[dict], list[dict]]:
    if not 1 <= n <= 4096:
        raise ValueError("fixture must contain 1..4096 roots")
    rng = random.Random(seed)
    obs, labels = [], []
    for i in range(n):
        winner = rng.randrange(4)
        choices = [f"t{j}" for j in range(4)]
        visits = []
        for k in range(1, 4):
            tiles = []
            for j, cand in enumerate(choices):
                # A purpose-built codebook-control fixture. Distractor LEDs
                # deliberately exceed the useful VALUE channel's brightness.
                amplitude = (0.81 + rng.random()*.16 if j == winner else .06 + rng.random()*.34)
                tiles.extend([
                    {"candidate_id": cand, "role": "value", "frame": "action",
                     "activation": amplitude, "polarity": 1 if j == winner else -1},
                    {"candidate_id": cand, "role": "message", "frame": "memory",
                     "activation": .92 + rng.random()*.07 if j != winner else .15, "polarity": 0},
                    {"candidate_id": cand, "role": "goal", "frame": "state",
                     "activation": .2 + rng.random()*.4, "polarity": 0},
                ])
            visits.append({"attempt": 1, "pass": k, "tiles": tiles})
        split = "train" if i % 5 < 3 else "calibration" if i % 5 == 3 else "ordinary_test"
        root = f"fixture-{seed}-{i:04d}"
        row = {"schema": SCHEMA_OBS, "root_id": root,
               "lineage": f"toy-family:{seed}:{i:04d}", "split": split,
               "source": {"kind": "synthetic_fixture", "source_sha256": digest({"fixture_seed":seed,"index":i}),
                          "checkpoint_sha256": None},
               "candidates": choices, "visits": visits}
        validate_observation(row)
        obs.append(row)
        labels.append({"schema": SCHEMA_LABEL, "root_id": root, "lineage": row["lineage"],
                       "evidence_class": "synthetic_fixture",
                       "outcomes": {c: "success" if j == winner else "failure" for j,c in enumerate(choices)}})
    return obs, labels


def _frame_svg(frame: str, x: float, y: float) -> str:
    col = '#807b70'
    if frame == 'state':
        return f'<rect x="{x+10}" y="{y+9}" width="48" height="48" rx="5" fill="none" stroke="{col}" stroke-width="1.5"/>'
    if frame == 'action':
        return f'<path d="M {x+34} {y+5} L {x+63} {y+34} L {x+34} {y+63} L {x+5} {y+34} Z" fill="none" stroke="{col}" stroke-width="1.5"/>'
    if frame == 'gate':
        return f'<path d="M {x+8} {y+59} V {y+25} Q {x+34} {y-6} {x+60} {y+25} V {y+59}" fill="none" stroke="{col}" stroke-width="1.6"/>'
    if frame == 'memory':
        return f'<circle cx="{x+34}" cy="{y+34}" r="30" fill="none" stroke="{col}" stroke-width="1.5"/>'
    if frame == 'closure':
        return f'<path d="M {x+34} {y+4} L {x+61} {y+20} V {y+49} L {x+34} {y+64} L {x+7} {y+49} V {y+20} Z" fill="none" stroke="{col}" stroke-width="1.5"/>'
    return f'<path d="M {x+9} {y+9} L {x+59} {y+59} M {x+59} {y+9} L {x+9} {y+59}" stroke="{col}" stroke-width="2"/>'


def fresco_svg(row: dict) -> str:
    """Pure, noninteractive, accessible SVG tile fresco. Labels are escaped."""
    validate_encoded(row)
    visits = row["visits"]
    identities = sorted({(t["candidate_id"], unglyph(t["glyph"])[0])
                         for v in visits for t in v["tiles"]},
                        key=lambda cr: (cr[0] != 'shared', cr[0], ROLES.index(cr[1])))
    cell_w, cell_h = 88, 82
    left, top = 176, 134
    w = left + len(visits)*cell_w + 26
    h = top + len(identities)*cell_h + 93
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="Lite Braille Fresco timeline for {html.escape(row["root_id"], quote=True)}">',
             f'<rect width="{w}" height="{h}" fill="#272d30"/>',
             f'<rect x="12" y="14" width="{w-24}" height="{h-28}" fill="#b8aa8b" rx="8"/>',
             '<text x="30" y="53" font-family="sans-serif" font-size="21" font-weight="bold" fill="#292d2a">LITE · BRAILLE · FRESCO</text>',
             f'<text x="30" y="83" font-family="monospace" font-size="12" fill="#454238">{html.escape(row["root_id"])} | {html.escape(row["source"]["kind"])}</text>']
    for vi, v in enumerate(visits):
        x = left + vi * cell_w
        parts.append(f'<text x="{x+3}" y="117" font-family="monospace" font-size="12" fill="#343b39">a{v["attempt"]} · p{v["pass"]}</text>')
    map_tiles = [{(t["candidate_id"],unglyph(t["glyph"])[0]):t for t in v["tiles"]} for v in visits]
    braille_positions = ((-11,-19),(-11,-7),(-11,5),(11,-19),(11,-7),(11,5),(-11,17),(11,17))
    for ri, (candidate, role) in enumerate(identities):
        y = top + ri*cell_h
        parts.append(f'<text x="23" y="{y+36}" font-size="12" font-family="monospace" fill="#272b27">{html.escape(candidate[:13])} · {html.escape(role)}</text>')
        for vi, tile_map in enumerate(map_tiles):
            x = left + vi*cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="77" height="71" rx="4" fill="#e6d8b9" fill-opacity=".8" stroke="#9a8d74"/>')
            t = tile_map.get((candidate,role))
            if t is None:
                continue
            frame = unglyph(t["glyph"])[1]
            pol = unglyph(t["glyph"])[2]
            parts.append(_frame_svg(frame,x+5,y+1))
            fill = '#49a9a8' if pol == 0 else '#94b74c' if pol == 1 else '#bc6f50'
            mask = ord(t["glyph"]) - 0x2800
            intensity = [0.13,.38,.68,1.0][t["light"]]
            for j,(dx,dy) in enumerate(braille_positions):
                on = bool(mask & (1<<j))
                parts.append(f'<circle cx="{x+39+dx}" cy="{y+36+dy}" r="{4.7 if on else 2.4}" fill="{fill if on else "#9a947f"}" opacity="{intensity if on else .3}"/>')
            parts.append(f'<title>{html.escape(candidate)} {html.escape(role)} {html.escape(frame)} light={t["light"]} polarity={pol}</title>')
    parts.append(f'<text x="30" y="{h-35}" font-family="sans-serif" font-size="12" fill="#343b39">Dot code is structural. Light is quantized. Lean verification stays external.</text>')
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


def fresh_dir(path: Path) -> None:
    if path.exists():
        raise ValueError(f"output must be a fresh directory: {path}")
    path.mkdir(parents=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    fx = sub.add_parser("fixture", help="Generate explicitly synthetic paired trace + private labels")
    fx.add_argument("--out", required=True, type=Path)
    fx.add_argument("--roots", type=int, default=40)
    fx.add_argument("--seed", type=int, default=20261123)
    ec = sub.add_parser("encode", help="Drop raw magnitudes; emit only public glyphs")
    ec.add_argument("--observations", type=Path, required=True)
    ec.add_argument("--out", required=True, type=Path)
    re = sub.add_parser("render", help="Draw one static SVG fresco")
    re.add_argument("--encoded", type=Path, required=True)
    re.add_argument("--root-id")
    re.add_argument("--out", type=Path, required=True)
    sc = sub.add_parser("score", help="Evaluate toy codebook policies using private labels")
    sc.add_argument("--encoded", type=Path, required=True)
    sc.add_argument("--labels", type=Path, required=True)
    sc.add_argument("--out", type=Path, required=True)
    sc.add_argument("--seed", type=int, default=17)
    args = p.parse_args(argv)
    if args.command == "fixture":
        fresh_dir(args.out)
        raw, labels = fixture(args.roots, args.seed)
        write_jsonl(args.out/"observations.jsonl", raw)
        write_jsonl(args.out/"labels.jsonl", labels)
        report = {"schema":"lbf.fixture_manifest.v1", "roots":args.roots,"seed":args.seed,
                  "scientific_eligible":False,"lean_calls":0,
                  "observations_sha256":file_sha(args.out/"observations.jsonl"),
                  "labels_sha256":file_sha(args.out/"labels.jsonl")}
        (args.out/"manifest.json").write_text(canonical(report)+"\n",encoding="utf-8")
        print(canonical(report))
    elif args.command == "encode":
        fresh_dir(args.out)
        records = [encode(row) for row in read_jsonl(args.observations)]
        write_jsonl(args.out/"encoded.jsonl", records)
        m = {"schema":"lbf.encode_manifest.v1", "observations_sha256":file_sha(args.observations),
             "encoded_sha256":file_sha(args.out/"encoded.jsonl"),"n":len(records),
             "scientific_eligible":False}
        (args.out/"manifest.json").write_text(canonical(m)+"\n",encoding="utf-8")
        print(canonical(m))
    elif args.command == "render":
        rows = read_jsonl(args.encoded)
        matches = [r for r in rows if args.root_id is None or r["root_id"] == args.root_id]
        if len(matches) < 1:
            raise ValueError("root missing")
        if args.out.exists():
            raise ValueError("fresco output already exists")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(fresco_svg(matches[0]),encoding="utf-8")
        print(args.out)
    elif args.command == "score":
        if args.out.exists():
            raise ValueError("score output already exists")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result = score(read_jsonl(args.encoded),read_jsonl(args.labels),args.seed)
        result["encoded_sha256"] = file_sha(args.encoded)
        result["labels_sha256"] = file_sha(args.labels)
        args.out.write_text(canonical(result)+"\n",encoding="utf-8")
        print(canonical(result))


if __name__ == "__main__":
    main()
