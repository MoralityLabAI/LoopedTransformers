"""LS-SAE12 frozen source pool, two independent exact solvers and source admission.

The pool is the next 48 files (codepoint order positions 20..67) of the public
Champagne collection at the commit LS-SAE11 pinned; positions 0..19 are the
LS-SAE11 pilot and hard cohorts. File names and Git blob SHAs below were read
from the pinned tree listing before any puzzle content was fetched. No model,
prediction or intervention result enters this module.
"""
from __future__ import annotations
import hashlib
import json
import urllib.request

from mechinterp.sae_causal_transfer.native_hard_task_v1 import families as S7
from mechinterp.sae_causal_transfer.native_trm_v1 import families as S2
from mechinterp.sae_causal_transfer.native_withinrun_v1 import families as S11

REPO = S11.REPO
COMMIT = S11.COMMIT
ROOT = S11.ROOT
TREE_SHA = "02ec0254a90be989b4253571341245aed3ed48a4"
POOL_START = 20
POOL_FILES = (
    ("104087.txt", "46cfbadfc3abe27fd0a781df6863ebf698bd6e5d"),
    ("104303.txt", "989cc68e5dae2c49e7bb12f309d9c501b1118568"),
    ("104520.txt", "a2d22fc0fd0dc316d66af1a8af00c8aa9399339d"),
    ("104738.txt", "44af7128a1810bb6828f1a19339e9ffce5f4b8a3"),
    ("104951.txt", "9b02655ab860ceb5e45ea1e2dff7b43a61c3a95e"),
    ("1050.txt", "c9f5fdac8fcd0f709fe3d2b68dabf1fabe94db4d"),
    ("10500.txt", "741eb3160bbd57b0013aa4a3608028d5f5d8e286"),
    ("105168.txt", "879b2e7b7b414799b1fd4563136dbe9b20efcb97"),
    ("105386.txt", "1622d479574b7f9dafbbf876c93ff9913391d42b"),
    ("105645.txt", "c3a839b4b20e95492e737c01089d3e17c175b6e6"),
    ("105890.txt", "dc0cdeab67f70b77f14f2aee8617db2021e7868c"),
    ("106126.txt", "592ac1c2417de949ce952257ed20b434c14a32bf"),
    ("106361.txt", "c049d905dbbcc44bbfaaeee7e6087145fd4504db"),
    ("106590.txt", "ed034b59d0106afd13e60011278d2e7767bd63bf"),
    ("106813.txt", "2c4f4d873e9389f701a71effde4f717009a0bdb1"),
    ("107037.txt", "e0040425c2fe39723d6c43eacfc2406e665969e8"),
    ("10710.txt", "46ff2cb0ab7d8baf06708c5253f48b4b2b876c5f"),
    ("107263.txt", "b14bf27885e4216cd2d8309e011a94a6e1f6fad4"),
    ("107496.txt", "d9fc4c3a0bd53aea9fac64d8d69d07a425b0f92e"),
    ("107715.txt", "7a9cf019d3194ee34b22057a4855ab458c54bbcb"),
    ("107943.txt", "e082c126993977e2606873b6e7cc6f96f21b1411"),
    ("108168.txt", "a890ae0126d67426b9f25dd33c5b62b76f52404b"),
    ("108388.txt", "a9b7b51d77234e4183e48df89dc3127bbd1da1a9"),
    ("108613.txt", "41550591cb39ea0d04c33e30e4b2a7672718d6bf"),
    ("108847.txt", "5a8fa1c2a6b1f8134ebd2d38066b34d209ec4aa9"),
    ("109068.txt", "749f6b8baa1dce7c13a57599a2dc008d176a129d"),
    ("10920.txt", "67372cff0a2e212bf4eacec9529f747aa71f7176"),
    ("109293.txt", "aed62fef6adc3781a23cac1be6d1c44ccbf48da4"),
    ("109509.txt", "7a0d9c4fbeb3bdc209274d6b9386c3dac3f44a1a"),
    ("109726.txt", "e09ab85de7ddaaaead8c4171fa917824f1b36639"),
    ("109942.txt", "fb99b745353105f08b82b6f886ca633d6de69c62"),
    ("110174.txt", "0bf573acd7e2a5c49e1698cc3187a8eceb446c0a"),
    ("110393.txt", "6aea7b580095e1319e23d45925b88672b62b7015"),
    ("110628.txt", "66ae1e3618872ede0012458c094681bf23865e4d"),
    ("110842.txt", "e7c2931757fdebee532181288634c4f171c09652"),
    ("111063.txt", "a87633e63430f8c3792086a8e61882b790727029"),
    ("111281.txt", "3d18514aea2c2a19b1e432c8db6f15850895d30f"),
    ("11130.txt", "8664732663da6ff1a5d28b4281ab625cd6be92c9"),
    ("111517.txt", "4e3a8e942bbee616f23b7426a90c0683f691ccf1"),
    ("111762.txt", "0e7c25610c0afbf82eca304f0577ffb40b166872"),
    ("111977.txt", "764a804cd249df3872d79ed1f5b328776b36f678"),
    ("112207.txt", "d6cd2570506c8e6abc8651a42f920d35d6099667"),
    ("112420.txt", "4720be8d98b8630375928a25ebdc7be7b00de02b"),
    ("112641.txt", "329ab91695b213b10ffb2de1a96b7d26adabcb9c"),
    ("112884.txt", "bdab96f38adc61ce67bdf0639ce2812dffeeda92"),
    ("113110.txt", "214e8d24941a6ee5037857345f3d593c98784d55"),
    ("113329.txt", "b11d1fa647c6168f8a73df1ad4b7086a63e66775"),
    ("11340.txt", "f35385b7b1e2dd655cca70fbd86ac32527dc9b4b"),
)
POOL_SIZE = 48
# LS-SAE11 engineering-pilot files reused here for engineering only.
PILOT_FILES = S11.SOURCE_FILES[:4]

UNITS = [[9 * r + c for c in range(9)] for r in range(9)]
UNITS += [[9 * r + c for r in range(9)] for c in range(9)]
UNITS += [[9 * (br + dr) + bc + dc for dr in range(3) for dc in range(3)]
          for br in (0, 3, 6) for bc in (0, 3, 6)]
PEERS = [sorted({j for u in UNITS if i in u for j in u} - {i}) for i in range(81)]


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def fetch(name, digest, files):
    if (name, digest) not in files:
        raise ValueError("unregistered LS-SAE12 source file requested")
    url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{ROOT}/{name}"
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read(82)
    return parse(name, digest, raw), raw


def parse(name, digest, raw):
    if len(raw) != 81 or git_blob_sha(raw) != digest:
        raise ValueError(f"Champagne blob/length mismatch: {name}")
    s = raw.decode("ascii")
    if any(ch not in "0123456789" for ch in s):
        raise ValueError(f"non-digit source content: {name}")
    return [int(ch) for ch in s]


def solve_dfs(grid, max_nodes=2_000_000):
    """Independent second solver: bitmask candidates, MRV cell choice, count to two.

    Written separately from LS-SAE11's 324-column Algorithm X; shares no code with it.
    """
    if len(grid) != 81 or any(type(v) is not int or not 0 <= v <= 9 for v in grid):
        raise ValueError("invalid Sudoku grid")
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    cells = list(grid)
    for i, v in enumerate(cells):
        if v:
            r, c = divmod(i, 9)
            b = 3 * (r // 3) + c // 3
            bit = 1 << v
            if rows[r] & bit or cols[c] & bit or boxes[b] & bit:
                return {"solutions": 0, "target": None, "nodes": 0}
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
    solutions = []
    nodes = 0
    full = 0x3FE

    def search():
        nonlocal nodes
        best = -1
        best_mask = 0
        best_count = 10
        for i in range(81):
            if cells[i] == 0:
                r, c = divmod(i, 9)
                mask = full & ~(rows[r] | cols[c] | boxes[3 * (r // 3) + c // 3])
                n = bin(mask).count("1")
                if n < best_count:
                    best, best_mask, best_count = i, mask, n
                    if n <= 1:
                        break
        if best < 0:
            solutions.append(list(cells))
            return
        if best_count == 0:
            return
        r, c = divmod(best, 9)
        b = 3 * (r // 3) + c // 3
        for v in range(1, 10):
            bit = 1 << v
            if not best_mask & bit:
                continue
            nodes += 1
            if nodes > max_nodes:
                raise RuntimeError("independent solver node cap exceeded")
            cells[best] = v
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            search()
            rows[r] &= ~bit
            cols[c] &= ~bit
            boxes[b] &= ~bit
            cells[best] = 0
            if len(solutions) >= 2:
                return

    search()
    return {"solutions": len(solutions), "target": solutions[0] if len(solutions) == 1 else None,
            "nodes": nodes}


def valid_complete(target, puzzle):
    if len(target) != 81 or any(not 1 <= v <= 9 for v in target):
        return False
    if any(len({target[i] for i in u}) != 9 for u in UNITS):
        return False
    return all(p == 0 or p == t for p, t in zip(puzzle, target))


def admit_one(index, name, digest, x, raw):
    a = S11.solve_unique(x)
    b = solve_dfs(x)
    if a["solutions"] != 1 or b["solutions"] != 1:
        raise ValueError(f"source admission failed: {name} solutions algX={a['solutions']} dfs={b['solutions']}")
    if a["target"] != b["target"]:
        raise ValueError(f"source admission failed: {name} independent solvers disagree")
    target = a["target"]
    if not valid_complete(target, x) or not S2.valid_solution(target):
        raise ValueError(f"source admission failed: {name} target invalid")
    vis = S7.visible_structure(x)
    if vis["propagation_solved"] or vis["deduced_missing_cells"] >= x.count(0):
        raise ValueError(f"source admission failed: {name} solved by singles")
    return {"pool_position": index, "source_file": name, "source_git_blob_sha1": digest,
            "source_raw_sha256": hashlib.sha256(raw).hexdigest(),
            "seed": int(name.removesuffix(".txt")), "x": x, "target": target,
            "holes": x.count(0), "visible_structural_difficulty": vis,
            "solver_algx_search_nodes": a["search_nodes"], "solver_dfs_nodes": b["nodes"],
            "source_repository_commit": COMMIT}


def fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_disjoint(rows, prior_inputs, prior_targets):
    xs = [tuple(r["x"]) for r in rows]
    ts = [tuple(r["target"]) for r in rows]
    if len(set(xs)) != len(xs) or len(set(ts)) != len(ts):
        raise ValueError("LS-SAE12 pool duplicates a puzzle or solution internally")
    if set(xs) & set(prior_inputs) or set(ts) & set(prior_targets):
        raise ValueError("LS-SAE12 pool overlaps LS-SAE08/LS-SAE11 puzzles or solutions")
    return True
