"""LS-SAE12 independent standard-library audit.

Re-derives, from the saved receipts only: source admission under both exact
solvers, argmax digits from saved logits, every verified-deduction schedule,
eligibility, both site rules, family assignment, isotropic directions from
their seeds, every per-variant outcome, every per-state contrast and the
registered family gates with their bootstrap. It shares no schedule, outcome
or gate code with the runner. It cannot recompute hidden states, Jacobian
power iterations or unsaved logits; that limitation is part of every report.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from mechinterp.sae_causal_transfer.active_reasoning_v1 import families as F
from mechinterp.sae_causal_transfer.native_hard_task_v1 import families as S7
from mechinterp.sae_causal_transfer.native_withinrun_v1 import families as S11

SCHEMA = "LS-SAE12/active-reasoning-verified-deduction-v1"
SOURCE_SCHEMA = "LS-SAE12/source-only-pool-48-v1"
STEPS = 16
TSTAR = 4
MINREC = 6
NEVER = 17
DOSES = ("0.02", "0.25", "2.00")
PRIMARY = "0.02"
ISO = 8
ISO_BASE = 2026111800000
POWER_BASE = 2026111900000
RULES = ("observable_error", "model_blind_median")
BOOT = {"active_a": 2026111813, "active_b": 2026111814}
NFAM = 12


def first_argmax(row):
    best = 0
    for j in range(1, 9):
        if row[j] > row[best]:
            best = j
    return best + 1


def sched(preds, x, target):
    out = {}
    for c in range(81):
        if x[c]:
            continue
        s = STEPS
        while s >= 1 and preds[s - 1][c] == target[c]:
            s -= 1
        out[c] = s + 1 if s < STEPS else NEVER
    return out


def gauss_unit(seed):
    rng = random.Random(seed)
    v = [rng.gauss(0.0, 1.0) for _ in range(512)]
    n = math.sqrt(sum(a * a for a in v))
    return [a / n for a in v]


def source_audit(source, check_solvers=True):
    if (source["schema"] != SOURCE_SCHEMA or source["external_repo_commit"] != F.COMMIT
            or source["external_tree_sha"] != F.TREE_SHA
            or source["pool_files"] != [list(p) for p in F.POOL_FILES]
            or source["model_inference_executed"] is not False
            or len(source["pool"]) != 48):
        raise ValueError("LS-SAE12 source receipt altered")
    if hashlib.sha256(json.dumps(source["pool"], sort_keys=True, separators=(",", ":")).encode()).hexdigest() != source["pool_fingerprint"]:
        raise ValueError("pool fingerprint mismatch")
    for k, row in enumerate(source["pool"]):
        name, blob = F.POOL_FILES[k]
        raw = bytes(48 + v for v in row["x"])
        if (row["pool_position"] != 20 + k or row["source_file"] != name
                or row["source_git_blob_sha1"] != blob
                or hashlib.sha1(b"blob 81\0" + raw).hexdigest() != blob
                or hashlib.sha256(raw).hexdigest() != row["source_raw_sha256"]
                or row["seed"] != int(name[:-4]) or row["holes"] != row["x"].count(0)):
            raise ValueError(f"pool slot {k} identity changed")
        t = row["target"]
        units = [[9 * r + c for c in range(9)] for r in range(9)] + [[9 * r + c for r in range(9)] for c in range(9)]
        units += [[9 * (br + dr) + bc + dc for dr in range(3) for dc in range(3)] for br in (0, 3, 6) for bc in (0, 3, 6)]
        if any(sorted(t[i] for i in u) != list(range(1, 10)) for u in units) or any(
                p and p != q for p, q in zip(row["x"], t)):
            raise ValueError(f"pool slot {k} target is not a completion")
        if check_solvers:
            a = S11.solve_unique(row["x"])
            b = F.solve_dfs(row["x"])
            if (a["solutions"] != 1 or b["solutions"] != 1 or a["target"] != t or b["target"] != t
                    or a["search_nodes"] != row["solver_algx_search_nodes"] or b["nodes"] != row["solver_dfs_nodes"]):
                raise ValueError(f"pool slot {k} uniqueness or solver receipt changed")
        if S7.visible_structure(row["x"]) != row["visible_structural_difficulty"] or row["visible_structural_difficulty"]["propagation_solved"]:
            raise ValueError(f"pool slot {k} singles-hardness changed")
    if len({tuple(r["x"]) for r in source["pool"]}) != 48 or len({tuple(r["target"]) for r in source["pool"]}) != 48:
        raise ValueError("pool duplicates")
    return True


def prior_disjoint_audit(source, ls_sae11_source=None, ls_sae08_full=None):
    """Recheck that no pool input or solution appears in LS-SAE11 or LS-SAE08."""
    prior = []
    if ls_sae11_source is not None:
        raw = Path(ls_sae11_source).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["prior_cohorts"]["ls_sae11_source_20_sha256"]:
            raise ValueError("LS-SAE11 source receipt differs from the one admission used")
        prior += [r for g in json.loads(raw)["rosters"].values() for r in g]
    if ls_sae08_full is not None:
        raw = Path(ls_sae08_full).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["prior_cohorts"].get("ls_sae08_full_sha256"):
            raise ValueError("LS-SAE08 evidence differs from the one admission used")
        full = json.loads(raw)
        prior += [r for g in ("test_52", "test_54") for r in full["families"][g]["source_roster"]]
    xs = {tuple(r["x"]) for r in prior}
    ts = {tuple(r["target"]) for r in prior}
    for row in source["pool"]:
        if tuple(row["x"]) in xs or tuple(row["target"]) in ts:
            raise ValueError(f"pool puzzle {row['source_file']} overlaps an earlier cohort")
    return len(prior)


def screen_audit(screen, source):
    srcs = {r["pool_position"]: r for r in source["pool"]}
    if [r["pool_position"] for r in screen["rows"]] != [20 + k for k in range(48)]:
        raise ValueError("screening denominator changed")
    eligible = []
    for row in screen["rows"]:
        src = srcs[row["pool_position"]]
        x, t = src["x"], src["target"]
        preds = row["clean_predicted_digits_by_step"]
        if len(preds) != STEPS or len(row["clean_logits_by_step"]) != STEPS:
            raise ValueError("screening steps incomplete")
        for s in range(STEPS):
            if [first_argmax(r) for r in row["clean_logits_by_step"][s]] != preds[s]:
                raise ValueError("screened digits differ from saved logits")
        sc = sched(preds, x, t)
        if {str(c): v for c, v in sc.items()} != row["clean_verified_schedule"]:
            raise ValueError("screening schedule recomputation differs")
        rec = sorted(c for c in sc if preds[TSTAR - 1][c] != t[c] and MINREC <= sc[c] <= STEPS)
        site = min(rec, key=lambda c: (sc[c], c)) if rec else None
        blanks = [i for i in range(81) if x[i] == 0]
        if (rec != row["recovering_errors_at_tstar"] or (site is not None) != row["eligible"]
                or site != row["selected_site"] or blanks[len(blanks) // 2] != row["control_site"]):
            raise ValueError("observable-error eligibility or site rule differs")
        if site is not None:
            eligible.append(row["pool_position"])
    fams = {"active_a": eligible[:NFAM], "active_b": eligible[NFAM:2 * NFAM]}
    if any(screen["families"][f] != fams[f] for f in fams):
        raise ValueError("family assignment differs")
    return fams


def variant_rows(doses):
    rows = [("clean", None, None, None)]
    for arm in ("structural", "logit"):
        rows += [(arm, None, d, s) for d in doses for s in ("plus", "minus")]
    rows += [("isotropic", k, d, s) for k in range(ISO) for d in doses for s in ("plus", "minus")]
    return rows


def state_audit(st, src, scr, rule):
    x, t = src["x"], src["target"]
    doses = DOSES if rule == "observable_error" else (PRIMARY,)
    site = scr["selected_site"] if rule == "observable_error" else scr["control_site"]
    pos = scr["pool_position"]
    if st["site_rule"] != rule or st["site"] != site or st["t_star"] != TSTAR:
        raise ValueError("intervention site or time changed")
    order = [(v["arm"], v["draw"], v["dose"], v["sign"]) for v in st["variants"]]
    if order != variant_rows(doses):
        raise ValueError("variant grid changed")
    d = st["directions"]
    seeds = [POWER_BASE + 1000 * pos + 100 * RULES.index(rule) + k for k in (0, 1)]
    if d["structural"]["init_seed"] != seeds[0] or d["logit"]["init_seed"] != seeds[1]:
        raise ValueError("power-iteration seeds changed")
    for kind in ("structural", "logit"):
        v = d[kind]["vector"]
        if len(v) != 512 or abs(math.sqrt(sum(a * a for a in v)) - 1) > 1e-9:
            raise ValueError("direction not unit 512-d")
    iso_seeds = [ISO_BASE + 1000 * pos + 100 * RULES.index(rule) + k for k in range(ISO)]
    if d["isotropic_seeds"] != iso_seeds:
        raise ValueError("isotropic seeds changed")
    if [gauss_unit(s) for s in iso_seeds] != d["isotropic_vectors"]:
        raise ValueError("isotropic vectors do not regenerate from their seeds")
    preds = st["predicted_digits_by_step"]
    clean_screen = scr["clean_predicted_digits_by_step"]
    for r, p in enumerate(preds):
        if len(p) != STEPS or p[:TSTAR] != clean_screen[:TSTAR]:
            raise ValueError("pre-intervention steps differ from screening")
    for step, arr in st["logits_at_steps"].items():
        for r, lg in enumerate(arr):
            if [first_argmax(row) for row in lg] != preds[r][int(step) - 1]:
                raise ValueError("saved digits differ from saved logits")
    clean = sched(preds[0], x, t)
    blanks = [i for i in range(81) if x[i] == 0]
    for r, p in enumerate(preds):
        e = sched(p, x, t)
        o = st["outcomes"][r]
        exp = {"vdd": sum(abs(e[c] - clean[c]) for c in clean),
               "blocked_recoveries": sum(clean[c] < NEVER and e[c] == NEVER for c in clean),
               "new_recoveries": sum(clean[c] == NEVER and e[c] < NEVER for c in clean),
               "accelerated": sum(e[c] < clean[c] for c in clean),
               "delayed": sum(e[c] > clean[c] for c in clean),
               "final_correct": sum(e[c] < NEVER for c in clean),
               "site_T": e[site], "site_shift": e[site] - clean[site],
               "site_blocked": clean[site] < NEVER and e[site] == NEVER,
               "ACT16_blank_correct": sum(p[15][c] == t[c] for c in blanks)}
        if o != exp:
            raise ValueError(f"variant {r} outcome recomputation differs")
    for dose in doses:
        idx = {arm: [i for i, v in enumerate(order) if v[0] == arm and v[2] == dose] for arm in ("structural", "logit", "isotropic")}
        V = {arm: sum(st["outcomes"][i]["vdd"] for i in ids) / len(ids) for arm, ids in idx.items()}
        s = st["summary"][dose]
        if (not all(abs(s["V"][a] - V[a]) < 1e-12 for a in V)
                or abs(s["D_structural_minus_isotropic"] - (V["structural"] - V["isotropic"])) > 1e-12
                or abs(s["D_logit_minus_isotropic"] - (V["logit"] - V["isotropic"])) > 1e-12
                or abs(s["D_structural_minus_logit"] - (V["structural"] - V["logit"])) > 1e-12):
            raise ValueError("per-state contrast differs")
    return True


def gate(diffs, family):
    if len(diffs) != NFAM:
        return {"complete": False, "registered_success": None}
    rng = random.Random(BOOT[family])
    boot = sorted(sum(diffs[rng.randrange(NFAM)] for _ in range(NFAM)) / NFAM for _ in range(2000))
    events = sum(v > 0 for v in diffs)
    return {"events_positive": events, "bootstrap95": [boot[50], boot[1949]],
            "registered_success": bool(events >= 9 and boot[50] > 0)}


def intervene_audit(report, screen, source):
    fams = screen_audit(screen, source)
    family = report["family"]
    if report["schema"] != SCHEMA or report["stage"] != "intervene" or report["primary_dose"] != PRIMARY:
        raise ValueError("intervention receipt altered")
    positions = [st["selected"]["pool_position"] for st in report["states"]]
    if positions != fams[family]:
        raise ValueError("family denominator changed")
    srcs = {r["pool_position"]: r for r in source["pool"]}
    scrs = {r["pool_position"]: r for r in screen["rows"]}
    for st in report["states"]:
        p = st["selected"]["pool_position"]
        state_audit(st["selected"], srcs[p], scrs[p], "observable_error")
        state_audit(st["control"], srcs[p], scrs[p], "model_blind_median")
    diffs = [st["selected"]["summary"][PRIMARY]["D_structural_minus_isotropic"] for st in report["states"]]
    g = gate(diffs, family)
    saved = report["statistics"][f"selected@{PRIMARY}@D_structural_minus_isotropic"]
    if saved.get("registered_success") != g["registered_success"] or (
            g.get("bootstrap95") and saved["bootstrap95"] != g["bootstrap95"]):
        raise ValueError("registered family gate differs")
    return {"schema": "LS-SAE12/independent-stdlib-audit-v1", "family": family, "states": len(positions),
            "registered_family_success": g["registered_success"], "primary": g,
            "limitation": "Recomputed from saved digits, logits, seeds and vectors; hidden states and Jacobian power iterations are not re-executed."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--screen", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--ls-sae11-source", type=Path)
    p.add_argument("--ls-sae08-full", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    source = json.loads(a.source.read_text())
    source_audit(source)
    result = {"schema": "LS-SAE12/independent-stdlib-audit-v1", "source_pass": True}
    if a.ls_sae11_source or a.ls_sae08_full:
        result["prior_puzzles_checked_disjoint"] = prior_disjoint_audit(source, a.ls_sae11_source, a.ls_sae08_full)
    if a.screen:
        screen = json.loads(a.screen.read_text())
        result["families"] = screen_audit(screen, source)
        result["screen_pass"] = True
        if a.input:
            result.update(intervene_audit(json.loads(a.input.read_text()), screen, source))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes((json.dumps(result, sort_keys=True, indent=2) + "\n").encode())
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
