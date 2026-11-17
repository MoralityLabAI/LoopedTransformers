"""LS-SAE12 native screening and mid-reasoning interventions on the pretrained TRM.

Stages (each writes JSON before the independent stdlib auditor reads it):
  pilot      engineering only: LS-SAE11 pilot puzzles, screening + interventions, no gate
  screen     clean ACT1..ACT16 trajectories of the 48-puzzle pool, registered
             observable-error eligibility, site selection and family assignment
  intervene  one family: structural / logit-sensitive / isotropic edits of the
             site z_H token after ACT4, native continuation to ACT16 in one batch

Directions are computed on each clean ACT4 state without labels. Exact-solver
targets enter only the verified-deduction schedule, after every edit has run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from mechinterp.loopscope02.scripts import trm_sudoku as TRM
from mechinterp.sae_causal_transfer.native_task_transfer_v1 import run_native as BASE
from mechinterp.sae_causal_transfer.native_withinrun_v1 import run_native as RUN11
from mechinterp.sae_causal_transfer.active_reasoning_v1 import families as F
from mechinterp.sae_causal_transfer.active_reasoning_v1 import schedule as S

SCHEMA = "LS-SAE12/active-reasoning-verified-deduction-v1"
SOURCE_SCHEMA = "LS-SAE12/source-only-pool-48-v1"
DOSES = ("0.02", "0.25", "2.00")
PRIMARY_DOSE = "0.02"
POWER_ITERS = 24
LOGIT_STEPS = (5, 16)
TRANSPORT_STEPS = (5, 8, 16)
ARMS = ("structural", "logit", "isotropic")
SIGNS = (("plus", 1.0), ("minus", -1.0))


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_model():
    torch.set_num_threads(4)
    torch.manual_seed(71)
    model, digest = TRM.load_trm_checkpoint("gaoxin492", "cpu", dtype=torch.float32)
    if digest != BASE.MODEL_SHA:
        raise ValueError("original pretrained checkpoint SHA mismatch")
    model.eval().requires_grad_(False)
    return model, digest


def platform_fingerprint():
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as fh:
            cpu = next(line.split(":", 1)[1].strip() for line in fh if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "cpu": cpu, "torch_threads": torch.get_num_threads()}


def question(x):
    return "".join("." if v == 0 else str(v) for v in x)


def argmax_digits(logits):
    """First-index argmax over digits 1..9 (native tie rule)."""
    return (np.argmax(logits, axis=-1) + 1).tolist()


def clean_trajectory(model, x):
    """Batch-1 clean run to ACT16; returns per-step predictions/logits and the ACT4 anchor."""
    preds, logits, anchor = [], [], None
    with torch.no_grad(), sdpa_kernel(backends=[SDPBackend.MATH]):
        inp = TRM.encode_questions([question(x)], "cpu")
        x_emb = model.encode_inputs(inp)
        zh, zl = model.init_states(1)
        for step in range(1, S.N_STEPS + 1):
            zh, zl = model.act_step(zh, zl, x_emb)
            arr = model.digit_logits(zh).double().cpu().numpy()[0]
            if arr.shape != (81, 9) or not np.isfinite(arr).all():
                raise ArithmeticError("nonfinite clean logits")
            preds.append(argmax_digits(arr))
            logits.append(arr.tolist())
            if step == S.T_STAR:
                anchor = {"zh": zh.detach().clone(), "zl": zl.detach().clone(), "x_emb": x_emb.detach().clone()}
    return preds, logits, anchor


def screen_row(model, src, position):
    preds, logits, _ = clean_trajectory(model, src["x"])
    site, rec, sched = S.select_site(preds, src["x"], src["target"])
    return {
        "pool_position": position, "source_file": src.get("source_file"), "seed": src["seed"],
        "clean_predicted_digits_by_step": preds, "clean_logits_by_step": logits,
        "clean_verified_schedule": {str(c): t for c, t in sched.items()},
        "recovering_errors_at_tstar": rec, "eligible": site is not None,
        "selected_site": site, "control_site": S.control_site(src["x"]),
        "clean_blank_correct_by_step": [sum(p[c] == src["target"][c] for c in S.blanks(src["x"])) for p in preds],
    }


def top_singular_direction(model, anchor, token, kind, seed, iters=POWER_ITERS):
    """Power iteration on J^T J for J = d(output)/d(z_H[site] after ACT4), output after ACT5.

    kind "structural": output = all 97x512 z_H; kind "logit": centered 81x9 digit logits.
    J u is obtained by double backward through the linear map w -> J^T w.
    """
    init = torch.tensor(S.gaussian_unit(seed), dtype=torch.float32)
    with torch.enable_grad(), sdpa_kernel(backends=[SDPBackend.MATH]):
        v0 = torch.zeros(512, dtype=torch.float32, requires_grad=True)
        zh = anchor["zh"].clone()
        delta = torch.zeros_like(zh)
        delta[0, token] = v0
        out_h, _ = model.act_step(zh + delta, anchor["zl"], anchor["x_emb"])
        if kind == "structural":
            out = out_h.reshape(-1)
        elif kind == "logit":
            lg = model.digit_logits(out_h)
            out = (lg - lg.mean(dim=-1, keepdim=True)).reshape(-1)
        else:
            raise ValueError("unknown direction kind")
        w = torch.zeros_like(out, requires_grad=True)
        jt_w = torch.autograd.grad(out, v0, grad_outputs=w, create_graph=True)[0]
        u = init.clone()
        history = []
        for _ in range(iters):
            ju = torch.autograd.grad(jt_w, w, grad_outputs=u, retain_graph=True)[0]
            jtju = torch.autograd.grad(out, v0, grad_outputs=ju, retain_graph=True)[0]
            history.append(float(torch.linalg.vector_norm(ju)))
            norm = torch.linalg.vector_norm(jtju)
            if not torch.isfinite(norm) or norm <= 0:
                raise ArithmeticError("power iteration degenerated")
            u = (jtju / norm).detach()
        ju = torch.autograd.grad(jt_w, w, grad_outputs=u, retain_graph=False)[0]
        sigma = float(torch.linalg.vector_norm(ju))
    vec = u.double().numpy()
    k = int(np.argmax(np.abs(vec)))
    if vec[k] < 0:
        vec = -vec
    vec = vec / np.linalg.norm(vec)
    return vec, {"sigma_estimate": sigma, "power_iterations": iters, "init_seed": seed,
                 "jv_norm_history": history,
                 "relative_change_last": abs(history[-1] - history[-2]) / max(history[-1], 1e-30)}


def variant_order(doses):
    order = [{"arm": "clean", "draw": None, "dose": None, "sign": None}]
    for arm in ("structural", "logit"):
        for dose in doses:
            for sign, _ in SIGNS:
                order.append({"arm": arm, "draw": None, "dose": dose, "sign": sign})
    for k in range(S.ISO_DRAWS):
        for dose in doses:
            for sign, _ in SIGNS:
                order.append({"arm": "isotropic", "draw": k, "dose": dose, "sign": sign})
    return order


def transport(dh, dl, base_logits, edited_logits, x, site, dose, q):
    board = dh[16:, :]
    full_h = float(np.linalg.norm(dh))
    board_norm = float(np.linalg.norm(board))
    row = float(np.linalg.norm(board @ q))
    centered = edited_logits - base_logits
    centered = centered - centered.mean(axis=1, keepdims=True)
    cert = RUN11.certificates(base_logits, edited_logits, np.flatnonzero(np.asarray(x) == 0))
    return {"h_gain": full_h / dose, "l_gain": float(np.linalg.norm(dl)) / dose,
            "h_site_share": float(np.linalg.norm(dh[16 + site])) / full_h if full_h > 0 else 0.0,
            "readout_fraction": row / board_norm if board_norm > 1e-12 else 0.0,
            "centered_logit_fro": float(np.linalg.norm(centered)),
            "strict_global_margin_certificate": cert["strict_global_margin_certificate"],
            "minimum_clean_blank_winner_margin": cert["minimum_clean_blank_winner_margin"],
            "all81_absolute_epsilon": cert["all81_absolute_epsilon"],
            "actual_blank_digit_flips": cert["actual_blank_digit_flips"]}


def run_state(model, src, screened, site_rule, q):
    x, target = src["x"], src["target"]
    site = screened["selected_site"] if site_rule == "observable_error" else screened["control_site"]
    doses = DOSES if site_rule == "observable_error" else (PRIMARY_DOSE,)
    position = screened["pool_position"]
    preds_clean_screen, _, anchor = clean_trajectory(model, x)
    if preds_clean_screen != screened["clean_predicted_digits_by_step"]:
        raise ArithmeticError("batch-1 clean trajectory does not reproduce the screening receipt")
    token = model.puzzle_emb_len + site
    dirs = {}
    meta = {}
    for kind in ("structural", "logit"):
        seed = S.direction_seed(S.POWER_SEED_BASE, position, site_rule, 0 if kind == "structural" else 1)
        dirs[kind], meta[kind] = top_singular_direction(model, anchor, token, kind, seed)
    iso_seeds = [S.direction_seed(S.ISO_SEED_BASE, position, site_rule, k) for k in range(S.ISO_DRAWS)]
    iso = [np.asarray(S.gaussian_unit(s), dtype=np.float64) for s in iso_seeds]
    order = variant_order(doses)
    n = len(order)
    blanks = S.blanks(x)
    preds = [[None] * S.N_STEPS for _ in range(n)]
    for r in range(n):
        for s in range(S.T_STAR):
            preds[r][s] = preds_clean_screen[s]
    logits_saved = {str(s): [] for s in LOGIT_STEPS}
    transport_saved = {str(s): [None] * n for s in TRANSPORT_STEPS}
    with torch.no_grad(), sdpa_kernel(backends=[SDPBackend.MATH]):
        zh = anchor["zh"].repeat(n, 1, 1)
        zl = anchor["zl"].repeat(n, 1, 1)
        x_emb = anchor["x_emb"].repeat(n, 1, 1)
        for r, v in enumerate(order[1:], 1):
            vec = dirs[v["arm"]] if v["arm"] != "isotropic" else iso[v["draw"]]
            s = 1.0 if v["sign"] == "plus" else -1.0
            zh[r, token] += torch.tensor(float(v["dose"]) * s * vec, dtype=zh.dtype)
        for step in range(S.T_STAR + 1, S.N_STEPS + 1):
            zh, zl = model.act_step(zh, zl, x_emb)
            arr = model.digit_logits(zh).double().cpu().numpy()
            if arr.shape != (n, 81, 9) or not np.isfinite(arr).all():
                raise ArithmeticError("nonfinite continuation logits")
            for r in range(n):
                preds[r][step - 1] = argmax_digits(arr[r])
            if step in LOGIT_STEPS:
                logits_saved[str(step)] = arr.tolist()
            if step in TRANSPORT_STEPS:
                h0 = zh[0].double().cpu().numpy()
                l0 = zl[0].double().cpu().numpy()
                for r in range(1, n):
                    dh = zh[r].double().cpu().numpy() - h0
                    dl = zl[r].double().cpu().numpy() - l0
                    transport_saved[str(step)][r] = transport(dh, dl, arr[0], arr[r], x, site,
                                                              float(order[r]["dose"]), q)
    clean_sched = S.verified_schedule(preds[0], x, target)
    screen_sched = {int(k): v for k, v in screened["clean_verified_schedule"].items()}
    outcomes = []
    for r in range(n):
        sched = S.verified_schedule(preds[r], x, target)
        out = S.deduction_change(clean_sched, sched, site)
        out["ACT16_blank_correct"] = sum(preds[r][15][c] == target[c] for c in blanks)
        outcomes.append(out)
    summary = summarize(order, outcomes, transport_saved, doses)
    return {
        "pool_position": position, "source_file": src.get("source_file"), "seed": src["seed"],
        "site_rule": site_rule, "site": site, "t_star": S.T_STAR,
        "site_clean_T_screen": screen_sched[site], "site_clean_T_same_forward": clean_sched[site],
        "same_forward_clean_matches_screen_steps": [preds[0][s] == preds_clean_screen[s] for s in range(S.N_STEPS)],
        "directions": {"structural": {"vector": dirs["structural"].tolist(), **meta["structural"]},
                       "logit": {"vector": dirs["logit"].tolist(), **meta["logit"]},
                       "isotropic_seeds": iso_seeds,
                       "isotropic_vectors": [v.tolist() for v in iso],
                       "cosine_structural_logit": float(dirs["structural"] @ dirs["logit"])},
        "variants": order,
        "predicted_digits_by_step": preds,
        "logits_at_steps": logits_saved,
        "transport_at_steps": transport_saved,
        "clean_verified_schedule": {str(c): t for c, t in clean_sched.items()},
        "outcomes": outcomes,
        "summary": summary,
    }


def summarize(order, outcomes, transport_saved, doses):
    out = {}
    for dose in doses:
        rows = {arm: [i for i, v in enumerate(order) if v["arm"] == arm and v["dose"] == dose] for arm in ARMS}
        mean = lambda idx, key: sum(float(outcomes[i][key]) for i in idx) / len(idx)
        V = {arm: mean(rows[arm], "vdd") for arm in ARMS}
        d = {"V": V, "D_structural_minus_isotropic": V["structural"] - V["isotropic"],
             "D_logit_minus_isotropic": V["logit"] - V["isotropic"],
             "D_structural_minus_logit": V["structural"] - V["logit"]}
        for key in ("blocked_recoveries", "new_recoveries", "accelerated", "delayed",
                    "ACT16_blank_correct", "site_shift"):
            d[key] = {arm: mean(rows[arm], key) for arm in ARMS}
        d["site_blocked_rate"] = {arm: mean(rows[arm], "site_blocked") for arm in ARMS}
        d["h_gain"] = {str(s): {arm: sum(transport_saved[str(s)][i]["h_gain"] for i in rows[arm]) / len(rows[arm])
                                for arm in ARMS} for s in TRANSPORT_STEPS}
        out[dose] = d
    return out


def load_source(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if (data["schema"] != SOURCE_SCHEMA or data["external_repo_commit"] != F.COMMIT
            or data["pool_files"] != [list(p) for p in F.POOL_FILES]
            or data["model_inference_executed"] is not False
            or F.fingerprint(data["pool"]) != data["pool_fingerprint"]):
        raise ValueError("LS-SAE12 source-only receipt altered")
    return data, sha_bytes(raw)


def readout_q(model):
    _, q, values = RUN11.readout(model)
    return q, values


def run_screen(source_path, out, model=None, digest=None):
    source, source_sha = load_source(source_path)
    if model is None:
        model, digest = load_model()
    rows = []
    for src in source["pool"]:
        row = screen_row(model, src, src["pool_position"])
        rows.append(row)
        print("LS_SAE12_SCREEN", src["pool_position"], src["source_file"], "eligible", row["eligible"],
              "site", row["selected_site"], "recovering", len(row["recovering_errors_at_tstar"]),
              "ACT4/ACT16 correct", row["clean_blank_correct_by_step"][3], row["clean_blank_correct_by_step"][15],
              "/", src["holes"], flush=True)
    fams = S.assign_families([r["pool_position"] for r in rows if r["eligible"]])
    result = {"schema": SCHEMA, "stage": "screen", "source_receipt_sha256": source_sha,
              "checkpoint_sha256": digest, "platform": platform_fingerprint(), "t_star": S.T_STAR, "min_recovery_step": S.MIN_RECOVERY_STEP,
              "rows": rows, "families": fams,
              "eligible_count": sum(r["eligible"] for r in rows)}
    write(out, result)
    return result


def run_intervene(source_path, screen_path, family, out, model=None, digest=None):
    source, source_sha = load_source(source_path)
    screen_raw = Path(screen_path).read_bytes()
    screen = json.loads(screen_raw)
    if screen["stage"] != "screen" or screen["source_receipt_sha256"] != source_sha:
        raise ValueError("screening receipt does not match source receipt")
    if model is None:
        model, digest = load_model()
    q, values = readout_q(model)
    by_pos = {r["pool_position"]: r for r in screen["rows"]}
    src_by_pos = {r["pool_position"]: r for r in source["pool"]}
    states = []
    for position in screen["families"][family]:
        sel = run_state(model, src_by_pos[position], by_pos[position], "observable_error", q)
        ctl = run_state(model, src_by_pos[position], by_pos[position], "model_blind_median", q)
        p = sel["summary"][PRIMARY_DOSE]
        print("LS_SAE12_STATE", family, position, sel["source_file"], "site", sel["site"],
              "V", {k: round(v, 3) for k, v in p["V"].items()},
              "D_SI", round(p["D_structural_minus_isotropic"], 3), flush=True)
        states.append({"selected": sel, "control": ctl})
    stats = family_statistics(states, family)
    result = {"schema": SCHEMA, "stage": "intervene", "family": family,
              "source_receipt_sha256": source_sha, "screen_receipt_sha256": sha_bytes(screen_raw),
              "checkpoint_sha256": digest, "readout_singular_values": values,
              "platform": platform_fingerprint(), "doses": list(DOSES), "primary_dose": PRIMARY_DOSE, "states": states, "statistics": stats}
    write(out, result)
    return result


def family_statistics(states, family):
    out = {}
    for rule, doses in (("selected", DOSES), ("control", (PRIMARY_DOSE,))):
        for dose in doses:
            for contrast in ("D_structural_minus_isotropic", "D_logit_minus_isotropic", "D_structural_minus_logit"):
                diffs = [st[rule]["summary"][dose][contrast] for st in states]
                g = S.family_gate(diffs, family)
                g["registered_primary"] = (rule == "selected" and dose == PRIMARY_DOSE
                                           and contrast == "D_structural_minus_isotropic")
                out[f"{rule}@{dose}@{contrast}"] = g
    primary = out[f"selected@{PRIMARY_DOSE}@D_structural_minus_isotropic"]
    out["registered_family_success"] = primary["registered_success"]
    return out


def run_pilot(source11_path, out):
    data = json.loads(Path(source11_path).read_bytes())
    rows = data["rosters"]["pilot_champagne"]
    if [(r["source_file"], r["source_git_blob_sha1"]) for r in rows] != [tuple(p) for p in F.PILOT_FILES]:
        raise ValueError("LS-SAE11 pilot roster differs")
    model, digest = load_model()
    q, values = readout_q(model)
    cases = []
    for r in rows:
        src = dict(r)
        sc = screen_row(model, src, r["source_index"])
        case = {"screen": {k: v for k, v in sc.items() if k != "clean_logits_by_step"}}
        if sc["eligible"]:
            case["selected"] = run_state(model, src, sc, "observable_error", q)
            case["control"] = run_state(model, src, sc, "model_blind_median", q)
            p = case["selected"]["summary"][PRIMARY_DOSE]
            print("LS_SAE12_PILOT", r["source_file"], "site", sc["selected_site"],
                  "V", {k: round(v, 3) for k, v in p["V"].items()}, flush=True)
        else:
            print("LS_SAE12_PILOT", r["source_file"], "ineligible", flush=True)
        cases.append(case)
    write(out, {"schema": SCHEMA, "stage": "pilot", "engineering_only": True,
                "platform": platform_fingerprint(), "checkpoint_sha256": digest, "readout_singular_values": values, "cases": cases})


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, sort_keys=True, allow_nan=False) + "\n").encode())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("pilot", "screen", "intervene", "full"), required=True)
    p.add_argument("--source", type=Path, help="LS-SAE12 source receipt (screen/intervene) or LS-SAE11 source_20.json (pilot)")
    p.add_argument("--screen", type=Path)
    p.add_argument("--family", choices=S.FAMILIES)
    p.add_argument("--out", type=Path, required=True, help="output file, or directory for --stage full")
    a = p.parse_args()
    if a.stage == "full":
        # Screening and both families in one process on one machine (see PROTOCOL).
        model, digest = load_model()
        run_screen(a.source, a.out / "screen.json", model, digest)
        for fam in S.FAMILIES:
            r = run_intervene(a.source, a.out / "screen.json", fam, a.out / f"{fam}.json", model, digest)
            print(fam, "registered_family_success", r["statistics"]["registered_family_success"], flush=True)
    elif a.stage == "pilot":
        run_pilot(a.source, a.out)
    elif a.stage == "screen":
        r = run_screen(a.source, a.out)
        print(json.dumps({"eligible": r["eligible_count"], "families": r["families"]}, indent=2))
    else:
        r = run_intervene(a.source, a.screen, a.family, a.out)
        print(json.dumps({k: v for k, v in r["statistics"].items() if k == "registered_family_success"}, indent=2))


if __name__ == "__main__":
    main()
