"""LBF-14 post-hoc continuous recurrent tensor, gate-opening, and frontier-geometry diagnostics.

Reproduces all quantitative tables in LBF_14_MECHANISTIC_AND_FRONTIER_ANALYSIS.md
without touching sealed ordinary/OOD splits or mutating frozen checkpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import gzip
import heapq
import io
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import lbf13_budget_fresco as lbf13
import lbf14_bidirectional_fresco as lbf14


def erank(mat: torch.Tensor) -> float:
    x = mat.detach().float()
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    s = torch.linalg.svdvals(x)
    p = s.square()
    tot = float(p.sum())
    if tot <= 1e-12:
        return 0.0
    p = p / tot
    p = p[p > 1e-12]
    return float(torch.exp(-(p * p.log()).sum()))


def stable_rank(mat: torch.Tensor) -> float:
    x = mat.detach().float()
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    s = torch.linalg.svdvals(x)
    if float(s[0]) <= 1e-12:
        return 0.0
    return float(s.square().sum() / s[0].square())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    denom = float(af.norm() * bf.norm())
    return float((af @ bf) / denom) if denom > 1e-12 else 0.0


def rms(x: torch.Tensor) -> float:
    return float(x.detach().float().square().mean().sqrt())


def summary_stats(vals: list[float]) -> dict:
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    var = sum((v - mean) ** 2 for v in s) / n
    q = lambda p: s[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return {
        "n": n, "mean": mean, "sd": math.sqrt(var),
        "min": s[0], "p25": q(0.25), "median": q(0.50),
        "p75": q(0.75), "p95": q(0.95), "max": s[-1],
    }


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den > 1e-12 else 0.0


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        out = [0.0] * len(vs)
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out
    return pearson(ranks(xs), ranks(ys))


def _load_torch_maybe_gz(path: Path):
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    gz = Path(str(path) + ".gz")
    with gzip.open(gz, "rb") as f:
        return torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)


def forward_intervened(model, goal: str, candidates: list[str], *, checks_remaining: int = 8,
                       mode: str = "normal", ortho_q: torch.Tensor | None = None):
    goal_emb = model.text([goal])[0]
    cand_emb = model.text(list(candidates))
    context = model.context(torch.cat([
        goal_emb, cand_emb.mean(0), goal_emb.new_tensor([checks_remaining / 8.0]),
    ]))
    z = (model.roles.unsqueeze(0) + context[None, None, :]).clone()
    z[:, 0] = z[:, 0] + goal_emb
    z[:, 4] = z[:, 4] + cand_emb.mean(0)
    z0 = model.prelude(model.work_norm(z))
    z = z0
    previous = torch.zeros_like(z)
    outs = []
    for k in range(2):
        msg = previous
        if k == 1:
            if mode == "zero":
                msg = torch.zeros_like(msg)
            elif mode == "roll":
                msg = msg.roll(1, dims=1)
            elif mode == "ortho_rot":
                msg = msg @ ortho_q
        z_in = z0 if (k == 1 and mode == "msg_only_frozen_resid") else z
        a = model.middle_a(z_in + model.gate_a(z_in, msg))
        b = model.middle_b(a + model.gate_b(a, msg))
        z = b
        previous = model.message_norm((a + b) * 0.5)
        read = model.coda(z)
        anchor = read[:, 0, :].expand(len(candidates), -1)
        feat = model.feature(torch.cat((anchor, cand_emb, anchor * cand_emb), -1))
        outs.append({"outcome": model.outcome(feat).squeeze(-1), "state": read})
    return outs


def analyze_lbf14_archive(rg, archive_dir: Path) -> dict:
    torch.set_num_threads(1)
    raw = _load_torch_maybe_gz(archive_dir / "raw_boundary_tensors.pt")
    by_arm = {"standard": [], "opened": []}
    for r in raw:
        by_arm[r["seed"].split(":")[1]].append(r)

    archived_metrics = {}
    for arm, records in by_arm.items():
        ga_p1, ga_p2, gb_p1, gb_p2 = [], [], [], []
        ga_rms_p2, gb_rms_p2, ga_rel_p2, gb_rel_p2 = [], [], [], []
        erank_bank_p1, erank_bank_p2, srank_bank_p1 = [], [], []
        erank_coda_p1, erank_coda_p2, slot_cos_bank_p1 = [], [], []
        disp_coda_tot, disp_coda_res, disp_coda_msg, disp_coda_slot = [], [], [], []
        disp_logit_tot, disp_logit_res, disp_logit_msg, disp_logit_slot = [], [], [], []
        margin_p1, margin_p2_norm, margin_p2_zero, margin_p2_roll = [], [], [], []
        cos_ga_zin, cos_gb_ain, cos_msg_vs_res_coda, cos_bank1_z1 = [], [], [], []
        coda1_anchors, coda2_anchors = [], []

        for r in records:
            nb = r["normal"]
            zb = r["alternatives"]["zero"]["bank"]
            rb = r["alternatives"]["roll"]["bank"]
            u1, u2_norm = r["utility_logits"][0], r["utility_logits"][1]
            u2_zero = r["alternatives"]["zero"]["utility_logits"][1]
            u2_roll = r["alternatives"]["roll"]["utility_logits"][1]

            ga_p1.append(float(nb["gate_a_strength"][0]["out"].sigmoid().mean()))
            ga_p2.append(float(nb["gate_a_strength"][1]["out"].sigmoid().mean()))
            gb_p1.append(float(nb["gate_b_strength"][0]["out"].sigmoid().mean()))
            gb_p2.append(float(nb["gate_b_strength"][1]["out"].sigmoid().mean()))

            ga_out2, gb_out2 = nb["gate_a"][1]["out"], nb["gate_b"][1]["out"]
            z_in2, a_in2 = nb["gate_a"][1]["state_in"], nb["gate_b"][1]["state_in"]
            bank_out1, bank_out2 = nb["message_norm"][0]["out"], nb["message_norm"][1]["out"]
            coda1, coda2_norm = nb["coda"][0]["out"], nb["coda"][1]["out"]
            coda2_zero, coda2_roll = zb["coda"][1]["out"], rb["coda"][1]["out"]

            ga_rms_p2.append(rms(ga_out2))
            gb_rms_p2.append(rms(gb_out2))
            ga_rel_p2.append(float(ga_out2.norm() / z_in2.norm().clamp_min(1e-12)))
            gb_rel_p2.append(float(gb_out2.norm() / a_in2.norm().clamp_min(1e-12)))
            erank_bank_p1.append(erank(bank_out1[0]))
            erank_bank_p2.append(erank(bank_out2[0]))
            srank_bank_p1.append(stable_rank(bank_out1[0]))
            erank_coda_p1.append(erank(coda1[0]))
            erank_coda_p2.append(erank(coda2_norm[0]))

            b0 = F.normalize(bank_out1[0].float(), dim=-1)
            slot_cos_bank_p1.append((float((b0 @ b0.T).sum()) - 8.0) / 56.0)

            dc_tot, dc_res = coda2_norm - coda1, coda2_zero - coda1
            dc_msg, dc_slot = coda2_norm - coda2_zero, coda2_norm - coda2_roll
            disp_coda_tot.append(rms(dc_tot))
            disp_coda_res.append(rms(dc_res))
            disp_coda_msg.append(rms(dc_msg))
            disp_coda_slot.append(rms(dc_slot))

            disp_logit_tot.append(float((u2_norm - u1).abs().max()))
            disp_logit_res.append(float((u2_zero - u1).abs().max()))
            disp_logit_msg.append(float((u2_norm - u2_zero).abs().max()))
            disp_logit_slot.append(float((u2_norm - u2_roll).abs().max()))

            margin_p1.append(float(u1[0].sigmoid() - u1[1].sigmoid()))
            margin_p2_norm.append(float(u2_norm[0].sigmoid() - u2_norm[1].sigmoid()))
            margin_p2_zero.append(float(u2_zero[0].sigmoid() - u2_zero[1].sigmoid()))
            margin_p2_roll.append(float(u2_roll[0].sigmoid() - u2_roll[1].sigmoid()))

            cos_ga_zin.append(cosine(ga_out2, z_in2))
            cos_gb_ain.append(cosine(gb_out2, a_in2))
            cos_msg_vs_res_coda.append(cosine(dc_msg, dc_res))
            cos_bank1_z1.append(cosine(bank_out1, z_in2))
            coda1_anchors.append(coda1[0, 0])
            coda2_anchors.append(coda2_norm[0, 0])

        archived_metrics[arm] = {
            "n_decisions": len(records),
            "gate_a_pass1": summary_stats(ga_p1),
            "gate_a_pass2": summary_stats(ga_p2),
            "gate_b_pass1": summary_stats(gb_p1),
            "gate_b_pass2": summary_stats(gb_p2),
            "gate_a_contrib_rms_pass2": summary_stats(ga_rms_p2),
            "gate_b_contrib_rms_pass2": summary_stats(gb_rms_p2),
            "gate_a_rel_norm_pass2": summary_stats(ga_rel_p2),
            "gate_b_rel_norm_pass2": summary_stats(gb_rel_p2),
            "slot_erank_bank_pass1": summary_stats(erank_bank_p1),
            "slot_erank_bank_pass2": summary_stats(erank_bank_p2),
            "slot_srank_bank_pass1": summary_stats(srank_bank_p1),
            "slot_erank_coda_pass1": summary_stats(erank_coda_p1),
            "slot_erank_coda_pass2": summary_stats(erank_coda_p2),
            "pop_erank_coda_anchor_pass1": erank(torch.stack(coda1_anchors)),
            "pop_erank_coda_anchor_pass2": erank(torch.stack(coda2_anchors)),
            "bank_offdiag_slot_cosine_pass1": summary_stats(slot_cos_bank_p1),
            "coda_rms_disp_total_p2_minus_p1": summary_stats(disp_coda_tot),
            "coda_rms_disp_residual_zero_minus_p1": summary_stats(disp_coda_res),
            "coda_rms_disp_msg_normal_minus_zero": summary_stats(disp_coda_msg),
            "coda_rms_disp_slot_normal_minus_roll": summary_stats(disp_coda_slot),
            "logit_max_disp_total_p2_minus_p1": summary_stats(disp_logit_tot),
            "logit_max_disp_residual_zero_minus_p1": summary_stats(disp_logit_res),
            "logit_max_disp_msg_normal_minus_zero": summary_stats(disp_logit_msg),
            "logit_max_disp_slot_normal_minus_roll": summary_stats(disp_logit_slot),
            "abs_margin_pass1": summary_stats([abs(x) for x in margin_p1]),
            "abs_margin_pass2_normal": summary_stats([abs(x) for x in margin_p2_norm]),
            "abs_margin_sharpening_tot": summary_stats([abs(b) - abs(a) for a, b in zip(margin_p1, margin_p2_norm)]),
            "top_flips_p1_to_p2_normal": sum((a >= 0) != (b >= 0) for a, b in zip(margin_p1, margin_p2_norm)),
            "top_flips_normal_vs_zero": sum((a >= 0) != (b >= 0) for a, b in zip(margin_p2_norm, margin_p2_zero)),
            "top_flips_normal_vs_roll": sum((a >= 0) != (b >= 0) for a, b in zip(margin_p2_norm, margin_p2_roll)),
            "cosine_gate_a_vs_state_in": summary_stats(cos_ga_zin),
            "cosine_gate_b_vs_state_in": summary_stats(cos_gb_ain),
            "cosine_msg_disp_vs_resid_disp": summary_stats(cos_msg_vs_res_coda),
            "cosine_bank1_vs_z1": summary_stats(cos_bank1_z1),
        }

    # Reconstruct LBF-14 calibration cases from registered generator seed 20261209
    cases_dict, _ = rg.generate(lbf14.SEED, lbf14.COUNTS)
    cal_cases = cases_dict["calibration"]
    cal_frames = []
    for c in cal_cases:
        s = lbf14.State.start(c)
        ordinal = 0
        while s.live(lbf14.WORK):
            ordinal += 1
            r = lbf14.WORK - s.work
            p, menu, _, _, _ = lbf14.features(rg, c, s)
            cs = [lbf14.forced(rg, c, s, i)[0] for i in (0, 1)]
            targets = [1.0 - cost / (r + 1) for cost in cs]
            cal_frames.append({"root": c.root, "ordinal": ordinal, "packet": p, "menu": menu,
                               "remaining": r, "cost": cs, "target": targets, "gap": targets[0] - targets[1]})
            lbf14.advance(rg, c, s, lbf14.smaller(s))

    gen = torch.Generator().manual_seed(20261124)
    q_mat, r_mat = torch.linalg.qr(torch.randn(32, 32, generator=gen))
    diag_sign = torch.sign(torch.diag(r_mat))
    diag_sign[diag_sign == 0] = 1.0
    ortho_q = q_mat * diag_sign

    ckpt_stats = {}
    causal_transmission = {}
    for seed in (107, 109):
        for arm in ("standard", "opened"):
            ck = _load_torch_maybe_gz(archive_dir / f"{seed}_{arm}" / "checkpoint.pt")
            m, _ = lbf14.new_model(seed, arm)
            m.load_state_dict(ck["weights"])
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            init_b = -6.0 if arm == "standard" else -2.0
            ckpt_stats[f"{seed}_{arm}"] = {
                "gate_a_bias_mean": float(m.gate_a.gate.bias.detach().mean()),
                "gate_a_bias_shift": float(m.gate_a.gate.bias.detach().mean() - init_b),
                "gate_a_weight_fro": float(m.gate_a.gate.weight.detach().norm()),
                "gate_a_proj_fro": float(m.gate_a.proj.weight.detach().norm()),
                "gate_b_bias_mean": float(m.gate_b.gate.bias.detach().mean()),
                "gate_b_bias_shift": float(m.gate_b.gate.bias.detach().mean() - init_b),
                "gate_b_weight_fro": float(m.gate_b.gate.weight.detach().norm()),
                "gate_b_proj_fro": float(m.gate_b.proj.weight.detach().norm()),
            }
            interventions = ("pass1", "zero", "msg_only_frozen_resid", "roll", "ortho_rot", "normal")
            per_mode = {k: {"mse": [], "pm": [], "tm": [], "nt_ok": [], "sg": []} for k in interventions}
            for fr in cal_frames:
                t = torch.tensor(fr["target"], dtype=torch.float32)
                tm = fr["gap"]
                with torch.no_grad():
                    on = forward_intervened(m, fr["packet"], fr["menu"], mode="normal")
                    oz = forward_intervened(m, fr["packet"], fr["menu"], mode="zero")
                    om = forward_intervened(m, fr["packet"], fr["menu"], mode="msg_only_frozen_resid")
                    oroll = forward_intervened(m, fr["packet"], fr["menu"], mode="roll")
                    oorth = forward_intervened(m, fr["packet"], fr["menu"], mode="ortho_rot", ortho_q=ortho_q)
                preds = {
                    "pass1": on[0]["outcome"].sigmoid(),
                    "zero": oz[1]["outcome"].sigmoid(),
                    "msg_only_frozen_resid": om[1]["outcome"].sigmoid(),
                    "roll": oroll[1]["outcome"].sigmoid(),
                    "ortho_rot": oorth[1]["outcome"].sigmoid(),
                    "normal": on[1]["outcome"].sigmoid(),
                }
                m_p1 = float(preds["pass1"][0] - preds["pass1"][1])
                for k, pv in preds.items():
                    pm = float(pv[0] - pv[1])
                    per_mode[k]["mse"].append(float(F.mse_loss(pv, t)))
                    per_mode[k]["pm"].append(pm)
                    per_mode[k]["tm"].append(tm)
                    if abs(tm) > 1e-12:
                        per_mode[k]["nt_ok"].append(int((pm >= 0) == (tm > 0)))
                        per_mode[k]["sg"].append((1.0 if tm > 0 else -1.0) * (pm - m_p1))
            frame_summary = {}
            for k, d in per_mode.items():
                nt_p = [p for p, tm in zip(d["pm"], d["tm"]) if abs(tm) > 1e-12]
                nt_t = [tm for tm in d["tm"] if abs(tm) > 1e-12]
                frame_summary[k] = {
                    "mse": sum(d["mse"]) / len(d["mse"]),
                    "pearson_all": pearson(d["pm"], d["tm"]),
                    "spearman_all": spearman(d["pm"], d["tm"]),
                    "pearson_nontied": pearson(nt_p, nt_t),
                    "nontied_sign_acc": sum(d["nt_ok"]) / len(d["nt_ok"]),
                    "mean_signed_margin_gain_vs_p1": sum(d["sg"]) / len(d["sg"]),
                }
            closed_loop = {}
            for mode_name in interventions:
                closures = total_work = pen_work = 0
                for c in cal_cases:
                    def rk(_c, p, menu, _xs, _rem, _n, _mn=mode_name):
                        with torch.no_grad():
                            if _mn == "pass1":
                                return forward_intervened(m, p, menu, mode="normal")[0]["outcome"].tolist()
                            return forward_intervened(m, p, menu, mode=_mn, ortho_q=ortho_q)[1]["outcome"].tolist()
                    res = lbf14.run(rg, c, "learned", rk)
                    closures += int(res["closed"])
                    total_work += res["work"]
                    pen_work += res["work"] if res["closed"] else 129
                closed_loop[mode_name] = {"closures": closures, "rate": closures / len(cal_cases),
                                          "work": total_work, "mean_penalized_work": pen_work / len(cal_cases)}
            causal_transmission[f"{seed}_{arm}"] = {"frame_level": frame_summary, "closed_loop_search": closed_loop}

    return {
        "archived_boundary_probes": archived_metrics,
        "checkpoint_parameters": ckpt_stats,
        "calibration_frames_count": len(cal_frames),
        "calibration_nontied_frames": sum(abs(f["gap"]) > 1e-12 for f in cal_frames),
        "calibration_mixed_closure_frames": sum((f["cost"][0] <= f["remaining"]) != (f["cost"][1] <= f["remaining"]) for f in cal_frames),
        "causal_transmission": causal_transmission,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rg = lbf14.base.load_upstream(args.upstream)
    res = analyze_lbf14_archive(rg, args.archive)
    args.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
