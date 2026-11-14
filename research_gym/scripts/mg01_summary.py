"""Collate RMP-MG-01 trained-cohort outputs against Q9 and SEM-ECHO-01 labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research_gym.scripts.loopscope_toy_tca import spearman  # noqa: E402


def load_arm_record(run_dirs: list[Path], cohort: str, arm: str) -> dict | None:
    for d in run_dirs:
        root = d / cohort / arm
        if root.is_dir():
            rec = {"cohort": cohort, "arm": arm}
            p_res = root / "probes_k8_16" / "results.json"
            if p_res.is_file():
                pdata = json.loads(p_res.read_text())
                rec["checkpoint_sha256"] = pdata.get("checkpoint_sha256")
                for st in pdata.get("results", []):
                    if st.get("shuffled_training_labels"):
                        continue
                    key = f"{st['family']}_{st['mode']}_k{st['target']}"
                    final = [r for r in st["rows"] if r["visit"] == st["target"]]
                    if final:
                        rec[key + "_gain_nats"] = float(final[0]["gain_from_input_only"]["mean_gain_nats"])
                        rec[key + "_test_acc"] = float(final[0]["test_group_equal_accuracy"])
            for obs in ("cursor_logits", "all_token_logits"):
                c_res = root / f"causal_{obs}" / "results.json"
                if c_res.is_file():
                    cdata = json.loads(c_res.read_text())
                    rec.setdefault("checkpoint_sha256", cdata.get("checkpoint_sha256"))
                    rows = cdata.get("rows", [])
                    sens = [float(r["local_direction"]["sensitivity"]) for r in rows]
                    null_dims = [int(r["local_direction"]["nullity"]) for r in rows]
                    opt_flips = [
                        bool(e["future_prediction_changed"])
                        for r in rows
                        for e in r.get("finite_interventions", [])
                        if e["kind"] == "optimized_observation_null" and abs(e["relative_radius"] - 0.01) < 1e-9
                    ]
                    rnd_flips = [
                        bool(e["future_prediction_changed"])
                        for r in rows
                        for e in r.get("finite_interventions", [])
                        if e["kind"] == "random_observation_null" and abs(e["relative_radius"] - 0.01) < 1e-9
                    ]
                    rec[f"sens_{obs}"] = float(np.mean(sens)) if sens else None
                    rec[f"nullity_{obs}"] = float(np.mean(null_dims)) if null_dims else None
                    rec[f"flip1e2_opt_{obs}"] = float(np.mean(opt_flips)) if opt_flips else None
                    rec[f"flip1e2_rnd_{obs}"] = float(np.mean(rnd_flips)) if rnd_flips else None
            return rec
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--q9-dir", type=Path, default=Path("experiments/recurrence_monitorability_v0/loopscope_toy_q9"))
    ap.add_argument("--sem-echo-summary", type=Path,
                    default=Path("experiments/recurrence_monitorability_v0/semantic_echo_q4q5_summary/summary.json"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    sem_by_arm = {}
    if args.sem_echo_summary.is_file():
        sdata = json.loads(args.sem_echo_summary.read_text())
        for row in sdata["arms"]:
            arm = row.get("arm")
            if arm:
                sem_by_arm[arm] = row

    rows = []
    for q9_path in sorted(args.q9_dir.glob("lam*.json")):
        q9 = json.loads(q9_path.read_text())
        arm = q9["arm"]
        cohort = q9["set"]
        rec = load_arm_record(args.runs, cohort, arm)
        if rec is None:
            continue
        if rec.get("checkpoint_sha256") and rec["checkpoint_sha256"] != q9["checkpoint_sha256"]:
            raise ValueError(f"{arm}: checkpoint hash mismatch vs Q9 receipt")
        rec["heldout_9_16"] = float(q9["heldout"])
        rec["f1_over_f8"] = float(q9["f1_over_f8"])
        rec["C_cursor"] = float(q9["C"])
        if arm in sem_by_arm:
            rec["sem_echo_primary"] = sem_by_arm[arm]["primary_item_mean"]
            rec["acc_17_32"] = sem_by_arm[arm]["acc_loops_17_32"]
        rows.append(rec)

    rows.sort(key=lambda r: r["f1_over_f8"])
    paired_channel_reduction = sum(
        1 for r in rows
        if r.get("sens_cursor_logits") is not None and r.get("sens_all_token_logits") is not None
        and r["sens_all_token_logits"] < r["sens_cursor_logits"]
    )
    paired_channel_total = sum(
        1 for r in rows
        if r.get("sens_cursor_logits") is not None and r.get("sens_all_token_logits") is not None
    )

    correlations = {}
    if len(rows) >= 4:
        held = np.array([r["heldout_9_16"] for r in rows])
        for key in ("mlp_separate_k8_gain_nats", "mlp_separate_k16_gain_nats",
                    "mlp_shared_k16_gain_nats", "sens_cursor_logits", "sens_all_token_logits"):
            vals = [r.get(key) for r in rows]
            if all(v is not None for v in vals):
                correlations[key + "_vs_heldout"] = float(spearman(np.array(vals, dtype=float), held))

    summary = {
        "schema_version": "rmp_mg01.summary.v1",
        "evidence_stage": "retrospective_development",
        "evaluated_models": len(rows),
        "all_token_reduces_sensitivity_count": paired_channel_reduction,
        "paired_channel_models": paired_channel_total,
        "spearman_correlations": correlations,
        "models": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    md = [
        "# RMP-MG-01 on the 24 trained Q4/Q5 checkpoints",
        "",
        f"- **Evaluated checkpoints:** {len(rows)}/24 (all SHA-256 hashes verified against Q4/Q5 & Q9 receipts).",
        f"- **Widened observation (`all_token_logits` < `cursor_logits` local sensitivity):** "
        f"{paired_channel_reduction}/{paired_channel_total} checkpoints.",
        "",
        "| arm | set | Q9 held-out | F1/F8 | MLP sep K=8 gain (nats) | MLP sep K=16 gain | MLP shared K=16 gain | sens (cursor) | sens (all-token) | opt flip@1e-2 (cursor / all) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        def fmt(k, p=3):
            v = r.get(k)
            return f"{v:.{p}f}" if v is not None else "—"
        md.append(
            f"| {r['arm']} | {r['cohort']} | {r['heldout_9_16']:.3f} | {r['f1_over_f8']:.4f} | "
            f"{fmt('mlp_separate_k8_gain_nats')} | {fmt('mlp_separate_k16_gain_nats')} | "
            f"{fmt('mlp_shared_k16_gain_nats')} | {fmt('sens_cursor_logits')} | {fmt('sens_all_token_logits')} | "
            f"{fmt('flip1e2_opt_cursor_logits')} / {fmt('flip1e2_opt_all_token_logits')} |"
        )
    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {args.out / 'summary.json'} and {args.out / 'summary.md'} ({len(rows)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
