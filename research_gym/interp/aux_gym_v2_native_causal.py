"""Native BPD read swaps on a frozen AUX teacher with matched physical-write controls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch

from research_gym.interp.aux_gym_v2 import digest_file, dump, rows, write_rows
from research_gym.interp.aux_gym_v2_native_bpd import (
    BPD_REVISION, SITE, import_bpd, weight_digest,
)


def component_ids(v, u, rank):
    if not 1 <= rank <= min(v.shape[1], u.shape[0], u.shape[1]):
        raise ValueError("invalid intervention rank")
    strength = v.float().norm(dim=0) * u.float().norm(dim=1)
    return strength.argsort(descending=True, stable=True)[:rank].tolist()


def dose_match(direction, dose, eps=1e-10):
    norm = direction.norm(dim=-1, keepdim=True)
    if float(norm.min()) <= eps:
        return None
    return direction * (dose / norm)


def read_swap_delta(anchor_x, donor_x, v, u):
    return ((donor_x - anchor_x).float() @ v.float()) @ u.float()


def summarize(records):
    out = {"statuses": dict(Counter(r["status"] for r in records)), "strata": {}}
    for kind in ("changed_target", "same_target"):
        for depth in sorted({r["depth"] for r in records if r["kind"] == kind}):
            stratum = [r for r in records if r["kind"] == kind and r["depth"] == depth]
            competent = [r for r in stratum if r["status"] == "evaluated"
                         and r["clean_both_correct"]]
            out["strata"][f"{kind}:{depth}"] = {
                "pairs": len(stratum), "competent_dose_matched": len(competent),
                "final_hits": {arm: sum(r["arms"][arm]["final_hit"] for r in competent)
                               for arm in ("native", "svd", "random")},
                "local_hits": {arm: sum(r["arms"][arm]["local_hit"] for r in competent)
                               for arm in ("native", "svd", "random")},
            }
    return out


@torch.no_grad()
def evaluate(args):
    from research_gym.interp.models import build_model
    if args.out.exists() or args.rank < 2 or args.visits < 1 or args.threads < 1:
        raise ValueError("fresh output, rank >= 2, positive visits and threads required")
    if args.split.startswith("locked") and not args.unlock:
        raise ValueError("locked split requires explicit --unlock")
    torch.set_num_threads(args.threads)
    bpd = import_bpd(args.bpd_checkout)
    teacher = json.loads((args.run / "receipt.json").read_text())
    factor = json.loads((args.factors / "receipt.json").read_text())
    pair_manifest = json.loads((args.pairs / "manifest.json").read_text())
    if (teacher["arm"] != "LOOP-T" or teacher["checkpoint_sha256"] != digest_file(args.run / "model.pt")
            or teacher["checkpoint_sha256"] != factor["teacher_checkpoint_sha256"]
            or teacher["manifest_sha256"] != pair_manifest["source_manifest_sha256"]
            or teacher["manifest_sha256"] != factor["manifest_sha256"]
            or factor["bpd_revision"] != BPD_REVISION or factor["site"] != SITE
            or factor["factor_sha256"] != digest_file(args.factors / "factors.pt")):
        raise ValueError("teacher, factors, or pairs provenance mismatch")
    model = build_model("LOOP-T", visits=teacher["visits"])
    model.load_state_dict(torch.load(args.run / "model.pt", weights_only=True, map_location="cpu"))
    model.eval()
    weight = model.get_submodule(SITE).weight.detach().clone()
    if weight_digest(weight) != factor["teacher_weight_sha256"]:
        raise ValueError("physical weight mismatch")
    factors = torch.load(args.factors / "factors.pt", weights_only=True, map_location="cpu")
    v, u = factors["V"], factors["U"]
    ids = component_ids(v, u, args.rank)
    v_sub, u_sub = v[:, ids], u[ids]
    svd = torch.linalg.svd(weight.float(), full_matrices=False).U[:, :args.rank]
    native_span = torch.linalg.svd(u_sub.float().T, full_matrices=False)
    if float(native_span.S[-1]) <= float(native_span.S[0]) * 1e-6:
        raise ValueError("selected native output span rank deficient")
    q = native_span.U[:, :args.rank]
    # Disable sphere rescaling only in this additive, exact-dose experiment.
    bmodel = bpd.BPDModel(model, {SITE: int(u.shape[0])}, max_passes=args.visits,
                         ablation=bpd.AblationConfig(sphere_rescale=False))
    wrapper = bmodel.wrappers[SITE]
    wrapper.V.copy_(v)
    wrapper.U.copy_(u)
    family = teacher["family"]
    name = f"{args.split}/{family}.jsonl"
    path = args.pairs / name
    if digest_file(path) != pair_manifest["files"][name]["sha256"]:
        raise ValueError("pair hash mismatch")
    pairs = rows(path)
    rng = torch.Generator().manual_seed(args.seed)
    records = []
    schedule = defaultdict(list)
    for p in pairs:
        if p["kind"] == "changed_target" and p["first_divergence"] <= args.visits:
            schedule[p["anchor"]["difficulty"]].append(p["first_divergence"])
    counters = Counter()

    def clean(row):
        x = torch.tensor([row["tokens"]])
        with bmodel.target_capture():
            o = model(x, depth_visits=args.visits)
        inputs = bmodel.collect_recorded_inputs(expected_passes=args.visits)[SITE]
        return o, inputs

    def additive(row, visit, delta):
        x = torch.tensor([row["tokens"]])
        calls = 0
        def inject(_module, _inputs, output):
            nonlocal calls
            calls += 1
            if calls == visit:
                result = output.clone()
                result[:, -1] += delta
                return result
            return None
        handle = wrapper.register_forward_hook(inject)
        try:
            with bmodel.target_capture():
                return model(x, depth_visits=args.visits)
        finally:
            handle.remove()

    def report(o, visit, local_target, final_target, anchor_target, clean_o, tolerance):
        logits = o.logits[0, -1].float()
        base = clean_o.logits[0, -1].float()
        local = model.lens(o.visit_outputs[visit - 1][:, -1])[0].float()
        return {"final_hit": int(logits.argmax()) == final_target,
                "local_hit": int(local.argmax()) == local_target,
                "final_prediction": int(logits.argmax()), "local_prediction": int(local.argmax()),
                "donor_margin_shift": float((logits[final_target] - logits[anchor_target])
                                           - (base[final_target] - base[anchor_target])),
                "local_teacher_kl": float(torch.nn.functional.kl_div(
                    local.log_softmax(-1),
                    model.lens(clean_o.visit_outputs[visit - 1][:, -1])[0].float().softmax(-1),
                    reduction="sum")),
                "calibration_tolerance": tolerance}

    for p in pairs:
        a, b = p["anchor"], p["donor"]
        d = a["difficulty"]
        record = {"anchor": a["example_id"], "donor": b["example_id"],
                  "kind": p["kind"], "depth": d, "edit": p["edit"]}
        visit = p["first_divergence"]
        if visit is None:
            choices = schedule[d]
            if not choices:
                records.append({**record, "status": "no_changed_visit_control"})
                continue
            visit = choices[counters[d] % len(choices)]
            counters[d] += 1
        if visit > args.visits:
            records.append({**record, "status": "outside_visit_budget", "visit": visit})
            continue
        oa, xa = clean(a)
        ob, xb = clean(b)
        a_x, b_x = xa[:, visit - 1, -1], xb[:, visit - 1, -1]
        a_target, b_target = a["target_token"], b["target_token"]
        local_target = b["symbolic_trace_tokens"][visit]
        correct = int(oa.logits[0, -1].argmax()) == a_target and int(ob.logits[0, -1].argmax()) == b_target
        delta = read_swap_delta(a_x, b_x, v_sub, u_sub)
        dose = float(delta.norm())
        record.update(visit=visit, clean_both_correct=correct, dose=dose,
                      anchor_prediction=int(oa.logits[0, -1].argmax()),
                      donor_prediction=int(ob.logits[0, -1].argmax()))
        if dose <= args.min_dose:
            records.append({**record, "status": "tiny_native_dose"})
            continue
        svd_delta = dose_match(((b_x - a_x) @ weight.T) @ svd @ svd.T, dose)
        random_delta = dose_match(torch.randn(delta.shape, generator=rng) @ q @ q.T, dose)
        if svd_delta is None or random_delta is None:
            records.append({**record, "status": "unmatchable_control"})
            continue
        # Preserve the anchor's component reads at every non-answer position.
        donor_acts = xa[:, visit - 1].float() @ v_sub.float()
        donor_acts[:, -1] = b_x.float() @ v_sub.float()
        with bmodel.intervene(SITE, ids, passes={visit - 1}, mode="swap_acts", donor_acts=donor_acts):
            native = model(torch.tensor([a["tokens"]]), depth_visits=args.visits)
        exact = additive(a, visit, delta)
        parity = float((native.logits - exact.logits).abs().max())
        if parity > 2e-4:
            raise RuntimeError(f"native API vs independently computed write delta mismatch: {parity}")
        ctrl_svd = additive(a, visit, svd_delta)
        ctrl_random = additive(a, visit, random_delta)
        tolerance = factor.get("tolerances", {}).get(str(visit - 1))
        arms = {key: report(o, visit, local_target, b_target, a_target, oa, tolerance)
                for key, o in (("native", native), ("svd", ctrl_svd), ("random", ctrl_random))}
        records.append({**record, "status": "evaluated", "api_parity_max_abs": parity,
                        "svd_dose": float(svd_delta.norm()), "random_dose": float(random_delta.norm()),
                        "arms": arms})
    args.out.mkdir(parents=True)
    write_rows(args.out / "predictions.jsonl", records)
    dump(args.out / "receipt.json", {
        "method": "native_BPD_swap_acts_answer_position", "sphere_rescale": False,
        "teacher_checkpoint_sha256": teacher["checkpoint_sha256"],
        "factor_sha256": factor["factor_sha256"], "pairs_sha256": digest_file(path),
        "predictions_sha256": digest_file(args.out / "predictions.jsonl"),
        "bpd_revision": BPD_REVISION, "component_ids": ids, "rank": args.rank,
        "visits": args.visits, "split": args.split, "seed": args.seed,
        "min_dose": args.min_dose, **summarize(records),
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bpd-checkout", "run", "factors", "pairs", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--split", choices=("validation", "locked_familiar", "locked_novel"), required=True)
    p.add_argument("--unlock", action="store_true")
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--visits", type=int, default=8)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=2026112605)
    p.add_argument("--min-dose", type=float, default=1e-7)
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
