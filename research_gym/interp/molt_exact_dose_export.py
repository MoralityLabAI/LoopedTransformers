"""Export MOLT-16 numerical witnesses for independent rational verification.

Reproduce the parent roster exactly. The exact checker does not import torch
or the teacher and does not trust the parent's numerical classifications.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
from pathlib import Path

import torch

from research_gym.interp.aux_gym_v2 import digest_file, dump
from research_gym.interp.molt_amortized_repair import prepare
from research_gym.interp.molt_bpd_source import load_verified
from research_gym.interp.molt_carry_feasibility import evaluate, fresh_roots, load_controllers, summarize
from research_gym.interp.molt_clock_carry import CUT, fit_source
from research_gym.interp.molt_typed_clock import MOD, SITE, bank_digest
from lsa.exact_carry_certificate import SCHEMA


def hex_vector(x):
    return [float(a).hex() for a in x.detach().cpu()]


def hex_matrix(x):
    return [hex_vector(row) for row in x]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("clock", "factors", "bpd-checkout", "bpd-source-receipt", "controllers", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--roots", type=int, default=16)
    args = p.parse_args()
    if args.out.exists() or not 1 <= args.roots <= 16:
        p.error("fresh output and bounded parent-replay roster required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model, v, u, native, ck, fh = fit_source(args)
    load_verified(args.bpd_checkout, args.bpd_source_receipt)
    bases, nets, mean, scale = load_controllers(args, model, native, ck, fh)
    worlds = [w for depth in (4, 8) for w in fresh_roots(depth, args.roots)]
    rows = evaluate(model, prepare(model, worlds, v, u), bases, nets, mean, scale,
                    export_witnesses=True)
    norm = model.decoder.final_norm
    weight = model.decoder.embedding.weight[:MOD]
    binding = {
        "checkpoint_sha256": ck, "factor_sha256": fh,
        "controller_sha256": digest_file(args.controllers / "controllers.pt"),
        "bpd_source_receipt_sha256": digest_file(args.bpd_source_receipt),
        "parent_source_sha256": digest_file(Path(__file__).with_name("molt_carry_feasibility.py")),
        "export_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "site": SITE, "cut": CUT, "root_roster_sha256": bank_digest(worlds),
    }
    records = [{"root": r["root"], "kind": r["kind"], "chart": r["chart"],
                "parent_status": r["status"], **r["witness_inputs"]} for r in rows]
    bundle = {
        "schema": SCHEMA, "width": weight.shape[1], "legal_classes": MOD,
        "rank": native.shape[1], "world_roots": len(worlds), "binding": binding,
        "weight": hex_matrix(weight), "gamma": hex_vector(norm.weight),
        "bias": hex_vector(norm.bias), "epsilon": float(norm.eps).hex(),
        "charts": {name: hex_matrix(q) for name, q in bases.items()},
        "records": records,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out / "problems.json", bundle)
    dump(args.out / "parent-rows.json", rows)
    report = {
        "schema": "molt17-export-v1", "status": "parent_roster_replayed",
        "binding": binding, "world_roots": len(worlds), "records": len(records),
        "problems_sha256": digest_file(args.out / "problems.json"),
        "parent_rows_sha256": digest_file(args.out / "parent-rows.json"),
        "parent_summary": summarize(rows),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "new_training_performed": False, "confirmation_ready": False,
        "promotion_authorized": False,
    }
    dump(args.out / "receipt.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
