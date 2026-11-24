"""Audit 05 - Custody hashes, suite determinism, and regeneration of the Lean proof certificates.

Read-only with respect to the repository.  Sealed cohort files are hashed as byte streams and
never decoded; the generator is only ever asked for the OPEN splits (train, calibration), which
consume the generator's random stream before any sealed split would be drawn.

Checks
  1. every SHA-256 recorded in results/lbf17 (suite, checkpoints, draws, traces, teacher,
     top-level artifacts) against the archived (gzip-compressed) files;
  2. that the frozen open splits are the deterministic image of the public seed under the
     pinned generator (so are the sealed splits, by construction: the "seal" is procedural);
  3. matched training draw schedules and the shared rel_prov_standard / rel_prov_opened init;
  4. regeneration of calibration_certificates/Proofs.lean (2,433 policy witnesses + 48
     references) from the archived episode paths, comparison with the receipt's source hash,
     and - with --lean - independent kernel checking of all 2,481 declarations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def canon_rows(rg, rows) -> bytes:
    return "".join(rg.canonical(x) + "\n" for x in rows).encode("utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pinned-sim", required=True)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--lean", default=None, help="optional Lean 4.19.0 executable for the certificate recheck")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    repo = Path(args.repo)
    scratch = common.scratch_dir(args.scratch)
    rg, m15, m16, m17 = common.load_modules(repo, Path(args.pinned_sim))
    res = common.lbf_dir(repo) / "results" / "lbf17"
    manifest = json.loads((res / "manifest.json").read_text())
    suite_m = json.loads((res / "suite" / "manifest.json").read_text())
    out: dict = {"review_commit": common.REVIEW_COMMIT}

    # ---- 1. hashes ---------------------------------------------------------------------------
    suite = {}
    for name, want in suite_m["artifacts"].items():
        gz = res / "suite" / (name + ".gz")
        if not gz.exists():
            suite[name] = {"archived": False, "registered_sha256": want}
            continue
        got = common.gz_stream_sha256(gz)  # byte stream only; sealed splits are never decoded
        suite[name] = {"archived": True, "matches": got == want,
                       "sealed_hash_only": any(name.startswith(s + ".") for s in common.SEALED_SPLITS)}
    out["suite_artifacts"] = suite
    def text_file_match(path: Path, want: str) -> dict:
        raw = path.read_bytes()
        crlf = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        return {"archived": True, "matches_as_checked_out": common.sha256_bytes(raw) == want,
                "matches_after_lf_to_crlf": common.sha256_bytes(crlf) == want}

    out["suite_manifest_sha256"] = text_file_match(res / "suite" / "manifest.json", manifest["suite_manifest_sha256"])

    fits = []
    for rec in manifest["fits"]:
        d = res / f'{rec["seed"]}_{rec["arch"]}'
        local = json.loads((d / "manifest.json").read_text())
        fits.append({
            "seed": rec["seed"], "arch": rec["arch"], "parameters": rec["parameters"],
            "checkpoint": common.gz_stream_sha256(d / "checkpoint.pt.gz") == rec["checkpoint_sha256"],
            "draws": common.gz_stream_sha256(d / "draws.jsonl.gz") == rec["draws_sha256"],
            "trace": common.gz_stream_sha256(d / "trace.jsonl.gz") == rec["trace_sha256"],
            "local_manifest_equals_top_manifest": local == rec,
            "fit_wall_clock_sec": rec["fit_wall_clock_sec"],
        })
    out["fits"] = fits
    out["all_24_checkpoint_draw_trace_hashes_match"] = all(f["checkpoint"] and f["draws"] and f["trace"] for f in fits)
    out["total_fit_cpu_hours"] = sum(f["fit_wall_clock_sec"] for f in fits) / 3600.0

    arts = {}
    for name, want in manifest["artifacts"].items():
        plain, gz = res / name, res / (name + ".gz")
        if plain.exists():
            arts[name] = text_file_match(plain, want)
        elif gz.exists():
            arts[name] = {"archived": True, "matches_as_checked_out": common.gz_stream_sha256(gz) == want}
        else:
            arts[name] = {"archived": False}
    out["top_level_artifacts"] = arts
    out["line_ending_note"] = ("registered hashes were computed over Windows CRLF bytes; .gitattributes normalises text "
                               "files to LF, so plain-text artifacts verify only after LF->CRLF conversion, while "
                               "gzip-archived artifacts verify byte-for-byte")
    out["teacher_frames_summary_matches"] = (
        common.gz_stream_sha256(res / "teacher" / "frames_summary.jsonl.gz") == manifest["training"]["frames_summary_sha256"])
    out["abstraction_diagnostics_sha256"] = common.sha256_file(res / "abstraction_diagnostics.json")
    cert_dir = res / "calibration_certificates"
    out["archived_certificate_files"] = sorted(p.name for p in cert_dir.iterdir())
    out["lbf16_suite_archived_in_repo"] = (common.lbf_dir(repo) / "results" / "lbf16" / "suite").exists()

    # ---- 2. suite determinism (open splits only) ----------------------------------------------
    counts_open = {"train": suite_m["counts"]["train"], "calibration": suite_m["counts"]["calibration"]}
    cases, witnesses = rg.generate(suite_m["seed"], counts_open)
    det = {}
    for split in common.OPEN_SPLITS:
        raw = canon_rows(rg, [x.public() for x in cases[split]])
        det[f"{split}.jsonl"] = hashlib.sha256(raw).hexdigest() == suite_m["artifacts"][f"{split}.jsonl"]
        rawc = raw.replace(b"\n", b"\r\n")
        det[f"{split}.jsonl_crlf_variant"] = hashlib.sha256(rawc).hexdigest() == suite_m["artifacts"][f"{split}.jsonl"]
        w = canon_rows(rg, witnesses[split])
        det[f"{split}_witnesses.jsonl"] = hashlib.sha256(w).hexdigest() == suite_m["artifacts"][f"{split}_witnesses.jsonl"]
        det[f"{split}_witnesses.jsonl_crlf_variant"] = (
            hashlib.sha256(w.replace(b"\n", b"\r\n")).hexdigest() == suite_m["artifacts"][f"{split}_witnesses.jsonl"])
    out["open_splits_regenerate_from_public_seed"] = det
    out["seal_is_procedural_not_cryptographic"] = {
        "generator_is_deterministic_in_public_seed": True,
        "sealed_files_stored_as_plain_gzip_in_repo": all(
            (res / "suite" / f"{s}.jsonl.gz").exists() for s in common.SEALED_SPLITS),
        "campaign_code_parses_sealed_files_for_lineage": "verify_historical_disjointness reads rg.read_rows(suite/'ordinary.jsonl') "
                                                          "and suite/'ood.jsonl' and regenerates every historical suite, sealed splits included",
        "manifest_flags_are_literals": "sealed_outcomes_opened and lbf16_sealed_cohorts_preserved_unopened are hard-coded constants",
    }

    # ---- 3. matched draws and shared initialisation --------------------------------------------
    by = {(f["seed"], f["arch"]): f for f in manifest["fits"]}
    out["draw_schedule_identical_across_archs_per_seed"] = {
        str(s): len({by[(s, a)]["draws_sha256"] for a in m17.ALL_LEARNED_ARCHS}) == 1 for s in m17.SEEDS}
    out["standard_opened_shared_initial_equal"] = {
        str(s): by[(s, "rel_prov_standard")]["shared_initial"] == by[(s, "rel_prov_opened")]["shared_initial"]
        for s in m17.SEEDS}
    out["parameter_counts"] = {a: by[(107, a)]["parameters"] for a in m17.ALL_LEARNED_ARCHS}
    out["registered_parameter_counts"] = {
        "rel_prov_opened": 53074, "rel_prov_standard": 53074, "adj2_noprov_opened": 53074, "lowrank_opened": 53041,
        "lbf16_pos1_opened": 53047, "param_nonrecurrent": 53089, "compute_nonrecurrent": 53066, "sym_prov_readout": 53078}
    out["registered_parameter_window"] = [53041, 53089]
    out["archs_outside_registered_window"] = sorted(
        a for a, n in out["parameter_counts"].items() if not (53041 <= n <= 53089))

    # ---- 4. regenerate the calibration proof certificates ---------------------------------------
    cal = cases["calibration"]
    lookup = {c.root: c for c in cal}
    cal_refs = {w["root"]: w["path"] for w in witnesses["calibration"]}
    rows = common.read_jsonl_gz(res / "episodes.jsonl.gz")
    proofs = []
    for i, row in enumerate(rows):
        if row["closed"]:
            rg.replay(lookup[row["root"]], row["path"])
            proofs.append((lookup[row["root"]], row["path"], f"policy_{i}"))
    n_policy = len(proofs)
    proofs.extend((c, cal_refs[c.root], "reference_" + c.root) for c in cal)
    names = [n for _, _, n in proofs]
    text = ("namespace LBF12Cert\n\n" + "\n\n".join(m17.base.render_proof(rg, c, p, n) for c, p, n in proofs)
            + "\n\nend LBF12Cert\n" + "\n".join(f"#print axioms LBF12Cert.{n}" for n in names) + "\n")
    receipt = json.loads((cert_dir / "manifest.json").read_text())
    sha_lf = hashlib.sha256(text.encode("utf-8")).hexdigest()
    sha_crlf = hashlib.sha256(text.replace("\n", "\r\n").encode("utf-8")).hexdigest()
    lens = [len(p) for _, p, _ in proofs[:n_policy]]
    certs = {
        "policy_proofs": n_policy, "reference_proofs": len(cal), "declarations": len(names),
        "declaration_list_equals_receipt": names == receipt["declarations"],
        "regenerated_source_sha256_lf": sha_lf, "regenerated_source_sha256_crlf": sha_crlf,
        "receipt_source_sha256": receipt["source_sha256"],
        "regenerated_source_matches_receipt": receipt["source_sha256"] in (sha_lf, sha_crlf),
        "policy_proof_length_min_median_max": [min(lens), sorted(lens)[len(lens) // 2], max(lens)],
        "families": {f: sum(1 for c, _, _ in proofs if c.family == f) for f in ("equality", "dependent_transport")},
        "what_a_certificate_states": "for all A, f_a..f_f : A -> A, x : A, and rule hypotheses h_i : forall z, lhs_i(z) = rhs_i(z): "
                                     "source(x) = target(x)  (or P source -> P target for the transport family). "
                                     "It certifies the witness path only: not the successor accounting, the budget, "
                                     "the abstraction, nor any B < 128 closure.",
    }
    if args.lean:
        path = scratch / "Proofs.lean"
        path.write_text(text, encoding="utf-8")
        version = subprocess.run([args.lean, "--version"], capture_output=True, text=True, check=True).stdout.strip()
        r = subprocess.run([args.lean, str(path)], capture_output=True, text=True, timeout=3600)
        log = r.stdout + r.stderr
        passed = set(re.findall(r"'LBF12Cert\.([a-z0-9_]+)' does not depend on any axioms", log))
        certs["independent_lean_recheck"] = {
            "lean_version": version, "returncode": r.returncode,
            "axiom_free_declarations": len(passed), "all_declarations_axiom_free": passed == set(names) and r.returncode == 0,
            "log_sha256_lf": hashlib.sha256(log.encode()).hexdigest(),
            "log_sha256_crlf": hashlib.sha256(log.replace("\n", "\r\n").encode()).hexdigest(),
            "receipt_log_sha256": receipt["log_sha256"],
        }
        certs["independent_lean_recheck"]["log_matches_receipt"] = receipt["log_sha256"] in (
            certs["independent_lean_recheck"]["log_sha256_lf"], certs["independent_lean_recheck"]["log_sha256_crlf"])
    out["calibration_certificates"] = certs
    common.write_json(Path(args.out), out)
    brief = {k: v for k, v in out.items() if k not in ("fits",)}
    print(json.dumps(brief, indent=1)[:6000])


if __name__ == "__main__":
    main()
