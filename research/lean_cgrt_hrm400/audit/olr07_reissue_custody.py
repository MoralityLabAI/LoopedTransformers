"""Authenticate the corrected OLR-07A/B reissue artifacts.

Checks, for each kind, that the artifact bytes match the digest GitHub recorded for
workflow run RUN_ID, that every file listed in the reissue manifest is present with its
hash and nothing else is, that the reissued registry equals the original registry in the
authenticated original archive, that the recorded tracer and runner hashes equal the
files at the run's head commit, and that both native scope sentinels passed.
Standard library only. Reads the inputs and writes one custody JSON.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

RUN_ID = 45388529946
HEAD = "ded064c8a993fea70c2d7033b064cd09ae788d19"
WORKFLOW = ".github/workflows/olr07-scope-reissue.yml"
SOURCE = "6caaee842e9495688c1567e78c0e68dbb96942aa"
VERSION = "pre_command_v1"
SCHEMA = "olr07.scope_reissue.v1"
TRACER = "research/lean_cgrt_hrm400/lean/Trace.lean"
RUNNER = "research/lean_cgrt_hrm400/ppt_gym/olr07_reissue.py"
# Digests shown on the run page for run RUN_ID.
DIGESTS = {
    "07a": "b3b2fcfb28ddadc9af262acf082194ad27d3c63a36a9e33df4511dcd1cd47720",
    "07b": "e1df19a94877646dc71d91cdafdfdb04dc81aea1cc5ce0243284a41d511b09e8",
}
PINS = {
    "07a": {"manifest": "manifest.json",
            "manifest_sha": "846958ede542bedd510d865b3559e6f4711a97963583084c2d1028e70da2328e",
            "registry": "bank/states.json",
            "registry_sha": "029b8d591851c7034736db1c1ec09c6e956c80e339fefb8c42e7a0429ce18151",
            "count": 207, "human_hb": 500},
    "07b": {"manifest": "audit_manifest.json",
            "manifest_sha": "b388335c52ffb4b40b63e3cc77a37202c8c70553901a115635dc74f1d735b1ad",
            "registry": "registry.json",
            "registry_sha": "d5cde5012d5312305b0b6e19b3ea823db8e55bdb95ef83dddfc71dfb2e8c212c",
            "count": 1441, "human_hb": 20000},
}
SENTINEL_KEYS = ("earlier_certified", "current_rejected", "later_rejected", "simp_current_rejected")


class CustodyError(ValueError):
    pass


def need(cond, msg):
    if not cond:
        raise CustodyError(msg)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def original(path: Path, kind: str):
    """Authenticate an original archive and return its parsed registry."""
    pin = PINS[kind]
    raw_zip = path.read_bytes()
    with zipfile.ZipFile(path) as z:
        raw = z.read(pin["manifest"])
        need(sha(raw) == pin["manifest_sha"], f"{kind}: original manifest bytes differ")
        files = json.loads(raw)["files"]
        need(files.get(pin["registry"]) == pin["registry_sha"], f"{kind}: original registry pin differs")
        for n, h in files.items():
            need(sha(z.read(n)) == h, f"{kind}: original file hash mismatch {n}")
        roots = json.loads(z.read(pin["registry"]))
    need(len(roots) == pin["count"], f"{kind}: original root census changed")
    return roots, {"archive_sha256": sha(raw_zip), "verified_files": len(files)}


def git_blob(git: str, repo: Path, rev: str, rel: str) -> bytes:
    return subprocess.check_output([git, "-C", str(repo), "show", f"{rev}:{rel}"])


def entries(z: zipfile.ZipFile, prefix: str):
    return {n[len(prefix):]: n for n in z.namelist() if n.startswith(prefix) and not n.endswith("/")}


def sentinel(z: zipfile.ZipFile, name: str, kind: str):
    need(name in z.namelist(), f"{kind}: sentinel PASS missing at {name}")
    p = json.loads(z.read(name))
    need(p.get("scope_version") == VERSION, f"{kind}: sentinel scope marker")
    need(all(p.get(k) is True for k in SENTINEL_KEYS), f"{kind}: sentinel verdict {p}")
    return {k: p[k] for k in SENTINEL_KEYS}


def reissue(zpath: Path, kind: str, roots, orig_custody, artifact, tracer_sha, runner_sha):
    raw_zip = zpath.read_bytes()
    need(sha(raw_zip) == DIGESTS[kind], f"{kind}: artifact bytes differ from the run's recorded digest")
    if artifact is not None:
        need(artifact.get("digest") == "sha256:" + DIGESTS[kind], f"{kind}: API digest differs from pinned digest")
        need(artifact.get("size_in_bytes") == len(raw_zip), f"{kind}: artifact size differs")
    pin = PINS[kind]
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        tops = sorted({n.split("/", 1)[0] for n in names})
        need(tops == sorted([f"scope-reissue-{kind}", f"scope-sentinel-{kind}"]), f"{kind}: artifact layout {tops}")
        pre = f"scope-reissue-{kind}/"
        raw_m = z.read(pre + "manifest.json")
        m = json.loads(raw_m)
        need(m.get("schema") == SCHEMA and m.get("scope_version") == VERSION and m.get("kind") == kind,
             f"{kind}: manifest schema/scope/kind")
        need(m.get("old_receipts_superseded") is True, f"{kind}: superseded flag")
        need(m.get("model_training_updates") == 0, f"{kind}: training updates recorded")
        need(m.get("architecture_promotion_eligible") is False, f"{kind}: promotion flag")
        listed = m["files"]
        present = entries(z, pre)
        present.pop("manifest.json")
        need(set(present) == set(listed),
             f"{kind}: unlisted or missing files {sorted(set(present) ^ set(listed))[:5]}")
        for rel, h in listed.items():
            need(sha(z.read(present[rel])) == h, f"{kind}: manifest-listed hash mismatch {rel}")
        c = m["custody"]
        need(c.get("manifest_sha256") == pin["manifest_sha"] and c.get("registry_sha256") == pin["registry_sha"],
             f"{kind}: custody pins")
        need(c.get("archive_sha256") == orig_custody["archive_sha256"],
             f"{kind}: reissue read a different original archive")
        need(c.get("verified_files") == orig_custody["verified_files"], f"{kind}: original file count")
        contract = json.loads(z.read(pre + "contract.json"))
        need(contract.get("kind") == kind and contract.get("scope_version") == VERSION, f"{kind}: contract kind/scope")
        need(contract.get("source_commit") == SOURCE, f"{kind}: contract source commit")
        need(contract.get("custody") == c, f"{kind}: contract custody differs from manifest")
        need(contract.get("candidate_hb") == 500 and contract.get("human_hb") == pin["human_hb"],
             f"{kind}: candidate or human budget changed")
        need(contract.get("tracer_sha256") == tracer_sha, f"{kind}: tracer differs from run head")
        need(contract.get("runner_sha256") == runner_sha, f"{kind}: runner differs from run head")
        frozen = json.loads(z.read(pre + "frozen_registry.json"))
        need(frozen == roots, f"{kind}: reissued registry differs from the original registry")
        rows = json.loads(z.read(pre + "receipts.json"))
        need([r["id"] for r in rows] == [r["id"] for r in roots], f"{kind}: receipt roster or order differs")
        for role, cen in m["census"].items():
            sub = [r for r in rows if r["role"] == role]
            ok = [r for r in sub if r["scored"]]
            need(cen["selected"] == len(sub) and cen["scored"] == len(ok)
                 and cen["human_closed"] == sum(bool(r["human_closed"]) for r in ok)
                 and cen["exclusions"] == dict(Counter(r.get("reason", "unknown") for r in sub if not r["scored"])),
                 f"{kind}: census for {role} does not recount")
            need(all(r.get("scope_version") == VERSION and r.get("env_scope") == "before"
                     and r.get("decl_in_env_before") is False for r in ok), f"{kind}: scored row scope markers")
        sentinels = {"standalone": sentinel(z, f"scope-sentinel-{kind}/PASS.json", kind),
                     "in_run": sentinel(z, pre + "sentinel/PASS.json", kind)}
    gates = {k: m.get(k) for k in ("native_valid", "source_calibration_gate", "fresh_gap_gate",
                                    "selected_on_calibration", "fixed_order", "elapsed_seconds")}
    return {
        "artifact": {"name": zpath.name, "id": (artifact or {}).get("id"), "bytes": len(raw_zip),
                     "sha256": DIGESTS[kind], "expires_at": (artifact or {}).get("expires_at")},
        "manifest_sha256": sha(raw_m), "listed_files_verified": len(listed),
        "original": {**c, "registry_roots": len(roots)},
        "contract": {k: contract[k] for k in ("lean_sha256", "repl_sha256", "tracer_sha256", "runner_sha256",
                                               "candidate_hb", "human_hb", "source_commit")},
        "sentinels": sentinels, "gates": gates, "census": m["census"],
        "clean_ceiling": m.get("clean_ceiling"), "paired": m.get("paired"),
        "calibration": m.get("calibration"), "evaluation": m.get("evaluation"),
    }


def run_meta(run, jobs):
    need(run.get("id") == RUN_ID and run.get("status") == "completed" and run.get("conclusion") == "success",
         "run is not the completed successful reissue run")
    need(run.get("head_sha") == HEAD and run.get("path") == WORKFLOW, "run head or workflow path differs")
    js = jobs.get("jobs", [])
    need(len(js) == 2 and all(j.get("conclusion") == "success" for j in js), "reissue jobs did not both succeed")
    return {"run_id": RUN_ID, "run_attempt": run.get("run_attempt"), "head_sha": HEAD, "event": run.get("event"),
            "workflow": WORKFLOW, "html_url": run.get("html_url"),
            "jobs": [{"id": j["id"], "name": j["name"], "started_at": j["started_at"],
                      "completed_at": j["completed_at"]} for j in js]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, required=True, help="holds olr07-scope-reissue-{07a,07b}.zip")
    p.add_argument("--original-07a", type=Path, required=True)
    p.add_argument("--original-07b", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True, help="git checkout containing commit HEAD")
    p.add_argument("--git", default="git")
    p.add_argument("--run", type=Path, help="API JSON for the run")
    p.add_argument("--jobs", type=Path, help="API JSON for the run's jobs")
    p.add_argument("--artifacts", type=Path, help="API JSON for the run's artifacts")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    tracer_sha = sha(git_blob(a.git, a.repo, HEAD, TRACER))
    runner_sha = sha(git_blob(a.git, a.repo, HEAD, RUNNER))
    arts = {}
    if a.artifacts:
        arts = {x["name"]: x for x in json.loads(a.artifacts.read_text(encoding="utf-8"))["artifacts"]}
    report = {"schema": "olr07.reissue_custody.v1", "pinned_digests": DIGESTS,
              "tracer_sha256_at_head": tracer_sha, "runner_sha256_at_head": runner_sha}
    if a.run and a.jobs:
        report["run"] = run_meta(json.loads(a.run.read_text(encoding="utf-8")),
                                 json.loads(a.jobs.read_text(encoding="utf-8")))
    for kind, orig in (("07a", a.original_07a), ("07b", a.original_07b)):
        roots, oc = original(orig, kind)
        name = f"olr07-scope-reissue-{kind}"
        report[kind] = reissue(a.dir / f"{name}.zip", kind, roots, oc, arts.get(name) if arts else None,
                               tracer_sha, runner_sha)
    report["authenticated"] = True
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("CUSTODY_OK", json.dumps({k: report[k]["gates"] for k in ("07a", "07b")}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (CustodyError, KeyError, OSError, ValueError, subprocess.CalledProcessError) as e:
        print(f"::error title=olr07 custody::{type(e).__name__}: {e}"[:3000], flush=True)
        raise SystemExit(1)
