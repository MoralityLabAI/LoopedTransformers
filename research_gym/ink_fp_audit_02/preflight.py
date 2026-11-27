"""Stage gates: resource preflight and Tinker qualification. Writes a receipt; never prints secrets."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

from . import resources as res
from . import tinker_backend as tb


def resource_preflight(limits: res.Limits = res.Limits()) -> dict:
    start = res.startup_check(limits)
    try:
        job = res.JobObject(limits.job_memory_cap)
        job.close()
        job_ok, job_err = True, None
    except OSError as exc:
        job_ok, job_err = False, type(exc).__name__
    return {"platform": platform.platform(), "python": sys.version.split()[0],
            "available_bytes": start["available_bytes"],
            "startup_min_available_bytes": limits.startup_min_available,
            "startup_memory_ok": start["ok"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_allocation_bytes": res.gpu_allocation_bytes(),
            "process_tree_rss_bytes": res.process_tree_rss(),
            "job_object_supported": job_ok, "job_object_error": job_err,
            "max_threads": limits.max_threads, "max_concurrency": limits.max_concurrency}


def api_preflight() -> dict:
    caps = tb.sdk_capabilities()
    return {"credential_present": tb.credential_present(), "model": tb.MODEL, "sdk": caps,
            "default_context": tb.DEFAULT_CONTEXT, "context_ceiling": tb.CONTEXT_CEILING,
            "default_max_tokens": tb.DEFAULT_MAX_TOKENS}


def gates(r: dict, a: dict) -> dict:
    blocked = []
    if not r["startup_memory_ok"]:
        blocked.append(f"available DRAM {r['available_bytes'] / res.GIB:.2f} GiB < "
                       f"{r['startup_min_available_bytes'] / res.GIB:.0f} GiB startup minimum")
    if not a["credential_present"]:
        blocked.append("TINKER_API_KEY not present in process environment")
    if not a["sdk"].get("installed"):
        blocked.append("tinker SDK not installed")
    if r["gpu_allocation_bytes"]:
        blocked.append("local GPU allocation nonzero")
    return {"remote_stages_allowed": not blocked, "blocked_reasons": blocked,
            "local_stages_allowed": r["gpu_allocation_bytes"] == 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res.set_below_normal_priority()
    r, api = resource_preflight(), api_preflight()
    receipt = {"schema": "ink-fp-audit-02.preflight.v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "resources": r, "api": api, "gates": gates(r, api)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(receipt["gates"], indent=2))


if __name__ == "__main__":
    main()
