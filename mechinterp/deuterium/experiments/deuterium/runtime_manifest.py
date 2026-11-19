"""Record hardware and software without collecting credentials or host secrets."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def capture() -> dict:
    cpu = {}
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() in {"model name", "vendor_id", "cpu cores", "flags"}:
                cpu.setdefault(key.strip(), value.strip())
    memory = {}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"MemTotal", "SwapTotal"}:
                memory[key] = value.strip()
    versions = {}
    for package in ("torch", "numpy", "pytest", "matplotlib", "reportlab"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    result = {
        "schema_version": 1,
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpus": os.cpu_count(),
        "cpu": cpu,
        "memory": memory,
        "packages": versions,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status_porcelain": subprocess.check_output(["git", "status", "--porcelain"], text=True).splitlines(),
        "analysis_sources": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path("experiments/deuterium").glob("*.py"))
        },
        "github": {key: os.environ[key] for key in (
            "GITHUB_REPOSITORY", "GITHUB_SHA", "GITHUB_REF", "GITHUB_RUN_ID",
            "GITHUB_RUN_ATTEMPT", "GITHUB_WORKFLOW", "RUNNER_OS", "RUNNER_ARCH",
            "ImageOS", "ImageVersion",
        ) if key in os.environ},
    }
    if "torch" in versions:
        import torch
        result["torch"] = {"cuda_available": torch.cuda.is_available(),
                           "threads": torch.get_num_threads(),
                           "config": torch.__config__.show()}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(capture(), indent=2) + "\n")
