"""Run the registered OLR-08A campaign through the package's own launch.main, with one
pre-registered stop: fresh evaluation outcomes are collected only if at least one learned
reader was fitted and frozen on source data. When no model is eligible (failed headroom
gate or fewer than 64 source frames) the run stops before the first fresh-evaluation stage,
writes summary.json with the frozen source/calibration results, and packages the audit and
FABLE archives exactly as launch.py does. Resuming launch.py on the same --out later
continues from the checkpoints under the same contract.

Usage: python run_full.py <package dir> <launch.py arguments>
"""
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

PKG = Path(sys.argv[1]).resolve()
sys.argv = [sys.argv[0], *sys.argv[2:]]
sys.path.insert(0, str(PKG))

import launch  # noqa: E402
import run08  # noqa: E402

FRESH_STAGES = ("menu_fresh",)


class NoEligibleModel(Exception):
    pass


def out_dir():
    i = sys.argv.index("--out")
    return Path(sys.argv[i + 1]).resolve()


def arg(name):
    return Path(sys.argv[sys.argv.index(name) + 1]).resolve() if name in sys.argv else None


_stage = run08.stage


def guarded(out, name, rows, fn, workers=1):
    if name in FRESH_STAGES:
        frozen = run08.read(out / "frozen_policy.json")
        if not frozen.get("models"):
            raise NoEligibleModel(name)
    return _stage(out, name, rows, fn, workers)


def package(out):
    """Same file selection and archives as launch.main, plus this driver."""
    home = launch.HOME
    code = out / "code"
    code.mkdir(exist_ok=True)
    for p in home.rglob("*"):
        if p.is_file() and (p.suffix in (".py", ".md", ".lean", ".toml") or p.name in (
                "lean-toolchain", "spent.json", "train_registry.json", "package_manifest.json")) \
                and ".lake" not in p.parts and "__pycache__" not in p.parts:
            dst = code / p.relative_to(home)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
    drv = out / "driver"
    drv.mkdir(exist_ok=True)
    for p in Path(__file__).resolve().parent.glob("*.py"):
        shutil.copy2(p, drv / p.name)
    for k, p in (("07a", arg("--reissue-a")), ("07b", arg("--reissue-b"))):
        if p:
            shutil.copy2(p, out / f"reissue_{k}_manifest.json")
    files = {p.relative_to(out).as_posix(): run08.sha(p) for p in out.rglob("*")
             if p.is_file() and p.name != "audit_manifest.json"}
    run08.write(out / "audit_manifest.json", {"files": files})
    full = out.with_name(out.name + "_AUDIT.zip")
    small = out.with_name(out.name + "_FABLE.zip")
    keep = ("summary.json", "contract.json", "registry_census.json", "resource.json", "frozen_policy.json", "prior.json")
    for zipname, names in ((full, [*files, "audit_manifest.json"]),
                           (small, [n for n in files if n.startswith(("code/", "driver/", "reissue_")) or n in keep]
                            + ["audit_manifest.json"])):
        with zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED) as z:
            for n in names:
                z.write(out / n, n)
        print(zipname, run08.sha(zipname), flush=True)


def main():
    out = out_dir()
    run08.stage = guarded
    try:
        launch.main()
    except NoEligibleModel as e:
        frozen = run08.read(out / "frozen_policy.json")
        run08.write(out / "summary.json", {
            "schema": "olr08a.cpu.result.v1", "profile": "full",
            "status": "stopped_before_fresh_evaluation_no_eligible_model",
            "stopped_before_stage": str(e), "frozen": frozen,
            "headroom_gate": frozen.get("headroom_gate"), "source_frames": frozen.get("source_frames"),
            "fresh_outcomes_collected": False, "training_updates": 0,
            "architecture_promotion_eligible": False, "sealed_SInt_opened": False,
            "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        r = run08.read(out / "resource.json")
        r["stop"] = "pre-registered: no eligible model before fresh evaluation"
        run08.write(out / "resource.json", r)
        print("STOPPED_BEFORE_FRESH_EVALUATION", e, flush=True)
        package(out)


if __name__ == "__main__":
    main()
