"""PPT-G03 command line: stage fresh theorems, run real Lean, audit replay.

Run from research/lean_cgrt_hrm400 with pinned Lean 4.19.0:
  python -m ppt_gym.live_cli freeze --out runs/ppt-g03-suite
  cd lean && lake build && cd ..
  python -m ppt_gym.live_cli run --suite runs/ppt-g03-suite \
      --checkpoint runs/PPT_A/checkpoint.pt --out runs/ppt-g03-live
  python -m ppt_gym.live_cli replay --suite runs/ppt-g03-suite \
      --session runs/ppt-g03-live --out runs/ppt-g03-replay

The run command requires a real local Lean binary, lgym server, and independent
standalone certificate compiler. Mock tests do not call this entry point.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from lcgrt import config
from lcgrt.lean_repl import LeanRepl

from .cli import checkpoint_load
from .data import canonical, sha256_file
from .live_campaign import audit_campaign, run_campaign
from .live_lean import fingerprint
from .live_suite import SPLITS, create_suite, load_suite


def _repl(args):
    if not config.REPL_EXE.is_file() or not config.LEAN_EXE.is_file():
        raise RuntimeError("pinned Lean/lgym executable absent; build lean/ with lake build")
    version = subprocess.run([str(config.LEAN_EXE), "--version"], check=True,
                             capture_output=True, text=True, timeout=20,
                             env=config.lean_env()).stdout
    if "4.19.0" not in version:
        raise RuntimeError("live PPT-G03 requires Lean 4.19.0")
    return LeanRepl(timeout=args.request_timeout, exe=config.REPL_EXE,
                    cwd=config.LEAN_DIR, env=config.lean_env())


def execute(args):
    model, checkpoint = checkpoint_load(args.checkpoint)
    train, train_src = load_suite(args.suite, "train", args.support_roots)
    queries, query_src = load_suite(args.suite, args.query_split, args.query_roots)
    if args.query_split == "train":
        raise ValueError("query split must be distinct from live training support")
    src = {"suite_manifest_sha256": train_src["suite_manifest_sha256"],
           "generator_sha256": train_src["generator_sha256"],
           "train_selected": train_src["selected_roots"],
           "query_selected": query_src["selected_roots"],
           "query_split": args.query_split,
           "checkpoint_evidence": checkpoint.get("source_evidence_class", "unknown"),
           "external_overlap": "unchecked",
           "c4_historical_benchmark": "excluded"}
    with _repl(args) as lean:
        src["lean"] = fingerprint(config.REPL_EXE, config.LEAN_EXE, lean.version)
        result = run_campaign(
            lean, model, train, queries, out=args.out, checks=args.checks,
            depth=args.depth, fast_lr=args.fast_lr, score_scale=args.fast_scale,
            max_seconds=args.max_seconds, checkpoint=args.checkpoint,
            source=src, certified_backend=True, cert_timeout=args.cert_timeout,
        )
    return result


def replay(args):
    manifest = json.loads((args.session / "manifest.json").read_text(encoding="utf-8"))
    declared = manifest.get("source", {}).get("lean", {})
    if declared.get("actual_lean_process") is not True:
        raise ValueError("scientific replay requires a real original Lean backend")
    if declared.get("repl_bin_sha256") != sha256_file(config.REPL_EXE):
        raise ValueError("live Lean server bytes changed since original session")
    if declared.get("lean_bin_sha256") != sha256_file(config.LEAN_EXE):
        raise ValueError("Lean kernel binary changed since original session")
    if declared.get("source") != "lcgrt.LeanRepl / lgym":
        raise ValueError("unexpected replay backend contract")
    if manifest["source"]["suite_manifest_sha256"] != sha256_file(args.suite / "manifest.json"):
        raise ValueError("frozen theorem suite changed")
    with _repl(args) as lean:
        if lean.version != declared["lean_version"]:
            raise ValueError("Lean version changed between collection and replay")
        result = audit_campaign(lean, args.suite, args.session)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "replay.json").write_text(canonical(result) + "\n", encoding="utf-8")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="Freeze generated theorem lineage before Lean execution")
    freeze.add_argument("--out", type=Path, required=True)
    freeze.add_argument("--seed", type=int, default=20261122)
    freeze.add_argument("--per-split", type=int, default=24)
    freeze.add_argument("--equal-alt-p", type=float, default=0.0,
                        help="LBF-09 opt-in genuine equal-cost source branches; 0 preserves legacy")

    run = commands.add_parser("run", help="Execute fixed eight-call Lean proof search")
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--support-roots", type=int, default=12)
    run.add_argument("--query-roots", type=int, default=12)
    run.add_argument("--query-split", choices=("calibration", "ordinary_test", "ood_family"),
                     default="calibration")
    run.add_argument("--checks", type=int, default=8)
    run.add_argument("--depth", type=int, default=3)
    run.add_argument("--max-seconds", type=float, default=360.0)
    run.add_argument("--request-timeout", type=float, default=30.0)
    run.add_argument("--cert-timeout", type=int, default=120)
    run.add_argument("--fast-lr", type=float, default=1.0)
    run.add_argument("--fast-scale", type=float, default=2.0)

    audit = commands.add_parser("replay", help="Re-execute every tactic against the same Lean binaries")
    audit.add_argument("--suite", type=Path, required=True)
    audit.add_argument("--session", type=Path, required=True)
    audit.add_argument("--out", type=Path, required=True)
    audit.add_argument("--request-timeout", type=float, default=30.0)
    export_cmd = commands.add_parser("export", help="Export train-only verified episodes after independent replay")
    export_cmd.add_argument("--suite", type=Path, required=True)
    export_cmd.add_argument("--session", type=Path, required=True)
    export_cmd.add_argument("--replay-receipt", type=Path, required=True)
    export_cmd.add_argument("--out", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "freeze":
        result = create_suite(args.out, seed=args.seed, per_split=args.per_split,
                              equal_alt_p=args.equal_alt_p)
    elif args.command == "run":
        if not 1 <= args.support_roots <= 1024 or not 1 <= args.query_roots <= 1024:
            raise ValueError("case counts must be 1..1024")
        if not (1 <= args.checks <= 32 and 1 <= args.depth <= 16 and args.max_seconds > 0):
            raise ValueError("invalid proof budgets or maximum wall time")
        result = execute(args)
    elif args.command == "replay":
        result = replay(args)
    elif args.command == "export":
        from .live_export import export
        result = export(args.session, args.suite, args.replay_receipt, args.out)
    else:
        raise ValueError("unknown command")
    print(canonical(result))
    return result


if __name__ == "__main__":
    main()
