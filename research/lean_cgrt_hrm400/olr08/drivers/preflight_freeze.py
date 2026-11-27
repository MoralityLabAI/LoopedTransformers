"""Run the registered OLR-08A full-profile prelude (contract, scope sentinel, registry
freeze) under the same CPU/RAM limits as launch.py, and stop before the first outcome
stage. The package code is imported unchanged; only run08.stage is replaced by a guard
that raises before any menu or search outcome is collected. Re-running launch.py on the
same --out then resumes from the sealed registry under the identical contract.
"""
import os
import sys
from pathlib import Path

PKG = Path(sys.argv[1]).resolve()
sys.argv = [sys.argv[0], *sys.argv[2:]]
sys.path.insert(0, str(PKG))

import argparse  # noqa: E402

import launch  # noqa: E402


class StopBeforeOutcomes(Exception):
    pass


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--ram-gib", type=float, default=12)
    a, rest = p.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    import run08
    args = run08.cli()
    if args.profile != "full" or a.ram_gib <= 0:
        raise SystemExit("preflight is for the capped full profile")
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                      OPENBLAS_NUM_THREADS=str(args.threads),
                      LEAN_NUM_THREADS=str(max(1, args.threads // args.workers)))
    peak, kind = launch.limit_memory(a.ram_gib)
    import psutil
    proc = psutil.Process()
    if hasattr(proc, "cpu_affinity"):
        proc.cpu_affinity(proc.cpu_affinity()[:args.threads])

    def guard(out, name, *rest, **kw):
        raise StopBeforeOutcomes(name)

    run08.stage = guard
    try:
        run08.main(args)
    except StopBeforeOutcomes as e:
        print("PREFLIGHT_STOPPED_BEFORE", e, "memory_limit", kind, "peak_job_commit", peak() if peak else None,
              flush=True)
        return
    raise SystemExit("run08.main returned without reaching an outcome stage")


if __name__ == "__main__":
    main()
