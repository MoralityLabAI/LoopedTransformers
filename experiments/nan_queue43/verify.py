"""Verify contained launch, descendant wait and model terminal custody on CPU."""
import argparse
import json
from pathlib import Path
import time

from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from . import supervisor
from .job import child_tree


def no_gpu_descendants(work):
    # The parent returns before its own delayed interpreter. A launcher exit is too early.
    marker = work / "grandchild_done.json"
    script = ("import subprocess,sys; "
              "subprocess.Popen([sys.executable,'-B','-c',sys.argv[1],sys.argv[2]])")
    descendant = ("import json,os,sys,time; time.sleep(3); "
                  "open(sys.argv[1],'x').write(json.dumps({'pid':os.getpid()}))")
    logs = work / "tree.log"
    launcher_gone_with_child = False
    observed = set()
    began = time.monotonic()
    with child_tree([str(supervisor.GPU_PYTHON), "-X", "utf8", "-B", "-u", "-c", script,
                     descendant, str(marker)], supervisor.REPO, logs) as tree:
        assert tree.launcher_creation() > 0
        while tree.accounting()[0]:
            if time.monotonic() - began > 20:
                raise TimeoutError("CPU job-tree fixture did not close")
            observed.update(tree.active_pids())
            try:
                if tree.launcher_code() == 0:
                    launcher_gone_with_child = True
            except RuntimeError:
                pass
            time.sleep(0.1)
        assert tree.launcher_code() == 0
        final_active, process_count = tree.accounting()
        valid = json.loads(marker.read_bytes())
        assert launcher_gone_with_child and final_active == 0 and process_count >= 3
        assert valid["pid"] in observed and valid["pid"] != tree.pid
    return {"launcher_ended_before_tree": launcher_gone_with_child, "tree_members_total": process_count,
            "tree_active_at_release": final_active, "delayed_interpreter_observed": True}


def receipt_fixture(work):
    target = supervisor.runtime.TARGETS["nf4"]
    lock = sha(supervisor.pilot.SEAL)
    path = work / "backend"
    path.mkdir()
    write_new(path / "preflight.json", {"fixture": True})
    write_new(path / "admission.json",
              {"backend": "nf4", "source_manifest_sha256": lock, "pid": 1234, "published_commit": "fixture"})
    write_new(path / "terminal.json",
              {"backend": "nf4", "source_manifest_sha256": lock, "published_commit": "fixture",
               "status": "native_gate_failed", "registered_focused_worlds": 16, "closed_focused_worlds": 0,
               "target_before": target, "target_after": target, "checkpoint_verified_after": True,
               "quant_unchanged": True, "preflight_sha256": sha(path / "preflight.json")})
    read = lambda pid: supervisor.verify_model(path, "nf4", lock, {"pid": pid})
    assert read(1234) == "native_gate_failed"
    rejected = 0
    try:
        read(8131)
    except ValueError:
        rejected += 1
    (path / "closed.jsonl").write_bytes(b"{}\n")
    try:
        read(1234)
    except ValueError:
        rejected += 1
    assert rejected == 2
    return {"native_qualification_refusal_recognized": True, "bad_terminal_controls_rejected": rejected}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    supervisor.below_normal()
    supervisor.verify_local()
    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    result = {"status": "passed", "local_manifest_sha256": sha(supervisor.MANIFEST),
              "worker_fixture": no_gpu_descendants(work), "terminal_fixture": receipt_fixture(work),
              "model_calls": 0}
    write_new(work / "verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
