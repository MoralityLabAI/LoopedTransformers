"""Admit grouped BF16 after the authenticated predecessor queue and NF4 host audit."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.nan_boundrescue109 import lane as serial_lane, roster
from experiments.nan_boundrescue109.host import MODE
from experiments.nan_delay39 import pilot as delay
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_readpath37.run import write_new
from experiments.nan_rescuegroup110 import verify as grouping
from experiments.nan_vocab83.custody import vocabulary

from .device import DOSE, FIXTURE_SEEDS, device
from .worker import OFFLOADED_LAYERS, VARIANT


REPO = serial_lane.REPO
HERE = Path(__file__).resolve().parent
MANIFEST = HERE/"RUNTIME_MANIFEST.json"
RECEIPT = HERE/"RUNTIME_VERIFICATION.json"
WORK = Path("E:/codex-artifacts/NanbeigeRescueGroup111-20261123/bf16")


def registration():
    parent = serial_lane.verified()
    serial_lane.commissioned(parent)
    local = [HERE/n for n in (".gitattributes","__init__.py","PROTOCOL.md","device.py","host.py",
                             "lane.py","worker.py","verify.py")]
    grouped = grouping.HERE/"CPU_VERIFICATION.json"
    proof = json.loads(grouped.read_bytes())
    if (grouping.registration()!=json.loads(grouping.MANIFEST.read_bytes())
            or proof["status"]!="passed" or proof["source_manifest_sha256"]!=sha(grouping.MANIFEST)
            or not proof["full_root_exact"] or proof["grouped_stage_copies"]!=90
            or proof["pretrained_forwards"]!=0):
        raise ValueError("The grouped native CPU evidence differs from its source")
    files = local + [serial_lane.MANIFEST,serial_lane.COMM,grouping.MANIFEST,grouped] + [REPO/n for n in (
        "experiments/nan_rescuegroup110/group.py","experiments/nan_stagegroup103/verify.py",
        "experiments/nan_boundrescue109/host.py","experiments/nan_boundrescue109/verify_runtime.py",
        "experiments/nan_vocab83/verify.py","experiments/nan_stream36/device_preflight.py",
        "experiments/nan_readout34/runtime.py","experiments/nan_vocab83/batches.py")]
    r = json.loads(roster.RECEIPT.read_bytes())
    return {"experiment":roster.EXPERIMENT,"execution_variant":VARIANT,"backend":"bf16",
            "predecessor_runtime_sha256":parent,"max_position":r["max_position"],"roots":r["roots"],
            "baseline_batches":r["b2_batches"],"seeds":list(FIXTURE_SEEDS),"capacity_edit_dose":DOSE,
            "offloaded_layers":OFFLOADED_LAYERS,"maximum_simultaneous_B2_branches":2,"limits":runtime.LIMITS,
            "files_sha256":{p.relative_to(REPO).as_posix():sha(p) for p in files}}


def verified():
    if not __debug__ or Path.cwd()!=REPO or json.loads(MANIFEST.read_bytes())!=registration():
        raise ValueError("The canonical grouped runtime source differs from registration")
    return sha(MANIFEST)


def commissioned(frozen):
    proof = json.loads(RECEIPT.read_bytes())
    if (proof["runtime_sha256"]!=frozen or proof["status"]!="passed" or proof["pretrained_forwards"]!=0
            or proof["cpu_capacity"]["probes"]!=2 or proof["cpu_capacity"]["exact_grouped_branches"]!=4
            or proof["host"]["fixture_closed_roots"]!=1 or not proof["host"]["tampered_grouped_count_refused"]
            or not proof["host"]["tampered_capacity_refused"] or not proof["host"]["synthetic_refused_as_pretrained"]):
        raise ValueError("The staged capacity and independent grouped-host fixture did not close")


def admit(args):
    args.backend = "bf16"
    serial_lane.admit(args,serial_lane.ARTIFACT/"bf16")
    if WORK.exists():
        raise FileExistsError("The grouped BF16 condition is already occupied")


def publication():
    commit = serial_lane.publication()
    for path in (MANIFEST,RECEIPT):
        name = path.relative_to(REPO).as_posix()
        remote = subprocess.check_output(["git","show",f"{commit}:{name}"],timeout=30,creationflags=runtime.FLAGS)
        if remote!=path.read_bytes():
            raise ValueError("Grouped CUDA freeze differs from the authenticated publication")
    return commit


def host(args, qualify=False):
    env = os.environ.copy()
    env.update(USE_TORCH="0",USE_TF="0",USE_FLAX="0",CUDA_VISIBLE_DEVICES="")
    command = [sys.executable,"-B","-X","utf8","-m","experiments.nan_rescuegpu111.host",
               "--work",str(WORK),"--source-dir",str(args.source_dir),"--snapshot",str(args.snapshot)]
    if qualify:
        command.append("--qualification")
    subprocess.run(command,env=env,timeout=135,check=True,creationflags=runtime.FLAGS)


def execute(args, admission):
    if {name:version(name) for name in runtime.GPU_PACKAGES}!=runtime.GPU_PACKAGES:
        raise RuntimeError("Pinned CUDA interpreter packages differ")
    with delay.resources(WORK) as guard:
        payload = roster.worker_source(args.source_dir,args.snapshot,guard)
        pins = json.loads(roster.RECEIPT.read_bytes())["tokenizer_pins"]
        _,classes = vocabulary(args.snapshot,guard,pins)
        import numpy as np
        import torch

        device(args,WORK,guard,payload,classes,admission,np,torch,registration()|{"repo":REPO},
               lambda:host(args,qualify=True))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze",action="store_true")
    cli.add_argument("--check-admission",action="store_true")
    for name in ("snapshot","source-dir","adapter"):
        cli.add_argument("--"+name,type=Path)
    args = cli.parse_args()
    if args.freeze:
        write_new(MANIFEST,registration())
        return
    if not all((args.snapshot,args.source_dir,args.adapter)):
        cli.error("Require the exact checkpoint, frozen source and isolated adapter")
    frozen = verified()
    commissioned(frozen)
    if "torch" in sys.modules:
        raise RuntimeError("A model package loaded before GPU admission")
    if args.check_admission:
        admit(args)
        print("Local prerequisites passed",flush=True)
        return
    with runtime.exclusive_lane():
        admit(args)
        commit = publication()
        WORK.mkdir(parents=True,exist_ok=False)
        proof = {"experiment":roster.EXPERIMENT,"backend":"bf16","execution_variant":VARIANT,
                 "runtime_sha256":frozen,"published_commit":commit,"mode":MODE,"pid":os.getpid()}
        write_new(WORK/"admission.json",proof)
        runtime.environment(WORK)
        try:
            execute(args,proof)
            host(args)
        except BaseException as error:
            write_new(WORK/"attempt_failure.json",{"error_type":type(error).__name__,"message":str(error)})
            raise


if __name__=="__main__":
    main()
