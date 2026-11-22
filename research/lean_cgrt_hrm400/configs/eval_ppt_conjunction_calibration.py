"""Evaluate frozen conjunction policies on the independent calibration partition."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from configs.freeze_ppt_conjunction_followup import INTERVENTIONS, SCHEMA, encode_worlds
from configs.probe_ppt_hybrid_bank import INTERACTION_DOSES
from configs.run_ppt_conjunction_pilot import ARMS
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase, load
from ppt_gym.ram_limits import TreeBudget


def read_calibration(registry, input_file):
    reg = json.loads(registry.read_bytes())
    doc = json.loads(input_file.read_bytes())
    if (doc["schema"] != SCHEMA or reg["schema"] != SCHEMA or doc["scope"] != "calibration"
            or doc["registry_digest"] != digest(reg) or reg["arms"] != list(ARMS)
            or reg["root_interventions"] != list(INTERVENTIONS)):
        raise ValueError("unregistered calibration input")
    pool = reg["calibration"]
    generated = encode_worlds(pool["pairs"], "calibration", pool["seed"])
    if doc["cases"] != [asdict(case) for case in generated]:
        raise ValueError("calibration case identities diverged from the registry")
    return reg, [SearchCase(**row) for row in doc["cases"]]


def worker(args, tree, reg, cases):
    from configs.analyze_ppt_hybrid_replication import read_arm
    from configs.probe_ppt_hybrid_bank import EditedTransportPolicy, RootInteractionDose
    from lcgrt import config
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym import ram_store
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.hybrid_rl.graph_model import GraphProofPolicy
    from ppt_gym.live_lean import fingerprint
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_campaign import configure
    from ppt_gym.ram_learning import tensor_hash

    if sha256_file(args.pilot) != reg["pilot_sha256"]:
        raise ValueError("source development cohort changed")
    original = json.loads(args.pilot.read_bytes())
    seed, = original["registration"]["model_seeds"]
    checkpoint_arm = "hybrid" if args.arm in INTERVENTIONS else args.arm
    source = args.sessions / f"{seed}-{checkpoint_arm}.sqlite"
    pilot_cases = {case.root: case for case in load(args.pilot)["dev"]}
    trial = read_arm(source, pilot_cases, reg["pilot_sha256"])
    model_spec = trial["manifest"]
    kind = tuple(model_spec[key] for key in ("architecture", "objective", "head_only",
                                             "residual_fact_transport", "premise_interaction"))
    if (kind != ARMS[checkpoint_arm] or model_spec["seed"] != seed or not model_spec["graph_context"]
            or model_spec["alias_training"] != "contrast-search" or set(trial["stages"]) != {"initial", "final"}):
        raise ValueError("wrong checkpoint for the registered calibration arm")
    with ram_store.open_store(source) as db:
        weights = ram_store.checkpoint(db)["model"]
    if tensor_hash(weights) != trial["final_weights_sha256"]:
        raise ValueError("checkpoint changed during read")
    configure(tree.threads, seed)
    model = GraphProofPolicy(kind[0], model_spec["width"], residual_transport=kind[3], premise_interaction=kind[4])
    model.load_state_dict(weights)
    edited = EditedTransportPolicy(model, interaction_only=True) if args.arm in INTERVENTIONS else None
    if args.root_dose:
        edited = RootInteractionDose(model, args.root_dose)
    args.cpu_seconds, args.max_macs = model_spec["cpu_seconds"], model_spec["max_macs"]
    identity = {
        "arm": args.arm, "registry_sha256": sha256_file(args.registry), "calibration_sha256": sha256_file(args.input),
        "pilot_session_sha256": trial["session_sha256"], "checkpoint_sha256": trial["final_weights_sha256"],
        "verifier": model_spec["verifier"], "engine_sha256": model_spec["source_sha256"],
        "proof_cpu_seconds": args.cpu_seconds, "max_macs": args.max_macs, "threads": tree.threads,
    }
    arm_id = args.arm
    if args.root_dose:
        arm_id = f"{args.arm}-{args.root_dose}"
        identity.update(arm=arm_id, root_dose=args.root_dose)
    engine = Path(sys.modules["ppt_gym.hybrid_rl.__main__"].__file__).parent
    if model_spec["source_sha256"] != {p.name: sha256_file(p) for p in sorted(engine.glob("*.py"))}:
        raise ValueError("inference implementation differs from the trained checkpoint")
    with LeanRepl(timeout=15) as repl, ram_store.open_store(args.output / (arm_id + ".sqlite"), write=True) as db:
        if fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version) != model_spec["verifier"]:
            raise ValueError("native Lean toolchain changed since training")
        ram_store.initialize(db, identity)
        completed = [row["episode"]["root"] for row in ram_store.ledger(db)]
        if completed != [case.root for case in cases[:len(completed)]]:
            raise ValueError("calibration ledger is out of order")
        for case in cases[len(completed):]:
            if edited is None:
                ep = rollout(repl, tree, case, model, args)
            else:
                with edited.edit(args.arm):
                    ep = rollout(repl, tree, case, edited, args)
                    ep["interaction_reads_edited"] = edited.hits
            ep["native_replay"] = replay_search(repl, case, ep)
            with db:
                ram_store.append(db, {"episode": ep})
    print(f"{arm_id} completed {len(cases)} calibration worlds", flush=True)


def campaign(args, tree):
    def run(arm, dose=None):
        arm_id = f"{arm}-{dose}" if dose else arm
        cmd = [sys.executable, "-u", "-m", "configs.eval_ppt_conjunction_calibration",
               str(args.registry), str(args.input), str(args.pilot), str(args.sessions), str(args.output),
               "--arm", arm]
        if dose:
            cmd.extend(("--root-dose", dose))
        with (args.output / (arm_id + ".log")).open("ab") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(arm_id, result.returncode, flush=True)
        return arm_id, result.returncode

    specs = ([(arm, dose) for arm in ("hybrid", "hybrid_imitation") for dose in INTERACTION_DOSES]
             if args.dose_study else [(arm, None) for arm in (*ARMS, *INTERVENTIONS)])
    statuses = {}
    try:
        with ThreadPoolExecutor(max_workers=len(specs)) as pool:
            futures = [pool.submit(run, *spec) for spec in specs]
            statuses = dict(future.result() for future in futures)
    finally:
        receipt = {"statuses": statuses, "memory": tree.stats(), "registry_sha256": sha256_file(args.registry),
                   "calibration_sha256": sha256_file(args.input)}
        (args.output / "calibration-resources.json").write_text(canonical(receipt) + "\n", encoding="utf-8")
    if len(statuses) != len(specs) or any(statuses.values()):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registry", "input", "pilot", "sessions", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--arm", choices=(*ARMS, *INTERVENTIONS))
    parser.add_argument("--root-dose", choices=INTERACTION_DOSES)
    parser.add_argument("--dose-study", action="store_true")
    args = parser.parse_args()
    if args.root_dose and args.arm not in ("hybrid", "hybrid_imitation"):
        parser.error("--root-dose requires a full hybrid RL or imitation arm")
    if args.dose_study and args.arm:
        parser.error("--dose-study compares both original full checkpoints")
    reg, cases = read_calibration(args.registry, args.input)
    args.output.mkdir(exist_ok=True)
    tree = TreeBudget(2 if args.arm else 6, 2)
    try:
        if args.arm:
            worker(args, tree, reg, cases)
        else:
            campaign(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
