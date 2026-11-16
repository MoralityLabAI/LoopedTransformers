"""AF-24B original frozen AF-23 networks on source-registered new grammars.

All original checkpoint ZIPs are GitHub Actions artifacts from immutable
AF-23 Track B run 43812708916. ZIP/member digests, original source hashes
and Torch/NumPy model parity are verified before inference. No training code
is called. Original AF-23 Bellman scores and AF-24 symbolic code are reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import zipfile

import torch

from experiments.activation_funnel_af23 import branching_train as original_train
from experiments.activation_funnel_af24 import symbolic
from . import gym, transfer_planner

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL_RUN = 43812708916
ORIGINAL_ZIPS = {
    17: (14200844340, "0440fcda5f8ca8562da385851d5765fbe9784d4d0800922e5d8633c082cfcffe"),
    29: (14201501545, "040c48cb4b5e2e02e6b8419521e570a0d68f325bcc8c480c80669c5d57b7bd86"),
    43: (14201486468, "43a36b5273858646779f8cedec1d9c84a2fea0e54d4f06b53d7942814263aae5"),
    61: (14201322685, "69b6f8ebf0d933fea02641ac9916646850358296bad65b9076e1ce5b2c3b95b2"),
    73: (14201536438, "64c4aaa3e82b40f733c9e00fc6cbf41511c68db54b8077f5d73662a5cdb30dc9"),
    89: (14200829424, "5fbe327d7ea494c1c097f3ee510b87eabd0ce48bec8ea61aa1f323fbc141aea0"),
    103: (14201820417, "32009b1ed5afc50f04b7863593ea49950d4064fa0857e11f516ab3e03bd6ef44"),
    131: (14201287338, "1fa688d73abe4c80517ff4f1a824e6b400b1778f2713f7009dea512e1f42e856"),
}
NEURAL = ("direct", "vi1", "vi2")
SYMBOLIC = ("fixed", "ancestor_greedy", "closure_direct",
            "bfs_cap16", "bfs_cap64", "shortest")
ALL = (*NEURAL, *SYMBOLIC)
SPLITS = gym.SPLITS

SOURCE_FILES = (
    "PROTOCOL.json","gym.py","transfer_planner.py","evaluate.py",
    "audit.py","aggregate.py","tests/test_source.py",
    "../../.github/workflows/activation-funnel-af24b.yml",
    "../../.github/workflows/activation-funnel-af24b-preflight.yml",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def hashes():
    return {p: sha256((HERE/p).read_bytes()) for p in SOURCE_FILES}


def original_artifact(zip_dir, seed):
    """Fetch only original signed artifacts, no recreated/retrained weights."""
    if seed not in gym.SEEDS:
        raise ValueError("unregistered seed")
    p = Path(zip_dir) / f"af23-branch-seed-{seed}.zip"
    if not p.is_file():
        raise FileNotFoundError(f"missing original AF23 seed {seed} ZIP")
    raw = p.read_bytes()
    artifact_id, expected = ORIGINAL_ZIPS[seed]
    if sha256(raw) != expected:
        raise ValueError(f"seed {seed}: original GH Actions artifact hash mismatch")
    with zipfile.ZipFile(p) as archive:
        names = set(archive.namelist())
        required = {
            "training/train_receipt.json",
            "training/train_receipt.sha256",
            "training/policy.pt",
            "training/policy.npz",
            "evaluation/evaluation.json",
            "independent_audit.json",
            "source_revision.txt",
        }
        if not required <= names:
            raise ValueError("missing original AF23 signed member")
        source_revision = archive.read("source_revision.txt").decode().strip()
        if source_revision != "1312d138b685d1054d527e56e16e1ea757a90b08":
            raise ValueError("original AF23 model training source revision changed")
        training = json.loads(archive.read("training/train_receipt.json"))
        audit = json.loads(archive.read("independent_audit.json"))
        ev = json.loads(archive.read("evaluation/evaluation.json"))
        if (training["seed"] != seed or not audit["verified"] or
                ev["seed"] != seed or ev["experiment"] != "AF-23" or
                ev["track"] != "B"):
            raise ValueError("wrong original AF23 checkpoint/audit provenance")
        receipt_raw = archive.read("training/train_receipt.json")
        if sha256(receipt_raw) != archive.read("training/train_receipt.sha256").decode().strip():
            raise ValueError("original AF23 train receipt hash mismatch")
        for ext in ("pt","npz"):
            if sha256(archive.read(f"training/policy.{ext}")) != training["weights_sha256"][ext]:
                raise ValueError("original AF23 model checkpoint hash mismatch")
        if audit["training_sha256"] != sha256(receipt_raw):
            raise ValueError("original AF23 independent audit source mismatch")
        payload = {k: archive.read(k) for k in required
                   if k.startswith("training/")}
    return payload, {
        "artifact_id":artifact_id, "zip_sha256":expected,
        "source_revision":source_revision,
        "training_sha256":sha256(receipt_raw),
        "checkpoint_sha256":training["weights_sha256"],
    }


def load_frozen_original(zip_dir, seed):
    payload, provenance = original_artifact(zip_dir, seed)
    with tempfile.TemporaryDirectory(prefix="af24b-original-model-") as tmp:
        folder = Path(tmp)
        for name, blob in payload.items():
            dest = folder/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(blob)
        receipt, model = original_train.load_model(folder/"training", seed, smoke=False)
        if receipt["seed"] != seed or receipt["parameters"] != 6036:
            raise ValueError("trained model metadata differs")
        return model, provenance


def symbolic_episode(c, mode, seed=17):
    if mode not in SYMBOLIC:
        raise ValueError("nonregistered symbolic control")
    graph = symbolic.PublicHorn.from_public_features(gym.public_features(c))
    state = graph.initial
    d = gym.shortest_steps(c)
    budget = d + 1 if d is not None else 2
    apos, _ = gym.physical_maps(c)
    physical_to_logical = {p:a for a,p in enumerate(apos)}
    history = []
    for t in range(budget):
        if graph.goal in state:
            break
        if mode == "shortest":
            source_state = frozenset(physical_to_logical[p] for p in state)
            action = gym.optimal_actions(c, source_state)[0]
            if action == gym.STOP:
                action = None
            work = {"arm":mode, "action":action,
                    "expanded":0,"rule_examinations":0,
                    "closure_passes":0, "wall_ns":0,
                    "search_exhausted":True, "oracle_only":True}
        else:
            work = symbolic.choose(graph, state, mode, seed, c.id, t)
            action = work["action"]
        before = sorted(physical_to_logical[p] for p in state)
        if action is None:
            after = state
            proof = None
        else:
            new_logical, proof = gym.sound_fire(c, frozenset(before), action)
            after = graph.fire(state, action)
            if after != frozenset(apos[a] for a in new_logical):
                raise ValueError("symbolic control/host rule disagreement")
        history.append({
            "before":before,"enabled":list(gym.enabled(c, frozenset(before))),
            "action":gym.STOP if action is None else action,
            "after":sorted(physical_to_logical[p] for p in after),
            "proof":proof, "work":work,"plan_trace":None,
            "decision_scores":None,
        })
        state = after
        if action is None or graph.goal in state:
            break
    final = frozenset(physical_to_logical[p] for p in state)
    return make_row(c, mode, d, budget, history, final)


def make_row(c, mode, d, budget, actions, final):
    solved = gym.GOAL in final
    stopped = bool(actions and actions[-1]["action"] == gym.STOP)
    return {
        "experiment":"AF-24B", "case_id":c.id, "family":c.family,
        "split":c.split, "mode":mode,
        "initial":sorted(gym.initial_facts(c)),"final":sorted(final),
        "goal_truth":int(d is not None), "optimal_steps":d,
        "budget":budget, "actions":actions, "solved":solved,
        "early_stop":bool(d is not None and not solved and stopped),
        "initial_stop":bool(actions and actions[0]["action"] == gym.STOP),
        "false_accepted_proof":bool(d is None and solved),
        "forward_nodes":sum(int((x.get("plan_trace") or {}).get(
            "nodes_forward_count",0)) for x in actions),
        "rule_examinations":sum(int((x.get("work") or {}).get(
            "rule_examinations",0)) for x in actions),
        "bfs_expansions":sum(int((x.get("work") or {}).get(
            "expanded",0)) for x in actions),
    }


def neural_rollout(model, cases, modes=NEURAL):
    if not modes or any(m not in NEURAL for m in modes):
        raise ValueError("nonregistered neural policy")
    state = {m: [gym.initial_facts(c) for c in cases] for m in modes}
    done = {m: [False] * len(cases) for m in modes}
    histories = {m: [[] for _ in cases] for m in modes}
    distances = [gym.shortest_steps(c) for c in cases]
    budgets = [d+1 if d is not None else 2 for d in distances]
    for t in range(max(budgets)):
        jobs = [(m,i,c,state[m][i]) for m in modes
                for i,c in enumerate(cases)
                if not done[m][i] and t < budgets[i]]
        plans = []
        for start in range(0, len(jobs), 32):
            sub = jobs[start:start+32]
            plans.extend(transfer_planner.decisions(
                model, [c for m,i,c,s in sub],
                [s for m,i,c,s in sub]
            ))
        for (mode,i,c,known), plan in zip(jobs,plans):
            action = plan["selected"][mode]
            after,proof = gym.sound_fire(c,known,action)
            histories[mode][i].append({
                "before":sorted(known),"enabled":list(gym.enabled(c,known)),
                "action":action,"after":sorted(after),"proof":proof,
                "decision_scores":plan["scores"][mode],
                "plan_trace":plan["trace"],
                "host_graph_sha256":plan["host_graph_sha256"],
            })
            state[mode][i] = after
            if action == gym.STOP or gym.GOAL in after:
                done[mode][i] = True
    return {
        mode:[make_row(c,mode,distances[i],budgets[i],
                       histories[mode][i],state[mode][i])
              for i,c in enumerate(cases)]
        for mode in modes
    }


def summarize(rows):
    yes = [r for r in rows if r["goal_truth"]]
    no = [r for r in rows if not r["goal_truth"]]
    if not yes or not no:
        raise ValueError("missing positive or negative evaluation control")
    return {
        "cases":len(rows), "provable":len(yes), "unprovable":len(no),
        "solved":sum(bool(r["solved"]) for r in yes),
        "proof_success":sum(bool(r["solved"]) for r in yes)/len(yes),
        "premature_stop":sum(bool(r["early_stop"]) for r in yes),
        "first_premature_stop":sum(bool(r["initial_stop"]) for r in yes),
        "correct_initial_stop":sum(bool(r["initial_stop"]) for r in no),
        "correct_initial_stop_rate":sum(bool(r["initial_stop"]) for r in no)/len(no),
        "false_accepted_proofs":sum(bool(r["false_accepted_proof"]) for r in rows),
        "neural_forward_nodes":sum(r["forward_nodes"] for r in rows),
        "symbolic_rule_examinations":sum(r["rule_examinations"] for r in rows),
        "symbolic_bfs_expansions":sum(r["bfs_expansions"] for r in rows),
    }


def evaluate(seed, zip_dir, out_dir, smoke=False):
    if seed not in gym.SEEDS:
        raise ValueError("unregistered seed")
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError("AF24B target folder must be fresh")
    out_dir.mkdir(parents=True)
    manifest = gym.manifest()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, provenance = load_frozen_original(zip_dir, seed)
    all_rows, scores = [], {}
    for split in SPLITS:
        cs = gym.cases_for(split)
        if smoke:
            # Functional source-only debug case subset, not a scored result.
            cs = [c for c in cs if c.designed_positive][:2] + [
                c for c in cs if not c.designed_positive][:2]
        groups = neural_rollout(model, cs)
        for mode in SYMBOLIC:
            groups[mode] = [symbolic_episode(c,mode,seed) for c in cs]
        if tuple(groups) != ALL:
            raise AssertionError("missing registered arm")
        scores[split] = {mode:summarize(rows) for mode,rows in groups.items()}
        for rows in groups.values():
            all_rows.extend(rows)
    data = "".join(json.dumps(r,sort_keys=True,allow_nan=False)+"\n"
                   for r in all_rows).encode()
    (out_dir/"episodes.jsonl").write_bytes(data)
    original_hashes = hashes()
    result = {
        "experiment":"AF-24B","seed":seed,"smoke":smoke,
        "registered_protocol_sha256":sha256((HERE/"PROTOCOL.json").read_bytes()),
        "registered_manifest_sha256":manifest["manifest_sha256"],
        "source_hashes":original_hashes,
        "original_af23_checkpoint":provenance,
        "episode_count":len(all_rows), "episodes_sha256":sha256(data),
        "all_arms":list(ALL),
        "scores":scores,
        "training_updates":0,"model_source":"AF23 original immutable Torch 2.5.1 checkpoint",
        "scientific_verdict":"SMOKE_ONLY" if smoke else "AWAITING_EIGHT_SEED_INDEPENDENT_AGGREGATE",
    }
    (out_dir/"evaluation.json").write_text(
        json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,choices=gym.SEEDS,required=True)
    parser.add_argument("--original-zip-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--smoke",action="store_true")
    a=parser.parse_args()
    r=evaluate(a.seed,a.original_zip_dir,a.out,a.smoke)
    print(json.dumps({
        "seed":a.seed,"smoke":a.smoke,"episode_count":r["episode_count"],
        "manifest_sha256":r["registered_manifest_sha256"],
        "scientific_verdict":r["scientific_verdict"],
    }))


if __name__=="__main__":
    main()
