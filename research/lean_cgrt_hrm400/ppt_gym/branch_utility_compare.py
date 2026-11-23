"""LSA-02J read-only native Lean branch-utility calibration and paired proof audit.

Do not run this job until *all* frozen model and random-DFS proof jobs have
finished. Only the TRAIN shard contributes optimizer updates. Every test
action's viability and shortest cost is obtained by native Lean exploration
after all policy episodes, and every shortest oracle script is separately
compiled in Lean. The 32 theorem roots are the independent statistical units.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import torch

from lcgrt import config
from lcgrt.certify import certify, decl_name
from lcgrt.extract import explore
from lcgrt.lean_repl import LeanRepl

from .branch_utility import BranchUtility, action_features, public_horn_costs
from .branch_utility_campaign import SCHEMA as NATIVE_SCHEMA, POLICIES
from .branch_utility_source import label_frame
from .branch_utility_train import SCHEMA as TRAIN_SCHEMA, SEEDS
from .cli import checkpoint_load
from .data import canonical, sha256_file
from .live_suite import load_suite
from .native_train import _load_teacher, _snapshot_trajectory


SCHEMA = "lsa02j.native_viability_root_comparison.v1"
DRAWS = 2000
UNIFORM_SEEDS = (17, 31, 47, 61, 83)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def _write(path, obj):
    Path(path).write_text(canonical(obj) + "\n", encoding="utf-8")


def _auth_files(folder, artifacts):
    if not artifacts:
        raise ValueError("missing native proof/compiler hash custody")
    for name, digest in artifacts.items():
        if name != Path(name).name or sha256_file(Path(folder) / name) != digest:
            raise ValueError("native checkpoint or proof receipt file was modified")


def _bootstrap(values):
    if len(values) < 2:
        raise ValueError("root bootstrap needs independent theorem roots")
    rng = random.Random(20261203)
    n = len(values)
    v = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(DRAWS)
    )
    return [v[int(.025*DRAWS)],v[int(.975*DRAWS)-1]]


def _postscore_calibration(repl, suite, teacher, out):
    if out.exists():
        raise ValueError("post-scoring reference destination must be fresh")
    cases, source = load_suite(suite, "calibration", 32)
    if _read(teacher / "manifest.json")["source"]["suite_manifest_sha256"] != source["suite_manifest_sha256"]:
        raise ValueError("native trainer and calibration suite ancestry changed")
    collected = []
    refs = []
    for case in cases:
        oracle = explore(repl, case.theorem())
        frames = _snapshot_trajectory(repl, case, oracle)
        collected.extend(label_frame(r, oracle) for r in frames)
        refs.append((case.tid, case.theorem(), oracle["proof"]["script"]))
    out.mkdir(parents=True)
    audit = certify(refs, timeout=420, keep=out/"standalone.lean")
    if audit.get("timed_out") or any(
        audit.get("status",{}).get(decl_name(tid)) != "passed"
        for tid,_,_ in refs
    ):
        raise ValueError("native calibration reference shortest proof failed independent Lean compiler")
    frames_path = out/"calibration_utility.jsonl"
    frames_path.write_text(
        "".join(canonical(r)+"\n" for r in collected), encoding="utf-8",
    )
    _write(out/"compiler_receipt.json",audit)
    outmeta = {
        "schema": "lsa02j.native_calibration_branch_labels.v1",
        "split": "calibration",
        "suite_manifest_sha256": source["suite_manifest_sha256"],
        "teacher_sha256": sha256_file(teacher/"manifest.json"),
        "independent_calibration_roots": len(cases),
        "native_reference_proofs_compiled": len(refs),
        "public_frames": len(collected),
        "candidate_actions": sum(len(r["candidates"]) for r in collected),
        "mixed_finite_forks": sum(r["has_mixed_finite_actions"] for r in collected),
        "mixed_finite_roots": len({r["root"] for r in collected if r["has_mixed_finite_actions"]}),
        "after_all_model_native_runs": True,
        "calibration_optimizer_updates": 0,
        "locked_test_opened": False,
        "artifacts": {
            n: sha256_file(out/n) for n in (
                "calibration_utility.jsonl","compiler_receipt.json","standalone.lean"
            )
        },
    }
    _write(out/"manifest.json",outmeta)
    return outmeta


@torch.no_grad()
def _score_calibration(model, head, frames):
    byroot = defaultdict(list)
    for row in frames:
        packet, menu = row["goal"], row["candidates"]
        output = model(
            __import__("ppt_gym.typed_state",fromlist=["encode_public_packet"]).encode_public_packet(
                packet, menu, view="pp96", max_bytes=384,
            ), menu, depth=2, checks_remaining=row["checks_remaining"],
            middle_order="ab",
        )[-1]
        vals = head(output["features"], action_features(packet,menu))
        probabilities = vals[:,0].sigmoid().tolist()
        costs = vals[:,1].sigmoid().tolist()
        labels = row["finite_viability"]
        target_cost = row["minimum_whole_state_tactic_cost"]
        point = {
            "all_action_accuracy": sum(
                int((p >= .5) == bool(y)) for p,y in zip(probabilities,labels)
            ) / len(labels),
            "brier": sum((p-y)**2 for p,y in zip(probabilities,labels)) / len(labels),
            "bounded_cost_mae": (
                sum(abs(pred-cost/16) for pred,cost in zip(costs,target_cost) if cost is not None)
                / max(1,sum(c is not None for c in target_cost))
            ),
        }
        if row["has_mixed_finite_actions"]:
            action = max(range(len(menu)),key=lambda i: probabilities[i]-0.5*costs[i])
            point["mixed_fork_correct"] = int(labels[action] == 1)
            point["mixed_fork_random_menu"] = sum(labels)/len(labels)
        classic = public_horn_costs(packet,menu)
        point["public_structural_cost_exact"] = int(classic == target_cost)
        byroot[row["root"]].append(point)
    return {
        root: {
            "states": len(samples),
            "mixed_forks": sum("mixed_fork_correct" in x for x in samples),
            "viability_accuracy": sum(x["all_action_accuracy"] for x in samples)/len(samples),
            "brier": sum(x["brier"] for x in samples)/len(samples),
            "cost_mae": sum(x["bounded_cost_mae"] for x in samples)/len(samples),
            "mixed_fork_accuracy": (
                sum(x.get("mixed_fork_correct",0) for x in samples)/
                sum("mixed_fork_correct" in x for x in samples)
                if any("mixed_fork_correct" in x for x in samples) else None
            ),
            "mixed_random_baseline": (
                sum(x.get("mixed_fork_random_menu",0) for x in samples)/
                sum("mixed_fork_random_menu" in x for x in samples)
                if any("mixed_fork_random_menu" in x for x in samples) else None
            ),
            "public_structural_exact_rate": sum(x["public_structural_cost_exact"] for x in samples)/len(samples),
        } for root,samples in byroot.items()
    }


def compare(repl, suite: Path, teacher: Path, labels: Path,
            trained: Path, proofs: Path, uniform: Path,
            reference: Path, out: Path):
    if out.exists():
        raise ValueError("output must be fresh for registered post-scoring evaluation")
    teacher_rows, teacher_meta = _load_teacher(teacher)
    source_sha = sha256_file(teacher/"manifest.json")
    training = _read(trained/"manifest.json")
    if not (
        training.get("schema") == TRAIN_SCHEMA
        and training.get("native_teacher_sha256") == source_sha
        and training.get("branch_labels_sha256") == sha256_file(labels/"manifest.json")
        and training.get("backbone_seed_roster") == list(SEEDS)
        and training.get("head_mode_roster") == ["viability","shuffled_sham"]
        and training.get("all_duplicate_fits_bitwise") is True
        and training.get("same_seed_source_draws") is True
        and training.get("frozen_backbone") is True
        and training.get("native_calibration_updates") == 0
        and training.get("test_or_ood_opened") is False
    ):
        raise ValueError("train-only learner or sham provenance invalid")
    uniform_meta = _read(uniform/"manifest.json")
    uniform_replay = _read(uniform/"replay"/"replay.json")
    _auth_files(uniform,uniform_meta.get("artifacts"))
    if not (
        uniform_meta.get("schema") == "lsa02i.native_bounded_backtrack.v1"
        and uniform_meta.get("status") == "complete"
        and uniform_meta.get("control") == "uniform"
        and uniform_meta.get("independent_theorem_roots") == 32
        and uniform_meta.get("effort_budget") == 16
        and uniform_meta.get("model_free_uniform_seeds") == list(UNIFORM_SEEDS)
        and uniform_meta.get("native_teacher_sha256") == source_sha
        and uniform_meta.get("scores",{}).get("uniform_dfs",{}).get("sum_model_calls") == 0
        and uniform_meta.get("backend",{}).get("actual_lean_process") is True
        and uniform_replay.get("all_replays_passed") is True
        and uniform_replay.get("verified_episodes") == 160
        and uniform_replay.get("session_manifest_sha256") == sha256_file(uniform/"manifest.json")
    ):
        raise ValueError("native seeded uniform DFS proof baseline invalid")
    uniform_roots = {r["root"]:r for r in _rows(uniform/"roots.jsonl")}
    if len(uniform_roots) != 32:
        raise ValueError("native random reference theorem count differs")

    all_roots = []
    model_sources = {}
    cells = []
    teacher_sha = sha256_file(teacher/"manifest.json")
    for seed in SEEDS:
        directory = proofs/str(seed)
        manifest = _read(directory/"manifest.json")
        replay = _read(directory/"replay"/"replay.json")
        _auth_files(directory,manifest.get("artifacts"))
        checkpoint = trained/"backbones"/"fits"/f"{seed}-pp96"/"checkpoint.pt"
        if not (
            manifest.get("schema") == NATIVE_SCHEMA
            and manifest.get("status") == "complete"
            and manifest.get("seed") == seed
            and manifest.get("source",{}).get("suite_manifest_sha256") ==
                teacher_meta["source"]["suite_manifest_sha256"]
            and manifest.get("independent_theorem_roots") == 32
            and manifest.get("effort_budget") == 16
            and manifest.get("policy_roster") == list(POLICIES)
            and manifest.get("checkpoint_sha256") == sha256_file(checkpoint)
            and manifest.get("native_teacher_sha256") == teacher_sha
            and manifest.get("branch_source_sha256") == sha256_file(labels/"manifest.json")
            and manifest.get("standalone_reference_certificates") == 32
            and manifest.get("backend",{}).get("actual_lean_process") is True
            and manifest.get("all_native_episode_replays_passed") is True
            and manifest.get("frozen_backbone") is True
            and manifest.get("frozen_utility_heads") is True
            and manifest.get("native_calibration_optimizer_steps") == 0
            and replay.get("all_replays_passed") is True
            and replay.get("verified_episodes") == 32*len(POLICIES)
            and replay.get("session_manifest_sha256") == sha256_file(directory/"manifest.json")
        ):
            raise ValueError("native utility/sham/public root scoring or independent replay custody invalid")
        roots = _rows(directory/"roots.jsonl")
        if len(roots) != 32 or len({r["lineage"] for r in roots}) != 32:
            raise ValueError("seed root cohort contains duplicates")
        all_roots.append({r["root"] for r in roots})
        model_sources[seed] = {r["root"]:r for r in roots}
        cells.append({
            "seed": seed,
            "scores": manifest["scores"],
            "native_attainable": manifest["independent_reference_attainable"],
            "frozen_backbone_checkpoint_sha256": sha256_file(checkpoint),
        })
    if all_roots[0] != all_roots[1] or all_roots[0] != set(uniform_roots):
        raise ValueError("same theorem roots were not used in all model/control arms")

    # Only after every proof result is verified, query native Lean's oracle.
    evidence = _postscore_calibration(repl, suite, teacher, reference)
    _auth_files(reference,evidence["artifacts"])
    frames = _rows(reference/"calibration_utility.jsonl")
    if len({r["root"] for r in frames}) != 32:
        raise ValueError("post-scoring native calibration label lineage list incomplete")
    torch.set_num_threads(1)
    scores_by_seed = {}
    for seed in SEEDS:
        model, cp = checkpoint_load(
            trained/"backbones"/"fits"/f"{seed}-pp96"/"checkpoint.pt"
        )
        if cp["seed"] != seed or cp["native_teacher_sha256"] != source_sha:
            raise ValueError("frozen calibration model provenance differs")
        model.eval()
        scores_by_seed[seed] = {}
        for mode in ("viability","shuffled_sham"):
            path = trained/"heads"/f"{seed}-{mode}"/"head.pt"
            meta = torch.load(path,map_location="cpu",weights_only=True)
            if (
                meta.get("schema") != "lsa02j.branch_utility_head.v1"
                or meta.get("seed") != seed
                or meta.get("mode") != mode
                or meta.get("backbone_checkpoint_sha256") != sha256_file(
                    trained/"backbones"/"fits"/f"{seed}-pp96"/"checkpoint.pt"
                )
                or meta.get("viability_source_sha256") != sha256_file(labels/"manifest.json")
            ):
                raise ValueError("frozen utility probe differs from registered training labels")
            head = BranchUtility(width=32)
            head.load_state_dict(meta["weights"])
            head.eval()
            scores_by_seed[seed][mode] = _score_calibration(model,head,frames)

    roots_report = []
    for root in sorted(all_roots[0]):
        refs = [model_sources[seed][root] for seed in SEEDS]
        if refs[0]["lineage"] != refs[1]["lineage"] or not all(
            r["reference_attainable"] == refs[0]["reference_attainable"] for r in refs
        ):
            raise ValueError("frozen Lean theorem root reference evidence differs")
        uniform_row = uniform_roots[root]
        if uniform_row["lineage"] != refs[0]["lineage"]:
            raise ValueError("native random control used another theorem")
        model_seeds = []
        for seed in SEEDS:
            rr = model_sources[seed][root]
            modes = scores_by_seed[seed]
            model_seeds.append({
                "seed": seed,
                "policies": rr["policies"],
                "viability_validation": {
                    mode: modes[mode][root] for mode in ("viability","shuffled_sham")
                },
            })
        random_avg = sum(
            int(uniform_row["policies"][f"uniform_{s}"]["closed"])
            for s in UNIFORM_SEEDS
        ) / len(UNIFORM_SEEDS)
        roots_report.append({
            "root": root,
            "lineage": refs[0]["lineage"],
            "reference_attainable": refs[0]["reference_attainable"],
            "uniform_dfs_mean": random_avg,
            "model_seeds": model_seeds,
            "utility_minus_frozen": sum(
                int(x["policies"]["utility_dfs"]["closed"]) -
                int(x["policies"]["frozen_dfs"]["closed"]) for x in model_seeds
            )/len(SEEDS),
            "utility_minus_sham": sum(
                int(x["policies"]["utility_dfs"]["closed"]) -
                int(x["policies"]["sham_utility_dfs"]["closed"]) for x in model_seeds
            )/len(SEEDS),
            "utility_minus_public_horn": sum(
                int(x["policies"]["utility_dfs"]["closed"]) -
                int(x["policies"]["public_horn_dfs"]["closed"]) for x in model_seeds
            )/len(SEEDS),
        })
    effect = {}
    for name in ("utility_minus_frozen","utility_minus_sham","utility_minus_public_horn"):
        values = [r[name] for r in roots_report]
        effect[name] = {
            "mean":sum(values)/len(values),
            "root_bootstrap95_descriptive":_bootstrap(values),
        }
    rates = {
        policy: sum(
            int(row["policies"][policy]["closed"])
            for root in roots_report for row in root["model_seeds"]
        ) / (len(SEEDS)*32)
        for policy in POLICIES
    }
    uniform_rate = sum(r["uniform_dfs_mean"] for r in roots_report)/32
    per_seed_nonneg = all(
        sum(
            int(model_sources[seed][root]["policies"]["utility_dfs"]["closed"]) -
            int(model_sources[seed][root]["policies"]["frozen_dfs"]["closed"])
            for root in all_roots[0]
        ) >= 0 for seed in SEEDS
    )
    screen = bool(
        effect["utility_minus_frozen"]["mean"] >= .05
        and effect["utility_minus_sham"]["mean"] >= .05
        and per_seed_nonneg
    )
    learned_advantage = (
        rates["utility_dfs"] >= .40
        and rates["utility_dfs"] - uniform_rate >= .10
        and rates["utility_dfs"] - rates["public_horn_dfs"] >= .10
    )
    viability_fork_roots = len({r["root"] for r in frames if r["has_mixed_finite_actions"]})
    result = {
        "schema": SCHEMA,
        "native_training_teacher_sha256": source_sha,
        "train_branch_source_sha256":sha256_file(labels/"manifest.json"),
        "same_worker_training_sha256":sha256_file(trained/"manifest.json"),
        "native_postscore_labels_sha256":sha256_file(reference/"manifest.json"),
        "independent_train_roots":teacher_meta["independent_lineages"],
        "train_public_frames":len(teacher_rows),
        "independent_calibration_roots":32,
        "calibration_public_frames":len(frames),
        "calibration_mixed_finite_forks":sum(r["has_mixed_finite_actions"] for r in frames),
        "calibration_roots_with_mixed_finite_forks":viability_fork_roots,
        "model_seed_roster":list(SEEDS),
        "uniform_seed_roster":list(UNIFORM_SEEDS),
        "effort_budget":16,
        "policy_roster":list(POLICIES),
        "model_cells":cells,
        "native_closure_rates":rates,
        "uniform_native_dfs_close_rate":uniform_rate,
        "native_reference_attainable":sum(r["reference_attainable"] for r in roots_report),
        "effects":effect,
        "registered_development_screen_passed":screen,
        "learned_competence_vs_strong_controls_passed":learned_advantage,
        "structural_control_is_model_free":True,
        "native_labels_collected_after_all_policy_scoring":True,
        "calibration_optimizer_updates":0,
        "heldout_test_or_ood_opened":False,
        "architecture_promotion_eligible":False,
        "scientific_eligible":False,
        "root_results":roots_report,
        "interpretation":(
            "Two frozen backbone seeds; learned utility vs a shuffled sham, "
            "unmodified DFS, independent native uniform DFS and a stronger "
            "public Horn fixed-point algorithm. Root-clustered exploratory "
            "proof outcomes, no Mathlib transfer claim."
        ),
    }
    out.mkdir(parents=True)
    _write(out/"matched_branch_utility.json",result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for item in ("suite","teacher","labels","trained","proofs","uniform","reference","out"):
        p.add_argument("--"+item,type=Path,required=True)
    p.add_argument("--request-timeout",type=int,default=60)
    a=p.parse_args(argv)
    with LeanRepl(timeout=a.request_timeout,exe=config.REPL_EXE,
                  cwd=config.LEAN_DIR,env=config.lean_env()) as repl:
        r=compare(repl,a.suite,a.teacher,a.labels,a.trained,
                  a.proofs,a.uniform,a.reference,a.out)
    print(canonical({
        "native_closure_rates":r["native_closure_rates"],
        "uniform_native_dfs":r["uniform_native_dfs_close_rate"],
        "calibration_mixed_finite_forks":r["calibration_mixed_finite_forks"],
        "calibration_fork_roots":r["calibration_roots_with_mixed_finite_forks"],
        "paired_effects":r["effects"],
        "development_screen":r["registered_development_screen_passed"],
        "learned_competence":r["learned_competence_vs_strong_controls_passed"],
    }))
    return r


if __name__ == "__main__":
    main()
