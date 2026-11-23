"""LBF-08 native model-visited proof-repair collection and matched CPU training.

Evidence boundaries:
* The behavior model acts before native source-oracle labels are reconstructed.
* Oracle continuation labels are trainer-only, derived from actual Lean transitions.
* Native theorem-root compilation, replay, and hashes certify every admitted root.
* Forced oracle recovery is *teacher assistance*, not autonomous model proof gain.
* All model weights and optimizer updates are CPU-only and train-lineage-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from lcgrt import config
from lcgrt.certify import certify, decl_name
from lcgrt.extract import explore, goal_key, optimal_script
from lcgrt.lean_repl import LeanRepl
from .cli import checkpoint_load
from .core import Config, PlasticProofTransformer, supervised_loss
from .data import canonical, sha256_file
from .lbf_hybrid import SCHEMAS
from .live_lean import (
    fingerprint, replay_episode, run_episode, public_state,
)
from .live_suite import load_suite
from .native_order import FrozenOrder
from .native_train import (
    _load_teacher, _model_fingerprint, _reference_optimal_actions,
)
from .typed_state import encode_public_packet

VISITED_SCHEMA = "lbf08.model_visited_shortest_set.v1"
COLLECT_SCHEMA = "lbf08.native_train_visited_repair.v1"
FIT_SCHEMA = "lbf08.matched_cpu_fit.v1"
REPLAY_SCHEMA = "lbf08.independent_native_replay.v1"
BEHAVIOR_VIEW = "hybrid_rule_intact"
ARMS = ("source_only", "visited_one", "visited_set")
CHECKS = 8


def _write(path: Path, item: dict) -> None:
    path.write_text(canonical(item) + "\n", encoding="utf-8")


def _rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def _load_manifest(directory: Path, required_schema: str) -> dict:
    path = directory / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != required_schema:
        raise ValueError("LBF-08 source manifest schema mismatch")
    for filename, expected in data.get("artifacts", {}).items():
        if filename != Path(filename).name or sha256_file(directory / filename) != expected:
            raise ValueError("LBF-08 artifact hash mismatch: " + filename)
    if not data.get("artifacts"):
        raise ValueError("LBF-08 source lacks immutable artifact hashes")
    return data


def _oracle_labels(first: dict, menu: list[str], oracle: dict) -> tuple[int, ...] | None:
    """Shortest native continuation among the *public* candidates.

    None means this visited first goal has no shortest-witness option in the
    model-visible action menu; all-zero targets are retained as censored.
    """
    if not menu or len(menu) > 64:
        raise ValueError("invalid public candidate count")
    if not isinstance(first, dict) or "target" not in first:
        raise ValueError("no public Lean goal target")
    target = goal_key(first)
    size = oracle["size"].get(target)
    if size is None or not isinstance(size, int) or size < 1:
        return None
    options = oracle["table"].get(target)
    if options is None:
        return None
    try:
        labels = tuple(_reference_optimal_actions(target, tuple(menu),
                                                  oracle["table"], oracle["size"]))
    except ValueError as exc:
        if "loses all native shortest continuations" in str(exc):
            return None
        if "public menu loses all" in str(exc):
            return None
        raise
    if not any(labels):
        return None
    return labels


def visited_frame(event: dict, ep: dict, oracle: dict) -> tuple[dict | None, str]:
    """Turn one genuine stepper-visited packet into an oracle-backed label.

    Only *after the first model decision* is considered a repair/continuation
    training state. The event already contains Lean's public input packet.
    """
    if not isinstance(event, dict) or type(event.get("attempt")) is not int:
        raise ValueError("invalid native episode event")
    if event["attempt"] <= 1:
        return None, "initial_root_not_repair"
    packet = event.get("input")
    menu = event.get("candidate_menu")
    if not isinstance(packet, dict) or not isinstance(menu, list):
        raise ValueError("native episode missing public input/menu")
    if set(packet) != {"state", "failed_choices", "checks_remaining"}:
        raise ValueError("native visited packet contains extra or missing fields")
    if not isinstance(packet["state"], dict) or not packet["state"].get("goals"):
        raise ValueError("native visited state has no public goal")
    if event.get("selected_index") not in range(len(menu)) or menu[event["selected_index"]] != event.get("tactic"):
        raise ValueError("native recorded selected action differs from its menu")
    labels = _oracle_labels(packet["state"]["goals"][0], menu, oracle)
    if labels is None:
        return None, "no_public_shortest_witness"
    if not 1 <= packet["checks_remaining"] <= CHECKS:
        return None, "out_of_budget"
    if event.get("status") not in ("rejected", "accepted_nonterminal", "kernel_closed"):
        raise ValueError("unrecognized native tactic status")
    first_positive = labels.index(1)
    frame = {
        "schema": VISITED_SCHEMA,
        "root": ep["root"], "lineage": ep["lineage"],
        "split": "train",
        "step": event["attempt"] - 1, "attempt": event["attempt"],
        "goal": canonical(packet),
        "candidates": list(menu),
        "labels": list(labels),
        "single_labels": [int(i == first_positive) for i in range(len(labels))],
        "checks_remaining": packet["checks_remaining"],
        "selected_tactic": event["tactic"],
        "selected_status": event["status"],
        "failed_choices": list(packet["failed_choices"]),
        "n_optimal": sum(labels),
        "source_public_packet_sha256": hashlib.sha256(canonical(packet).encode()).hexdigest(),
        "source_menu_sha256": hashlib.sha256(canonical(menu).encode()).hexdigest(),
        "label_semantics": "native_Lean_shortest_AND_OR_action_set_on_model_visited_state",
        "evidence": "native_replay_and_independently_compiled_root_oracle",
    }
    return frame, "witnessed"


def shortest_finite_script(table: dict, sizes: dict, root: str) -> list[str]:
    """Recursively choose only finite-cost exact native shortest continuations.

    The general lcgrt extractor uses a numeric INF internally, then serializes
    unreachable proof costs as None. A model-visited state may contain
    alternative branches with None children; filter them before argmin.
    """
    minimum = sizes.get(root)
    if not isinstance(minimum, int) or minimum < 1:
        raise ValueError("native visited goal lacks a finite shortest proof")
    options = []
    for tactic, children in table.get(root, {}).items():
        costs = [sizes.get(g) for g in children]
        if any(not isinstance(n, int) or n < 1 for n in costs):
            continue
        if 1 + sum(costs) == minimum:
            options.append((tactic, children))
    if not options:
        raise ValueError("native exact shortest continuation certificate is inconsistent")
    tactic, children = min(options, key=lambda row: row[0])
    script = [tactic]
    for child in children:
        script += shortest_finite_script(table, sizes, child)
    return script


def forced_recovery(repl, case, oracle: dict, accepted_script: list[str]) -> dict:
    """Kernel-check a source-oracle continuation from the model's accepted prefix.

    This is teacher assistance and requires subsequent standalone compilation.
    It never changes or takes credit for an autonomous behavior episode.
    """
    repl.reset()
    state = repl.init(case.statement, pp=True)
    for tactic in accepted_script:
        response = repl.run(state["sid"], tactic)
        if response.get("ok") is not True:
            raise ValueError("behavior accepted prefix disagrees with native Lean")
        state = response["state"]
    before_goals = [goal_key(g) for g in state.get("goals", [])]
    suffix = []
    for goal in before_goals:
        if oracle["size"].get(goal) is None:
            return {"status": "unreachable_visited_state",
                    "prefix_tactics": list(accepted_script),
                    "remaining_goals": before_goals}
        suffix.extend(shortest_finite_script(oracle["table"], oracle["size"], goal))
    for tactic in suffix:
        response = repl.run(state["sid"], tactic)
        if response.get("ok") is not True:
            raise ValueError("source native shortest-continuation failed on model prefix")
        state = response["state"]
    if state.get("goals"):
        raise ValueError("teacher-forced continuation left unproved subgoals")
    kernel = repl.kernel(state["sid"])
    if kernel.get("ok") is not True or kernel.get("axioms") != []:
        raise ValueError("teacher-forced continuation failed native kernel")
    script = list(accepted_script) + suffix
    return {
        "status": "native_kernel_recovered" if suffix else "native_kernel_already_closed",
        "prefix_tactics": list(accepted_script),
        "continuation_tactics": suffix,
        "full_script": script,
        "full_script_length": len(script),
        "within_eight_accepted_tactics": len(script) <= CHECKS,
        "native_kernel_verified": True,
        "standalone_certificate": None,
    }


def collect(suite: Path, teacher: Path, checkpoint_path: Path, out: Path, *,
            roots: int = 48, checks: int = CHECKS, depth: int = 2,
            max_seconds: float = 900.0, cert_timeout: int = 360) -> dict:
    if out.exists() or roots != 48 or checks != CHECKS or depth != 2:
        raise ValueError("LBF-08 native collection fixes fresh output, 48 roots, 8 checks, depth 2")
    if not 0 < max_seconds <= 1800 or not 1 <= cert_timeout <= 900:
        raise ValueError("bounded collection/certifier budget required")
    if not config.REPL_EXE.is_file() or not config.LEAN_EXE.is_file():
        raise ValueError("LBF-08 requires real Lean and lgym")
    source_rows, source = _load_teacher(teacher)
    cases, suite_source = load_suite(suite, "train", roots)
    if (source["independent_lineages"] != roots
        or source["source"]["suite_manifest_sha256"] != suite_source["suite_manifest_sha256"]
        or {x["lineage"] for x in source_rows} != {c.lineage for c in cases}):
        raise ValueError("native train teacher and suite root identities differ")
    model, checkpoint = checkpoint_load(checkpoint_path)
    if (model.cfg.plan != "b" or model.cfg.width != 32 or model.cfg.max_bytes != 384
        or checkpoint.get("input_view") != BEHAVIOR_VIEW
        or checkpoint.get("input_view_schema") != SCHEMAS[BEHAVIOR_VIEW]
        or checkpoint.get("train_order") != "ab"
        or checkpoint.get("seed") != 107 or checkpoint.get("train_steps") != 360
        or checkpoint.get("source_manifest_sha256") != suite_source["suite_manifest_sha256"]
        or checkpoint.get("native_teacher_sha256") != sha256_file(teacher / "manifest.json")
        or checkpoint.get("source_evidence_class") != "lean_kernel_verified_reference_path"):
        raise ValueError("behavior checkpoint is not the registered train-only frozen model")
    torch.set_num_threads(1)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    frozen_sha = _model_fingerprint(model)
    out.mkdir(parents=True)
    begun = time.perf_counter()
    frames, censored, episodes, recoveries, certified_sources = [], [], [], [], []
    attempted = Counter()
    with torch.no_grad(), LeanRepl(timeout=60, exe=config.REPL_EXE,
                                    cwd=config.LEAN_DIR, env=config.lean_env()) as repl:
        if not str(repl.version).startswith("4.19.0"):
            raise ValueError("LBF-08 expected native Lean 4.19.0")
        native_backend = fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version)
        for case in cases:
            if time.perf_counter() - begun >= max_seconds:
                raise TimeoutError("LBF-08 native visit collection exceeded wall budget")
            ep = run_episode(
                repl, case, FrozenOrder(model, "ab", view=BEHAVIOR_VIEW),
                policy="frozen", checks=checks, depth=depth,
            )
            ep["max_checks"] = checks
            ep["frozen_behavior_checkpoint_sha256"] = sha256_file(checkpoint_path)
            ep["replay_passed"] = bool(replay_episode(repl, case, ep))
            if not ep["replay_passed"]:
                raise ValueError("model-visited native episode failed independent replay")
            episodes.append(ep)
            oracle = explore(repl, case.theorem())
            if oracle["proof"]["repl_kernel"] != {"ok": True, "axioms": []}:
                raise ValueError("source oracle root lacks real Lean proof")
            for event in ep["events"]:
                attempted[event["status"]] += 1
                frame, status = visited_frame(event, ep, oracle)
                if frame is None:
                    censored.append({"root": case.root, "lineage": case.lineage,
                                     "attempt": event["attempt"], "reason": status,
                                     "public_packet_sha256": event["public_packet_sha256"]})
                else:
                    frames.append(frame)
            # Forced repair is a source-only label with a real complete proof.
            recovery = forced_recovery(repl, case, oracle, ep["accepted_script"])
            recovery["root"] = case.root
            recovery["lineage"] = case.lineage
            recovery["model_autonomous_closed"] = bool(ep["live_kernel"])
            recoveries.append(recovery)
            if recovery.get("native_kernel_verified"):
                name = case.tid + "_lbf08_forced"
                certified_sources.append((name, case.theorem(), recovery["full_script"]))
    if len(episodes) != roots or len(recoveries) != roots:
        raise RuntimeError("partial native source collection prohibited")
    if not certified_sources:
        raise ValueError("no native forced-recovery scripts available")
    # Independent standalone compiler checks entire model prefix + forced suffix.
    receipt = certify(certified_sources, timeout=cert_timeout, keep=out / "standalone.lean")
    if receipt.get("timed_out"):
        raise ValueError("standalone repair proof compiler timed out")
    by_root = {r["root"]: r for r in recoveries}
    for name, _theorem, _script in certified_sources:
        case = next(c for c in cases if c.tid + "_lbf08_forced" == name)
        verified = receipt.get("status", {}).get(decl_name(name)) == "passed"
        if not verified:
            raise ValueError("source-forced proof failed independent standalone compilation")
        by_root[case.root]["standalone_certificate"] = "passed"
    roots_with_visits = len({r["lineage"] for r in frames})
    if roots_with_visits < 12:
        raise ValueError("INSUFFICIENT_REPAIR_SOURCE: <12 independent roots with witnessed visit continuations")
    if _model_fingerprint(model) != frozen_sha:
        raise ValueError("behavior weights changed during native collection")
    artifact_rows = {
        "visited_frames.jsonl": frames,
        "censored_visits.jsonl": censored,
        "behavior_episodes.jsonl": episodes,
        "forced_recoveries.jsonl": recoveries,
    }
    for filename, rows in artifact_rows.items():
        with (out / filename).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(canonical(row) + "\n")
    _write(out / "standalone_receipt.json", receipt)
    artifacts = {f: sha256_file(out / f) for f in
                 [*artifact_rows, "standalone_receipt.json", "standalone.lean"]}
    report = {
        "schema": COLLECT_SCHEMA, "source_split": "train",
        "native_backend": native_backend,
        "suite_manifest_sha256": suite_source["suite_manifest_sha256"],
        "native_teacher_manifest_sha256": sha256_file(teacher / "manifest.json"),
        "behavior_checkpoint_sha256": sha256_file(checkpoint_path),
        "behavior_frozen_weights_sha256": frozen_sha,
        "behavior_evidence": "source_native_360_step_PPT_B",
        "behavior_root_episodes": len(episodes),
        "model_visitation_frames": len(frames),
        "visited_witnessed_lineages": roots_with_visits,
        "native_attempt_status_counts": dict(attempted),
        "censored_visits": len(censored),
        "multiple_shortest_action_frames": sum(f["n_optimal"] > 1 for f in frames),
        "model_autonomous_kernel_closures": sum(bool(ep["live_kernel"]) for ep in episodes),
        "teacher_forced_kernel_recoveries": len(certified_sources),
        "teacher_forced_standalone_certificates": sum(r.get("standalone_certificate") == "passed" for r in recoveries),
        "forced_recoveries_within_eight_accepted_tactics": sum(
            r.get("within_eight_accepted_tactics") is True for r in recoveries),
        "oracle_assisted_not_model_success": True,
        "source_reference_compiler_passed": True,
        "query_roots_exposed_to_collection": 0,
        "locked_test_opened": False,
        "scientific_eligible": False,
        "elapsed_seconds": round(time.perf_counter()-begun,3),
        "artifacts": artifacts,
    }
    _write(out / "manifest.json", report)
    return report


def replay(suite: Path, teacher: Path, checkpoint_path: Path, source_dir: Path,
           out: Path) -> dict:
    if out.exists():
        raise ValueError("LBF-08 replay output must be fresh")
    meta = _load_manifest(source_dir, COLLECT_SCHEMA)
    train, suite_meta = load_suite(suite, "train", 48)
    original, teacher_meta = _load_teacher(teacher)
    if (meta["native_teacher_manifest_sha256"] != sha256_file(teacher / "manifest.json")
        or meta["suite_manifest_sha256"] != suite_meta["suite_manifest_sha256"]
        or meta["behavior_checkpoint_sha256"] != sha256_file(checkpoint_path)
        or {x["lineage"] for x in original} != {c.lineage for c in train}):
        raise ValueError("LBF-08 replay references changed sources")
    by_root={c.root:c for c in train}
    episodes=_rows(source_dir/"behavior_episodes.jsonl")
    forced=_rows(source_dir/"forced_recoveries.jsonl")
    if len(episodes)!=48 or len(forced)!=48:
        raise ValueError("incomplete behavior/recovery episode roster")
    expected = meta["native_backend"]
    count=0
    recovery_checks=0
    with LeanRepl(timeout=60,exe=config.REPL_EXE,
                  cwd=config.LEAN_DIR,env=config.lean_env()) as repl:
        if fingerprint(config.REPL_EXE,config.LEAN_EXE,repl.version) != expected:
            raise ValueError("native Lean executable/version fingerprint changed")
        for ep in episodes:
            case=by_root.get(ep["root"])
            if not case or case.lineage != ep["lineage"]:
                raise ValueError("native source theorem root/lineage changed")
            if not replay_episode(repl,case,ep):
                raise ValueError("native behavior episode replay failed")
            count+=1
        for rec in forced:
            if not rec.get("native_kernel_verified"):
                continue
            case=by_root.get(rec["root"])
            if not case or case.lineage!=rec["lineage"]:
                raise ValueError("native recovery theorem lineage changed")
            repl.reset()
            state=repl.init(case.statement,pp=True)
            for tactic in rec["full_script"]:
                result=repl.run(state["sid"],tactic)
                if result.get("ok") is not True:
                    raise ValueError("independent replay of oracle-assisted script rejected")
                state=result["state"]
            if state.get("goals"):
                raise ValueError("oracle-assisted script left unproved goals")
            kern=repl.kernel(state["sid"])
            if kern.get("ok") is not True or kern.get("axioms") != []:
                raise ValueError("independent recovery kernel failed")
            recovery_checks+=1
    if count !=48 or recovery_checks!=meta["teacher_forced_standalone_certificates"]:
        raise ValueError("incomplete independent model-free repair replay")
    out.mkdir(parents=True)
    result={
        "schema": REPLAY_SCHEMA, "native_replay_passed": True,
        "suite_manifest_sha256": meta["suite_manifest_sha256"],
        "collection_manifest_sha256": sha256_file(source_dir/"manifest.json"),
        "behavior_checkpoint_sha256": meta["behavior_checkpoint_sha256"],
        "behavior_episodes_replayed": count,
        "teacher_forced_full_proofs_replayed": recovery_checks,
        "query_roots_exposed_to_replay": 0,
        "locked_test_opened": False, "scientific_eligible": False,
    }
    _write(out/"replay.json",result)
    return result


def _load_visited(directory: Path, teacher: Path, suite: Path,
                  checkpoint_path: Path, independent_replay: Path) -> tuple[list[dict],dict]:
    meta=_load_manifest(directory,COLLECT_SCHEMA)
    r=json.loads(independent_replay.read_text(encoding="utf-8"))
    if (r.get("schema")!=REPLAY_SCHEMA or r.get("native_replay_passed") is not True
        or r.get("collection_manifest_sha256")!=sha256_file(directory/"manifest.json")
        or r.get("behavior_checkpoint_sha256")!=sha256_file(checkpoint_path)
        or r.get("behavior_episodes_replayed")!=48
        or r.get("teacher_forced_full_proofs_replayed")!=meta.get("teacher_forced_standalone_certificates")
        or r.get("query_roots_exposed_to_replay")!=0
        or meta["native_teacher_manifest_sha256"] != sha256_file(teacher/"manifest.json")
        or meta["suite_manifest_sha256"] != sha256_file(suite/"manifest.json")
        or meta["behavior_checkpoint_sha256"] != sha256_file(checkpoint_path)
        or meta["native_backend"]["actual_lean_process"] is not True
        or meta["query_roots_exposed_to_collection"]!=0
        or meta["visited_witnessed_lineages"]<12):
        raise ValueError("repair corpus lacks independent source replay and provenance")
    frames=_rows(directory/"visited_frames.jsonl")
    if len(frames)!=meta["model_visitation_frames"]:
        raise ValueError("LBF-08 model-visitation frame census differs")
    train,_src=_load_teacher(teacher)
    lineages={f["lineage"] for f in train}
    for frame in frames:
        if (frame.get("schema")!=VISITED_SCHEMA or frame.get("split")!="train"
            or frame.get("lineage") not in lineages
            or not 1<=frame.get("checks_remaining",0)<=8
            or len(frame.get("candidates",[])) != len(frame.get("labels",[]))
            or len(frame.get("candidates",[])) != len(frame.get("single_labels",[]))
            or sum(frame["single_labels"])!=1 or sum(frame["labels"])<1
            or any(type(x) is not int or x not in (0,1) for x in frame["labels"]+frame["single_labels"])
            or frame["source_public_packet_sha256"]!=hashlib.sha256(frame["goal"].encode()).hexdigest()):
            raise ValueError("invalid visited native shortest-action frame")
    return frames,meta


def frame_schedule(source_frames: list[dict], visited_frames: list[dict],
                   seed:int, steps:int=720) -> list[dict]:
    """Freeze matching source-root/frame draws independent of the chosen arm."""
    if not steps in range(1,5001):
        raise ValueError("invalid update count")
    by_source=defaultdict(list)
    by_visit=defaultdict(list)
    for f in source_frames:
        by_source[f["lineage"]].append(f)
    for f in visited_frames:
        by_visit[f["lineage"]].append(f)
    rng=random.Random(seed)
    names=sorted(by_source)
    if len(names)!=48 or not any(by_visit.values()):
        raise ValueError("LBF-08 requires 48 native train lineages and witnessed model visits")
    roster=[]
    for step in range(steps):
        lineage=names[rng.randrange(len(names))]
        source_idx=rng.randrange(len(by_source[lineage]))
        visit_idx=(rng.randrange(len(by_visit[lineage]))
                   if by_visit.get(lineage) else None)
        roster.append({
            "step":step+1,"lineage":lineage,
            "source_index":source_idx,"visit_index":visit_idx,
            "use_model_visited": bool(step%2==0 and visit_idx is not None),
        })
    return roster


def fit(teacher:Path, visited:Path, suite:Path, behavior_checkpoint:Path,
        replay_receipt:Path, out:Path, *, arm:str, seed:int,
        steps:int=720, threads:int=1, max_seconds:float=600.0) -> dict:
    if (arm not in ARMS or seed not in (107,109) or steps!=720 or threads!=1
        or out.exists() or not 0<max_seconds<=1200):
        raise ValueError("LBF-08 training must use registered arm/seed/steps/CPU and fresh output")
    source_frames,source=_load_teacher(teacher)
    visited_frames,repair_meta=_load_visited(visited,teacher,suite,behavior_checkpoint,replay_receipt)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    cfg=Config(plan="b",width=32,heads=4,max_bytes=384).validate()
    model=PlasticProofTransformer(cfg).cpu().float()
    initial=_model_fingerprint(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    src=defaultdict(list)
    vis=defaultdict(list)
    for f in source_frames:
        src[f["lineage"]].append(f)
    for f in visited_frames:
        vis[f["lineage"]].append(f)
    roster=frame_schedule(source_frames,visited_frames,seed,steps)
    roster_sha=hashlib.sha256("".join(canonical(row)+"\n" for row in roster).encode()).hexdigest()
    out.mkdir(parents=True)
    started=time.perf_counter()
    draws=[]
    for row in roster:
        if time.perf_counter()-started>max_seconds:
            raise TimeoutError("LBF-08 fixed 720-step source training exceeded CPU budget")
        lineage=row["lineage"]
        use_visit=(arm!="source_only" and row["use_model_visited"])
        if use_visit:
            example=vis[lineage][row["visit_index"]]
            labels=example["single_labels"] if arm=="visited_one" else example["labels"]
        else:
            example=src[lineage][row["source_index"]]
            labels=example["labels"]
        goal=encode_public_packet(example["goal"],example["candidates"],
                                  view=BEHAVIOR_VIEW,max_bytes=cfg.max_bytes)
        outputs=model(goal,example["candidates"],depth=2,
                      checks_remaining=example["checks_remaining"],
                      return_all=True,middle_order="ab")
        loss=supervised_loss(outputs,tuple(labels))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite native frame imitation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("nonfinite model gradient")
        optimizer.step()
        draws.append({
            "step":row["step"],"lineage":lineage,
            "chosen_source":"model_visited" if use_visit else "native_reference",
            "source_frame":row["source_index"],"visit_frame":row["visit_index"],
            "optimal_action_count":sum(labels),
            "loss":float(loss.detach()),
        })
    model.eval()
    target=out/"checkpoint.pt"
    teacher_sha=sha256_file(teacher/"manifest.json")
    repair_sha=sha256_file(visited/"manifest.json")
    torch.save({
        "schema":"ppt_gym.checkpoint.v1","config":asdict(cfg),
        "weights":model.state_dict(),
        "source_manifest_sha256":source["source"]["suite_manifest_sha256"],
        "source_evidence_class":source["evidence_class"],
        "native_teacher_sha256":teacher_sha,
        "lbf08_repair_source_manifest_sha256":repair_sha,
        "lbf08_independent_replay_sha256":sha256_file(replay_receipt),
        "lbf08_training_arm":arm,
        "train_steps":steps,"seed":seed,"train_order":"ab",
        "input_view":BEHAVIOR_VIEW,"input_view_schema":SCHEMAS[BEHAVIOR_VIEW],
    },target)
    with (out/"optimizer_steps.jsonl").open("x",encoding="utf-8") as f:
        for row in draws:
            f.write(canonical(row)+"\n")
    visible=sum(x["chosen_source"]=="model_visited" for x in draws)
    result={
        "schema":FIT_SCHEMA,"seed":seed,"arm":arm,"train_order":"ab",
        "input_view":BEHAVIOR_VIEW,"native_source":True,
        "train_source_manifest_sha256":source["source"]["suite_manifest_sha256"],
        "native_teacher_manifest_sha256":teacher_sha,
        "repair_source_manifest_sha256":repair_sha,
        "independent_replay_sha256":sha256_file(replay_receipt),
        "source_roots":len(src),"visited_roots":len(vis),
        "registered_reference_visitor_draw_schedule_sha256":roster_sha,
        "updates":len(draws),"requested_steps":steps,
        "model_visited_updates":visible,
        "reference_path_updates":steps-visible,
        "one_positive_visited_updates":sum(
            d["chosen_source"]=="model_visited" and d["optimal_action_count"]==1 for d in draws),
        "multi_positive_visited_updates":sum(
            d["chosen_source"]=="model_visited" and d["optimal_action_count"]>1 for d in draws),
        "model":model.architecture(),
        "parameter_count":model.parameter_count(),
        "initial_weights_sha256":initial,
        "final_weights_sha256":_model_fingerprint(model),
        "checkpoint_sha256":sha256_file(target),
        "optimizer_steps_sha256":sha256_file(out/"optimizer_steps.jsonl"),
        "mean_last_32_loss":sum(d["loss"] for d in draws[-32:])/32,
        "training_clock_seconds":round(time.perf_counter()-started,3),
        "calibration_updates":0,"locked_test_opened":False,"scientific_eligible":False,
    }
    _write(out/"manifest.json",result)
    return result


def main(argv:list[str]|None=None)->None:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    cp=sub.add_parser("collect")
    cp.add_argument("--suite",type=Path,required=True)
    cp.add_argument("--teacher",type=Path,required=True)
    cp.add_argument("--checkpoint",type=Path,required=True)
    cp.add_argument("--out",type=Path,required=True)
    cp.add_argument("--roots",type=int,default=48)
    cp.add_argument("--max-seconds",type=float,default=900)
    cp.add_argument("--cert-timeout",type=int,default=360)
    rp=sub.add_parser("replay")
    rp.add_argument("--suite",type=Path,required=True)
    rp.add_argument("--teacher",type=Path,required=True)
    rp.add_argument("--checkpoint",type=Path,required=True)
    rp.add_argument("--collection",type=Path,required=True)
    rp.add_argument("--out",type=Path,required=True)
    fp=sub.add_parser("fit")
    fp.add_argument("--suite",type=Path,required=True)
    fp.add_argument("--teacher",type=Path,required=True)
    fp.add_argument("--collection",type=Path,required=True)
    fp.add_argument("--checkpoint",type=Path,required=True)
    fp.add_argument("--replay-receipt",type=Path,required=True)
    fp.add_argument("--out",type=Path,required=True)
    fp.add_argument("--arm",choices=ARMS,required=True)
    fp.add_argument("--seed",type=int,required=True)
    fp.add_argument("--max-seconds",type=float,default=600)
    args=parser.parse_args(argv)
    if args.command=="collect":
        result=collect(args.suite,args.teacher,args.checkpoint,args.out,
                       roots=args.roots,max_seconds=args.max_seconds,
                       cert_timeout=args.cert_timeout)
    elif args.command=="replay":
        result=replay(args.suite,args.teacher,args.checkpoint,args.collection,args.out)
    else:
        result=fit(args.teacher,args.collection,args.suite,args.checkpoint,
                   args.replay_receipt,args.out,arm=args.arm,seed=args.seed,
                   max_seconds=args.max_seconds)
    print(canonical(result))


if __name__=="__main__":
    main()
