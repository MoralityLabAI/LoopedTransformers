"""Run CPU-only PPT-G04 campaigns against the native Lean proof server."""
from __future__ import annotations

import json
import platform
import random
import time
from pathlib import Path

import torch

from lcgrt import config
from lcgrt.certify import certify, decl_name, render
from lcgrt.lean_repl import LeanRepl

from . import ram_curriculum, ram_learning, ram_store
from .data import canonical, digest, sha256_file
from .live_lean import fingerprint
from .live_search import replay_search, search_episode
from .ram_learning import Online, build_model, credit, fit_encoder, model_macs, read_features, tensor_hash

SCHEMA = "ppt_gym.ram_session.v1"


def configure(threads, seed):
    if torch.version.cuda is not None or getattr(torch.version, "hip", None) is not None:
        raise RuntimeError("PPT-G04 requires a CPU-only PyTorch build")
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.random.default_generator.manual_seed(seed)


def horn_choice(text, menu):
    goal = json.loads(text)["state"]["goals"][0]
    rules = {}
    for hyp in goal["hypotheses"]:
        target, premises = hyp["type"], []
        while "imp" in target:
            premise, target = target["imp"]
            premises.append(premise)
        rules["apply " + hyp["name"]] = (canonical(target), premises)
    infinite = float("inf")
    costs = {}

    def cost(term):
        if "and" in term:
            return 1 + sum(cost(child) for child in term["and"])
        return costs.get(canonical(term), infinite)

    for _ in rules:
        changed = False
        for head, premises in rules.values():
            value = 1 + sum(cost(premise) for premise in premises)
            if value < costs.get(head, infinite):
                costs[head] = value
                changed = True
        if not changed:
            break

    def action_cost(tactic):
        if tactic == "constructor":
            return cost(goal["target"])
        return 1 + sum(cost(premise) for premise in rules[tactic][1])

    return min(range(len(menu)), key=lambda index: (action_cost(menu[index]), menu[index]))


def native_certificate(case, episode, timeout):
    if not episode["live_kernel"]:
        return None
    items = [(case.tid, case.theorem(), episode["accepted_script"])]
    result = certify(items, timeout=timeout)
    source, _, _ = render(items)
    return {"status": result["status"].get(decl_name(case.tid)),
            "source_sha256": result["source_sha256"], "source": source,
            "returncode": result["returncode"], "lean_version": result["lean_version"],
            "output": result["output_tail"]}


def episode_run(repl, budget, model, online, case, *, arm, depth, options, probe=False):
    mode = {"zero_messages": "zero", "rolled_messages": "roll"}.get(arm, "normal")
    depth = min(16, 2 * depth) if arm == "extra_compute" else depth
    costs, macs = [], 0

    def rank(text, menu, remaining):
        nonlocal macs
        if arm == "horn":
            return horn_choice(text, menu), False
        if len(menu) == 1:
            return 0, False
        estimate = model_macs(model, depth, len(menu), mode)
        if options["max_macs"] and macs + estimate > options["max_macs"]:
            return None, False
        macs += estimate
        record = []
        wall, cpu = time.perf_counter_ns(), time.process_time_ns()
        out = read_features(model, text, menu, remaining, depth, mode, record)
        index = int((out["logits"] + 3 * online.score(arm, out["features"])).argmax())
        costs.append({"wall_ns": time.perf_counter_ns() - wall, "cpu_ns": time.process_time_ns() - cpu,
                      "passes": record, "estimated_macs": estimate})
        return index, True

    lean_before, cpu_before = repl.seconds, budget.cpu_seconds()

    def within_budget():
        return budget.cpu_seconds() - cpu_before < options["cpu_seconds"]

    ep = search_episode(repl, case, model, policy=arm, depth=depth, check_budget=8,
                        rollback_budget=0 if arm == "no_rollback" else 8,
                        ranker=None if arm == "auto8" else rank, within_budget=within_budget,
                        probe=probe)
    work_cpu = budget.cpu_seconds() - cpu_before
    ep["search_cpu_seconds"] = work_cpu
    ep["verification_seconds"] = repl.seconds - lean_before
    ep["forward_calls"], ep["estimated_macs"] = costs, macs
    start = time.perf_counter()
    cert = native_certificate(case, ep, options["timeout"])
    ep["certificate_seconds"] = time.perf_counter() - start if cert else 0.0
    ep["standalone_certificate"] = cert["status"] if cert else None
    ep["certificate"] = cert
    ep["verified_closed"] = bool(cert and cert["status"] == "passed")
    ep["within_cpu_budget"] = work_cpu <= options["cpu_seconds"]
    return ep


def evidence_update(online, model, ep, rng):
    observations = credit(ep)
    shuffled = [item[0] for _, item in observations]
    rng.shuffle(shuffled)
    records = []
    for (index, (label, reason)), permuted in zip(observations, shuffled):
        event = ep["events"][index]
        with torch.no_grad():
            out = read_features(model, canonical(event["input"]), event["candidate_menu"],
                                event["input"]["checks_remaining"], ep["depth"])
        update = online.update(out["features"][event["selected_index"]], label, permuted)
        records.append({**update, "event": index, "credit": reason})
    return records


def bind(curriculum, budget, options, repl):
    if not str(repl.version).startswith(config.LEAN_VERSION):
        raise ValueError(f"expected Lean {config.LEAN_VERSION}, found {repl.version}")
    files = (list(Path(__file__).parent.glob("*.py"))
             + [config.LEAN_DIR / "Main.lean", Path(config.__file__)])
    return {"schema": SCHEMA, "curriculum_sha256": sha256_file(curriculum),
            "verifier": fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version),
            "options": options, "threads": budget.threads, "ram_bytes": budget.limit,
            "engine": {path.name: sha256_file(path) for path in sorted(files)},
            "runtime": {"torch": str(torch.__version__), "python": platform.python_version(),
                        "platform": platform.platform(), "machine": platform.machine(),
                        "cpu_only_build": True, "master_dtype": "float32", "gpu_allocations": 0}}


def training_episode(db, root):
    for (raw,) in db.execute("SELECT json FROM ledger WHERE json_extract(json, '$.task.root') = ?",
                             (root,)):
        item = json.loads(raw)
        if item["task"]["kind"] in ("warmup", "support"):
            return item["episode"]
    raise ValueError("missing train-only episode for encoder fit")


def run(session, curriculum, budget, options, steps):
    registered, cohorts = ram_curriculum.load(curriculum)
    cases = {case.root: case for group in cohorts.values() for case in group}
    tasks = ram_curriculum.schedule(registered, cohorts, **{
        name: options[name] for name in ("support", "rounds", "persistent", "meta_steps",
                                         "backbone_every", "arms", "depths", "final_only")})
    model = build_model(options["plan"], options["preset"])
    count = model.parameter_count()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    online = Online(model.cfg.width)
    rng = random.Random(options["seed"])
    verified_support = 0
    with LeanRepl(timeout=options["timeout"]) as repl, ram_store.open_store(session, write=True) as db:
        manifest = {**bind(curriculum, budget, options, repl), "architecture": model.architecture(),
                    "schedule_sha256": digest(tasks), "scheduled_entries": len(tasks)}
        ram_store.initialize(db, manifest)
        with db:
            db.execute("BEGIN IMMEDIATE")
            state = ram_store.checkpoint(db)
            model_sha = tensor_hash(model.state_dict())
            if state is None:
                db.execute("INSERT INTO weights VALUES (?, ?)", (model_sha, ram_store.pack(model.state_dict())))
                db.execute("INSERT INTO checkpoint VALUES (0, ?)", (ram_store.pack({
                    "model_sha": model_sha, "optimizer": optimizer.state_dict(), "online": online.state(),
                    "torch_rng": torch.get_rng_state(), "rng": rng.getstate(), "verified_support": 0,
                }),))
        total_updates = 0
        cursor = 0
        previous_cursor = -2
        while cursor < len(tasks):
            with db:
                db.execute("BEGIN IMMEDIATE")
                cursor = db.execute("SELECT COUNT(*) FROM ledger").fetchone()[0]
                if cursor >= len(tasks) or (steps and total_updates >= steps):
                    break
                state = ram_store.checkpoint(db)
                if cursor == 0 or cursor != previous_cursor + 1:
                    model_sha = state["model_sha"]
                    model.load_state_dict(ram_store.load_backbone(db, model_sha))
                    optimizer.load_state_dict(state["optimizer"])
                    online.restore(state["online"])
                    rng.setstate(state["rng"])
                    torch.set_rng_state(state["torch_rng"])
                    verified_support = state["verified_support"]
                previous_cursor = cursor
                task = tasks[cursor]
                kind = task["kind"]
                row = {"task": task, "model_sha256": model_sha, "adapter_before": tensor_hash(online.weights()),
                       "adapter_version_before": online.version}
                backbone = None
                if kind == "reset":
                    online = Online(model.cfg.width)
                    verified_support = 0
                elif kind == "fit":
                    update = fit_encoder(model, optimizer, training_episode(db, task["support"]),
                                         training_episode(db, task["query"]),
                                         options["depths"][0])
                    model_sha = tensor_hash(model.state_dict())
                    backbone = (model_sha, model.state_dict())
                    row["encoder_fit"] = update
                else:
                    case = cases[task["root"]]
                    arm = task.get("arm", options["collector"] if kind == "support" else "horn")
                    ep = episode_run(repl, budget, model, online, case, arm=arm,
                                     depth=task.get("depth", options["depths"][0]), options=options,
                                     probe=kind == "warmup" or (kind == "support" and task["dose"] % 4 == 0))
                    row["episode"] = ep
                    if kind == "support":
                        started = time.perf_counter()
                        row["updates"] = evidence_update(online, model, ep, rng)
                        row["update_wall_seconds"] = time.perf_counter() - started
                        verified_support += bool(row["updates"])
                row.update(model_after_sha256=model_sha, adapter_after=tensor_hash(online.weights()),
                           adapter_version_after=online.version, verified_support_roots=verified_support)
                changed = kind in ("reset", "fit", "support")
                state = ({"model_sha": model_sha, "optimizer": optimizer.state_dict(), "online": online.state(),
                          "torch_rng": torch.get_rng_state(), "rng": rng.getstate(),
                          "verified_support": verified_support} if changed else None)
                ram_store.append(db, row, state, backbone)
                total_updates += 1
        last = db.execute("SELECT COUNT(*) FROM ledger").fetchone()[0]
    return {"entries": last, "scheduled": len(tasks), "entries_this_invocation": total_updates,
            "parameters": count, "memory": budget.stats()}


def replay(db, curriculum, budget, output):
    manifest = ram_store.read_manifest(db)
    options = manifest["options"]
    registered, cohorts = ram_curriculum.load(curriculum)
    cases = {case.root: case for group in cohorts.values() for case in group}
    tasks = ram_curriculum.schedule(registered, cohorts, **{
        name: options[name] for name in ("support", "rounds", "persistent", "meta_steps",
                                         "backbone_every", "arms", "depths", "final_only")})
    model = build_model(options["plan"], options["preset"])
    online = Online(model.cfg.width)
    rng = random.Random(options["seed"])
    receipts, last_seq = [], -1
    backbone_optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    model_sha = tensor_hash(model.state_dict())
    with LeanRepl(timeout=options["timeout"]) as repl:
        identity = bind(curriculum, budget, options, repl)
        for key in ("curriculum_sha256", "verifier", "threads", "engine", "runtime"):
            if identity[key] != manifest[key]:
                raise ValueError(f"cannot replay changed {key}")
        if digest(tasks) != manifest["schedule_sha256"]:
            raise ValueError("replay schedule changed")
        for row in ram_store.ledger(db):
            task, last_seq = row["task"], row["seq"]
            kind = task["kind"]
            if (task != tasks[last_seq] or row["adapter_before"] != tensor_hash(online.weights())
                    or row["model_sha256"] != model_sha):
                raise ValueError("replay schedule or starting fast weights differ")
            if kind == "reset":
                online = Online(model.cfg.width)
            if kind == "fit":
                fit = fit_encoder(model, backbone_optimizer, training_episode(db, task["support"]),
                                  training_episode(db, task["query"]), options["depths"][0])
                recorded = row["encoder_fit"]
                if (fit is None) != (recorded is None) or (fit and fit["loss"] != recorded["loss"]):
                    raise ValueError("encoder meta-training loss changed")
                model_sha = tensor_hash(model.state_dict())
                if tensor_hash(ram_store.load_backbone(db, model_sha)) != model_sha:
                    raise ValueError("recorded backbone checkpoint bytes changed")
            if "episode" in row:
                case = cases[task["root"]]
                ep = row["episode"]
                result = replay_search(repl, case, ep)
                cert = native_certificate(case, ep, options["timeout"])
                if (cert != ep["certificate"]
                        or bool(cert and cert["status"] == "passed") != ep["verified_closed"]):
                    raise ValueError("independent Lean certificate replay changed")
                receipts.append({"seq": last_seq, "root": case.root, "episode_sha256": digest(ep),
                                 "certificate_sha256": cert["source_sha256"] if cert else None, **result})
                if kind == "support" and evidence_update(online, model, ep, rng) != row["updates"]:
                    raise ValueError("verifier-grounded fast-weight replay changed")
            if tensor_hash(online.weights()) != row["adapter_after"] or model_sha != row["model_after_sha256"]:
                raise ValueError("recorded post-update fast weights differ")
    state = ram_store.checkpoint(db)
    latest = Online(model.cfg.width)
    latest.restore(state["online"])
    if (tensor_hash(latest.weights()) != tensor_hash(online.weights()) or state["model_sha"] != model_sha
            or tensor_hash(ram_store.load_backbone(db, model_sha)) != model_sha):
        raise ValueError("final checkpoint fast weights differ from replay")
    head = db.execute("SELECT hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
    result = {"ledger_head": head[0] if head else None, "entries": last_seq + 1,
              "verified_episodes": receipts, "fast_weights_replayed": True,
              "verifier": manifest["verifier"], "curriculum_sha256": manifest["curriculum_sha256"],
              "memory": budget.stats()}
    output.write_text(canonical(result) + "\n", encoding="utf-8")
    return result
