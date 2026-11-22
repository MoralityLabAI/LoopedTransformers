"""Build separate native successor training and held-out structural proof roots."""

import argparse
from collections import Counter
import json
from pathlib import Path

from configs.collect_ppt_native_feedback import collect_graph
from configs.freeze_ppt_native_progress import (
    add_sources, advance, candidate_menu, declaration_type, native_sources, observe, shortest, support_paths,
)
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt import fragment
from lcgrt.config import LEAN_VERSION
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest, sha256_file
from ppt_gym.hybrid_rl.corpus import REPO
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.native-successor-curriculum.v1"
FAMILIES = ("chain-deep", "tree-dead", "alt-deep")
TRAIN = 24
HELD = 30


def proof_script(theorem, sizes):
    def prove(goal):
        if goal in theorem.facts:
            return [f"apply h{theorem.facts.index(goal)}"]
        index = next(i for i, (premises, head) in enumerate(theorem.rules)
                     if head == goal and 1 + sum(sizes[p] for p in premises) == sizes[goal])
        return [f"apply r{index}", *(action for goal in theorem.rules[index][0] for action in prove(goal))]

    prefix = ["constructor"] if len(theorem.goal) > 1 else []
    return prefix + [action for goal in theorem.goal for action in prove(goal)]


def horn_cases(previous):
    cohort = REPO / "research/lean_cgrt_hrm400/results/ppt_hybrid_rl/development-cohort.json"
    seen = {row["lineage"] for row in json.loads(cohort.read_bytes())["cases"]}
    reserved = {" ".join(row["statement"].split()) for row in previous["cases"] + previous["unresolved"]}
    for split, count in (("train", TRAIN), ("held", HELD)):
        chosen = []
        for step in range(4096):
            family = FAMILIES[len(chosen) % len(FAMILIES)]
            seed = int(digest([SCHEMA, split, step])[:15], 16)
            theorem = fragment.generate(seed, 2 + step % 3, family, mirror_p=.8, conj_p=.35)
            lineage = fragment.canonical_hash(theorem)
            statement = fragment.statement(theorem)
            sizes = fragment.reference_sizes(theorem)
            length = sum(sizes[goal] for goal in theorem.goal) + int(len(theorem.goal) > 1)
            producers = fragment.reference_legal(theorem, theorem.goal[0])
            if lineage in seen or " ".join(statement.split()) in reserved or not 3 <= length <= 7 or len(producers) < 2:
                continue
            seen.add(lineage)
            tid = f"successor_{split}_{len(chosen)}"
            chosen.append({"module": lineage, "source": "LoopedTransformers Horn", "theorem": tid,
                           "statement": statement, "reference": proof_script(theorem, sizes), "checks": length,
                           "family": family, "seed": seed, "cohort": split})
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError("not enough disjoint multi-step roots")
        yield chosen


def remaining_evalmath(args, prior):
    path = args.evalmath / "fixtures/bpd_lean_curriculum_v2/train.jsonl"
    manifest = json.loads(path.with_name("manifest.json").read_bytes())
    if sha256_file(path) != manifest["sha256"][path.name]:
        raise ValueError("the EvalMath training fixture changed")
    seen = {row["theorem"] for row in prior["cases"] + prior["unresolved"]}
    existing = REPO / "research/lean_cgrt_hrm400/results/ppt_hybrid_rl/root-adaptation-cohort.json"
    seen.update(row["tid"] for row in json.loads(existing.read_bytes())["cases"] if row["source"] == "evalmath_lg01")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != manifest["split_counts"]["train"] or any(row["split"] != "train" for row in rows):
        raise ValueError("a held-out EvalMath file entered the source selection")
    for row in rows:
        name, header = row["public_goal_source"].removeprefix("theorem ").split(" ", 1)
        if name not in seen and row["host_label"] == "positive_kernel_checked":
            yield {"module": "EvalMath:" + row["family"], "source": "EvalMath", "theorem": name,
                   "statement": declaration_type(header, None), "cohort": "held"}


def certify(repl, case):
    repl.reset()
    state = repl.init(case["statement"], pp=True)
    initial = observe(repl, state)
    if "reference" in case:
        visible = initial
        for index, action in enumerate(case["reference"], start=1):
            if action not in candidate_menu(visible):
                raise ValueError("a registered reference action vanished from the public menu")
            status, state, visible = advance(repl, state, visible, action)
            if status != ("closed" if index == case["checks"] else "progress"):
                raise ValueError("Lean disagreed with the exact Horn certificate")
        script = case["reference"]
    else:
        script, _ = shortest(repl, state, initial)
    return {**case, "reference": script, "checks": len(script), "visible_sha256": digest(initial),
            "intro_names": [hyp["name"] for hyp in initial["goals"][0]["hypotheses"]]}


def build(args, tree):
    original = read_receipt(args.registry)
    train, held = horn_cases(original)
    held.extend(remaining_evalmath(args, original))
    data, cases, unresolved = [], [], []
    with native_sources(support_paths(args)) as (directory, env, imports), LeanRepl(imports=imports, env=env, cwd=directory) as repl:
        if repl.version != LEAN_VERSION:
            raise ValueError("unexpected native verifier")
        seen = {row["visible_sha256"] for row in original["cases"]}
        for row in held:
            certified = certify(repl, row)
            if certified["checks"] >= 2 and certified["visible_sha256"] not in seen:
                cases.append(certified)
                seen.add(certified["visible_sha256"])
            else:
                unresolved.append(certified)
        for case in train:
            certified = certify(repl, case)
            graph = collect_graph(repl, certified, seen)
            if graph is None:
                raise ValueError("a fresh training graph shares a scoring state")
            # Complete observed demonstrations before expanding arbitrary alternatives.
            known = {tuple(node["prefix"]) for node in graph["nodes"]}
            repl.reset()
            state = repl.init(case["statement"], pp=True)
            visible = observe(repl, state)
            prefix = ()
            for action in certified["reference"]:
                if prefix not in known:
                    choices = {}
                    for option in candidate_menu(visible):
                        status, _, after = advance(repl, state, visible, option)
                        choices[option] = {"status": status, "after_sha256": digest(after)}
                    graph["nodes"].append({"prefix": prefix, "public": visible, "choices": choices})
                status, state, visible = advance(repl, state, visible, action)
                prefix = (*prefix, action)
            data.append(graph)
    if sum(row["source"] == "LoopedTransformers Horn" for row in cases) != HELD:
        raise ValueError("the independent generated bank changed size")
    print("new training roots", len(data), "prefixes", sum(len(row["nodes"]) for row in data),
          "fresh evaluation", dict(Counter(row["source"] for row in cases)), flush=True)
    write_receipt(args.output, {"schema": SCHEMA, "previous_registry_sha256": sha256_file(args.registry),
                                "generator_sha256": sha256_file(Path(fragment.__file__)),
                                "training": data, "cases": cases, "unresolved": unresolved, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_sources(parser)
    for name in ("registry", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        build(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
