"""Audit residual fact transport and paired credit against matched controls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from configs.analyze_ppt_graph_alias_pilot import fact_switch, pair_counts
from configs.analyze_ppt_hybrid_replication import PROTOCOL_FIELDS, read_arm
from ppt_gym import ram_store
from ppt_gym.data import digest, sha256_file
from ppt_gym.hybrid_rl.corpus import PAIRED_FAMILIES, load, proposal_attempt
from ppt_gym.hybrid_rl.__main__ import proposal_event

ARMS = {"transport": ("hybrid", False, True, "standard"),
        "transport-head": ("hybrid", True, True, "standard"),
        "trm": ("trm", False, False, "standard"),
        "head-only": ("trm", True, False, "standard")}
PAIRED_ARMS = {"contrast": ("hybrid", False, True, "contrast"),
               "ordinary": ("hybrid", False, True, "paired"),
               "trm-contrast": ("trm", False, False, "contrast"),
               "head-contrast": ("hybrid", True, True, "contrast")}


def summarize_fact_switch(report):
    return {key: value for key, value in report.items() if key != "pairs"}


def audit_proposals(path, train_cases, rollouts, *, require_both=False):
    proposals = {"audited": 0, "censored": 0, "selected_first": 0}
    schedule = []
    with ram_store.open_store(path) as db:
        for row in ram_store.ledger(db):
            if row["phase"] != "train":
                continue
            episodes = row["episodes"]
            alias = [ep for ep in episodes if train_cases[ep["root"]].family in PAIRED_FAMILIES]
            if not alias:
                continue
            if len(alias) != len(episodes) or len(alias) != 2 * rollouts:
                raise ValueError("unexpected matched batch size")
            choices = []
            for a, b in zip(alias[::2], alias[1::2], strict=True):
                if (a["root"] == b["root"] or train_cases[a["root"]].cluster != train_cases[b["root"]].cluster
                        or any(ep.get("root_proposal_probability") != 0.5 for ep in (a, b))):
                    raise ValueError("worlds did not use the registered paired proposal")
                if a.get("reward_censored") and b.get("reward_censored"):
                    proposals["censored"] += 1
                    continue
                first, second = proposal_event(a), proposal_event(b)
                if (a.get("reward_censored") or b.get("reward_censored")
                        or first["attempt"] != second["attempt"]
                        or first["attempt"] != proposal_attempt(train_cases[a["root"]].family)
                        or len(first["candidate_menu"]) != 2 or first["candidate_menu"] != second["candidate_menu"]
                        or first["selected_index"] != second["selected_index"]):
                    raise ValueError("worlds received different root actions")
                proposals["audited"] += 1
                proposals["selected_first"] += first["selected_index"] == 0
                choices.append(first["selected_index"])
            if require_both and sorted(choices) != [0, 1]:
                raise ValueError("paired batch must try both root actions in each world")
            schedule.append((alias[0]["root"], choices))
    if not proposals["audited"]:
        raise ValueError("missing paired root proposals")
    if require_both:
        proposals["schedule_sha256"] = digest(schedule)
    return proposals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("sessions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--paired", action="store_true")
    args = parser.parse_args()
    registration = json.loads(args.cohort.read_bytes())["registration"]
    corpus = load(args.cohort)
    cases = {case.root: case for case in corpus["dev"]}
    train_cases = {case.root: case for case in corpus["train"]}
    clusters = defaultdict(list)
    for root, case in cases.items():
        clusters[case.cluster].append(root)
    if len(clusters) != registration["dev_alias_clusters"] or any(len(pair) != 2 for pair in clusters.values()):
        raise ValueError("incomplete paired-world cohort")
    seed, = registration["model_seeds"]
    cohort_sha = sha256_file(args.cohort)
    arms = PAIRED_ARMS if args.paired else ARMS
    trials, proposals, protocol = {}, {}, None
    for arm, (architecture, head, residual, alias_training) in arms.items():
        path = args.sessions / f"{seed}-{arm}.sqlite"
        trial = read_arm(path, cases, cohort_sha)
        m = trial["manifest"]
        identity = (m["architecture"], m["head_only"], m["residual_fact_transport"], m["seed"],
                    m["graph_context"], m.get("alias_training", "standard"))
        if identity != (architecture, head, residual, seed, True, alias_training):
            raise ValueError("unregistered transport arm")
        identity = tuple(m[field] for field in PROTOCOL_FIELDS)
        if protocol is None:
            protocol = identity
        if protocol != identity:
            raise ValueError("transport arms did not share the training and proof budgets")
        trials[arm] = trial
        if args.paired:
            proposals[arm] = audit_proposals(path, train_cases, m["rollouts"])
    if len({trial["training_exposure_sha256"] for trial in trials.values()}) != 1:
        raise ValueError("transport arms did not use matching training roots")
    initial_matches = (("contrast", "ordinary"), ("contrast", "head-contrast")) if args.paired else (
        ("transport", "transport-head"), ("trm", "head-only"))
    for reference, match in initial_matches:
        initial = trials[reference]["stages"]["initial"]
        other = trials[match]["stages"]["initial"]
        if any(initial[root]["tactics"] != other[root]["tactics"] for root in cases):
            raise ValueError("full and frozen arms began with different proof policies")

    from ppt_gym.ram_campaign import configure

    configure(2, seed)
    margins = {
        arm: fact_switch(args.sessions / f"{seed}-{arm}.sqlite", clusters, architecture,
                         trials[arm]["manifest"]["width"])
        for arm, (architecture, _, _, _) in arms.items()
    }
    counts = {arm: {phase: pair_counts(stage, clusters) for phase, stage in trial["stages"].items()}
              for arm, trial in trials.items()}
    report = {"scope": "reused development pairs, one model seed", "cohort_sha256": cohort_sha,
              "registered_pairs": len(clusters), "worlds": len(cases), "counts": counts,
              "fact_switch": margins, "trials": trials, "matched_proposals": proposals}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for arm in arms:
        print(arm, counts[arm]["initial"], counts[arm]["final"], summarize_fact_switch(margins[arm]))


if __name__ == "__main__":
    main()
