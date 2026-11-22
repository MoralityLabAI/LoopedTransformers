"""Recorded original MathCan training effects with exact-prefix transitions."""
from __future__ import annotations

import json
import zipfile
from collections import defaultdict

from ..data import canonical, digest, sha256_file
from .corpus import REPO

ARCHIVE_SHA = "a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc"


def public(state):
    return {"goals": state.get("goals"), "definitions": state.get("definitions", [])}


def state_key(row):
    return row["source_file"], row["theorem"], tuple(row["prefix"])


class MathCanReplay:
    """A finite recorded environment. Unsupported continuations are censored."""

    def __init__(self, archive):
        if sha256_file(archive) != ARCHIVE_SHA:
            raise ValueError("expected the immutable RAM-08 original MathCan training archive")
        self.states = defaultdict(dict)
        protected = set()
        for experiment in ("lean_ram10", "lean_ram16", "lean_ram17", "lean_ram18"):
            config = json.loads((REPO / "experiments" / experiment / "config.json").read_text(encoding="utf-8"))
            modules = config.get("source_modules", config.get("modules", {})).get("test", {})
            protected.update(modules)
            protected.update(config.get("excluded_pef_modules", ()))
            protected.update(config.get("protected_test_modules_older", ()))
        with zipfile.ZipFile(archive) as z:
            rows = z.read("data/transitions.jsonl").splitlines()
            audit = json.loads(z.read("data/original_mathcan_independent_audit.json"))
            if audit["status"] != "passed" or audit["checked"] != len(rows) or audit["failures"]:
                raise ValueError("original-source Lean audit is incomplete")
        for raw in rows:
            row = json.loads(raw)
            module = row["source_file"].rsplit("/", 1)[-1].removesuffix(".lean")
            if row["development"] is not False or module in protected:
                continue
            if not row["source_file"].startswith("MathCan/") or not row["before"]["accepted"]:
                raise ValueError("invalid original training source entry")
            key = state_key(row)
            previous = self.states[key].get(row["script"])
            if previous is not None and (public(previous["after"]) != public(row["after"])
                                          or previous["after"]["accepted"] != row["after"]["accepted"]):
                raise ValueError("conflicting audited verdict for one script")
            self.states[key][row["script"]] = row
        self.observations = {}
        for key, menu in self.states.items():
            states = {canonical(public(row["before"])) for row in menu.values()}
            if len(states) != 1:
                raise ValueError("one original prefix has inconsistent public observations")
            self.observations[key] = json.loads(states.pop())
        self.starts = sorted(key for key in self.states if not key[2] and len(self.states[key]) > 1)
        if not self.starts:
            raise ValueError("zero auditable original MathCan TRAIN theorem starts")

    def step(self, key, action):
        row = self.states[key][action]
        observed = row["after"]
        if not observed["accepted"]:
            return "rejected", key
        if observed["goals"] == []:
            return "archived_closure", None
        successor = (*key[:2], (*key[2], action))
        if self.observations.get(successor) == public(observed):
            return "accepted_nonterminal", successor
        return "censored", None

    def rollout(self, key, policy, rng, args):
        import torch

        start = key
        attempts, macs = [], 0
        passed = False
        status = "budget_exhausted"
        excluded = set()
        passes = 4 if policy.architecture == "hybrid" else 1
        with torch.no_grad():
            for attempt in range(1, 9):
                menu = tuple(script for script in sorted(self.states[key]) if (key, script) not in excluded)[:64]
                if not menu:
                    status = "no_candidates"
                    break
                cost = policy.macs(len(menu), passes)
                if macs + cost > args.max_macs:
                    status = "compute_budget_exhausted"
                    break
                macs += cost
                state = self.observations[key]
                packet = {"state": {"goals": [{"target": {"pp": goal}, "hypotheses": []}
                                              for goal in state["goals"]]},
                          "failed_choices": sorted(script for parent, script in excluded if parent == key),
                          "checks_remaining": 9 - attempt}
                logits, _ = policy(canonical(packet), menu, passes)
                selected = int(torch.multinomial(logits.softmax(-1), 1, generator=rng))
                action = menu[selected]
                status, after = self.step(key, action)
                row = self.states[key][action]
                attempts.append({"kind": "tactic", "attempt": attempt, "input": packet,
                                 "candidate_menu": list(menu), "selected_index": selected, "tactic": action,
                                 "status": status, "original_state_sha256": row["state_sha256"],
                                 "original_source_sha256": row["source_sha256"], "original_prefix": list(key[2])})
                if status == "rejected":
                    excluded.add((key, action))
                elif status == "accepted_nonterminal":
                    key = after
                else:
                    passed = status == "archived_closure"
                    break
        return {"source": "mathcan_ram08_audited_train", "root": digest(start), "family": "mathcan",
                "original_root": list(start[:2]), "events": attempts, "status": status, "depth": passes,
                "verified_closed": False, "archived_closure": passed, "within_cpu_budget": True,
                "reward_censored": status == "censored", "tactic_checks": len(attempts), "model_macs": macs}
