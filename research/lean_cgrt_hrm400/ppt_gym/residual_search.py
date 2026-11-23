"""A paid sibling trial and a bounded native proof continuation."""

from dataclasses import dataclass, replace


PROBE = "sibling"
TARGET = "target"
STOP = ("stop", "")
FEES = {PROBE: .05, TARGET: .025}
PROBES = ("assumption", "rfl", "simp", "omega")


@dataclass(frozen=True)
class State:
    left: int
    branch: str = ""
    probe: tuple[str, str] = ("", "")
    last: tuple[str, str] = ("", "")


class ResidualSearch:
    def __init__(self, task):
        self.root = task["target"]
        self.sibling = task["sibling"]

    def node(self, state):
        return self.root["actions"][state.branch]["child"] if state.branch else self.root

    def menu(self, state):
        moves = [STOP]
        if not state.left:
            return moves
        if not state.last[0] and not state.probe[0]:
            moves.extend((PROBE, tactic) for tactic in PROBES if tactic in self.sibling["actions"])
        blocked = state.last[0] if state.last[1] not in ("", "progress") else ""
        moves.extend((TARGET, tactic) for tactic in self.node(state)["actions"] if tactic != blocked)
        return moves

    def transition(self, state, action):
        role, tactic = action
        if role == STOP[0]:
            return 0., None, ""
        if role == PROBE:
            edge = self.sibling["actions"][tactic]
            return -FEES[PROBE], replace(state, probe=(tactic, edge["status"])), edge["status"]
        edge = self.node(state)["actions"][tactic]
        status = edge["status"]
        reward = float(status == "closed") - FEES[TARGET]
        child = None
        if state.left > 1 and status != "closed":
            branch = tactic if status == "progress" else state.branch
            child = State(state.left - 1, branch, state.probe, (tactic, status))
        return reward, child, status

    def run(self, budget, choose):
        state, script, trace, utility = State(budget), [], [], 0.
        while state is not None:
            action = choose(state)
            if action not in self.menu(state):
                raise ValueError("the residual policy selected an unavailable tactic")
            reward, following, status = self.transition(state, action)
            if action != STOP:
                role, tactic = action
                trace.append({"role": role, "action": tactic, "status": status})
                if role == TARGET and status in ("closed", "progress"):
                    script.append(tactic)
            utility += reward
            state = following
        return {"utility": utility, "trace": trace, "script": script,
                "proof": bool(trace and trace[-1]["role"] == TARGET and trace[-1]["status"] == "closed")}
