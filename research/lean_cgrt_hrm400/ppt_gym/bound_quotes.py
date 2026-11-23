"""Rule-indexed quotes with source-private and development-shared tariff carriers."""

import random

from .data import digest
from .private_quotes import QUOTES, private_steps


REGIMES = ("strict", "direct", "compound", "expensive", "overlap")


def shared_steps(plans):
    def downstream(plan):
        return plan[2 if plan[0] == "constructor" else 1:]

    return tuple(move for move in downstream(plans[0]) if move in downstream(plans[1])
                 and all(plan.count(move) == 1 for plan in plans))


def arity(graph, tactic):
    name = tactic.removeprefix("apply ")
    return len(graph["rules"][int(name[1:])][0]) if name.startswith("r") else 0


def bind(case, *, shared):
    """Assign literal rule identities before training or any policy evaluation."""
    rng = random.Random(int(digest(["bound-tariffs", case["module"]])[:15], 16))
    plans = case["plans"].copy()
    rng.shuffle(plans)
    prime, other = private_steps(plans)
    priced, counterpart = rng.choice(prime), rng.choice(other)
    private_flags = [move for move in (*prime, *other) if move not in (priced, counterpart)]
    if shared:
        flags = [rng.choice(shared_steps(plans)), rng.choice(private_flags)]
    else:
        flags = rng.sample(private_flags, 2)
    quotes = list(QUOTES)
    rng.shuffle(quotes)
    flag_u, flag_v, band = quotes
    channels = {flag_u: {"rule": flags[0]}, flag_v: {"rule": flags[1]},
                band: {"rule": priced, "counterpart": counterpart}}
    rng.shuffle(quotes)
    return {**case, "plans": plans, "channels": channels, "quote_order": quotes, "shared_tariff": shared}


def public_view(case, regime):
    base = len(case["plans"][0])
    band = next(name for name, channel in case["channels"].items() if "counterpart" in channel)
    slack = {"strict": 0, "direct": 2 + int(case["module"][-1], 16) % 2,
             "compound": 4, "expensive": 2, "overlap": 6 + int(case["module"][-2], 16) % 2}[regime]
    fees = dict.fromkeys(QUOTES, 1)
    if regime in ("compound", "expensive"):
        fees[band] = 5
    links = {}
    for name, channel in case["channels"].items():
        rule = channel["rule"]
        mate = channel.get("counterpart")
        links[name] = {"arity": arity(case["graph"], rule), "banded": int(mate is not None),
                       "direct": [plan.count(rule) for plan in case["plans"]],
                       "inverse": [plan.count(mate) for plan in case["plans"]]}
    return {"plans": case["plans"], "quote_order": case["quote_order"], "fees": fees, "links": links,
            "min_proof_effort": base, "budget": base + slack, "max_surcharge": 7}


def outcomes(case):
    """The tariff meter uses private billing flags, kept out of model inputs."""
    flags = [name for name in QUOTES if "counterpart" not in case["channels"][name]]
    band = next(name for name in QUOTES if name not in flags)
    rule = case["channels"][band]["rule"]
    mate = case["channels"][band]["counterpart"]
    high = 5 + min(arity(case["graph"], rule), 2)
    result = []
    for world in range(4):
        u, v = divmod(world, 2)
        parity = u ^ v
        bits = {flags[0]: u, flags[1]: v, band: parity}
        prices = dict.fromkeys((step for plan in case["plans"] for step in plan), 1)
        prices[case["channels"][flags[0]]["rule"]] = 1 + u
        prices[case["channels"][flags[1]]["rule"]] = 1 + v
        prices[rule] = 1 + high * parity
        prices[mate] = 1 + high * (1 - parity)
        if bits != {name: int(prices[entry["rule"]] > 1) for name, entry in case["channels"].items()}:
            raise ValueError("the observation differs from the charged rule price")
        result.append((prices, bits))
    return result
