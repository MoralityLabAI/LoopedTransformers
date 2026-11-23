"""Paid price observations over a fixed menu of native Lean proof plans."""

from functools import lru_cache
from typing import NamedTuple


class Regime(NamedTuple):
    slack: int
    fees: tuple
    surcharge: int


REGIMES = {
    "strict": Regime(0, (1, 1, 1), 6),
    "direct": Regime(1, (1, 1, 1), 6),
    "two_quotes": Regime(2, (1, 1, 3), 6),
    "too_expensive": Regime(1, (1, 1, 3), 6),
    "either_plan": Regime(1, (1, 1, 1), 0),
}
QUOTES = ("ledger_u", "ledger_v", "reconcile")
COMMITS = ("commit_0", "commit_1")
PENALTY = 1 / 16


def private_steps(plans):
    """Eligible tariff carriers occur once, after an equal-price first producer."""
    return tuple(tuple(dict.fromkeys(
        move for move in plan[2 if plan[0] == "constructor" else 1:]
        if move not in other and plan.count(move) == 1
    )) for plan, other in (plans, plans[::-1]))


def tariff_layout(plans):
    carriers, other = private_steps(plans)
    if len(plans[0]) != len(plans[1]) or len(carriers) < 3 or not other:
        raise ValueError("each plan needs its own downstream tariff carriers")
    return dict(zip(QUOTES, carriers[:3], strict=True)), other[0]


def scenario(plans, ledger, other, regime, world):
    """Compute the environment's actual prices from two unobserved billing flags."""
    u, v = divmod(world, 2)
    parity = u ^ v
    price = dict.fromkeys((*plans[0], *plans[1]), 1)
    price[ledger["ledger_u"]] += u
    price[ledger["ledger_v"]] += v
    price[ledger["reconcile"]] += 2 - u - v + regime.surcharge * parity
    price[other] += 2 + regime.surcharge * (1 - parity)
    return price


def quote_bit(quote, ledger, prices):
    price = prices[ledger[quote]]
    return int(price > (3 if quote == "reconcile" else 1))


def available(order, fees, history, left):
    used = {action for action, _ in history}
    return (*COMMITS, *(action for action in order if action not in used and fees[action] <= left))


def public_view(case, regime_name):
    regime = REGIMES[regime_name]
    base = len(case["plans"][0]) + 2
    return {"plans": case["plans"], "ledger": case["ledger"], "fees": dict(zip(QUOTES, regime.fees, strict=True)),
            "quote_order": case["quote_order"], "min_proof_effort": base, "budget": base + regime.slack,
            "max_surcharge": regime.surcharge}


def rollout(case, regime_name, world, choose, *, patch=None):
    """Charge actual prices while exposing only a public view and paid observations."""
    regime = REGIMES[regime_name]
    plans, ledger = case["plans"], case["ledger"]
    price = scenario(plans, ledger, case["other"], regime, world)
    view = public_view(case, regime_name)
    initial = view["budget"]
    left = initial
    fees = view["fees"]
    history, trace = (), []
    while True:
        menu = available(case["quote_order"], fees, history, left)
        action = choose(menu, history, left, fees)
        if action not in menu:
            raise ValueError("a policy chose an unavailable diagnostic")
        if action in COMMITS:
            plan = plans[COMMITS.index(action)]
            spent, script = 0, []
            for move in plan:
                cost = price[move]
                if spent + cost > left:
                    break
                spent += cost
                script.append(move)
            proof = len(script) == len(plan)
            return {"proof": proof, "script": script, "plan": action, "trace": trace,
                    "budget": initial, "proof_effort": spent, "quote_effort": initial - left,
                    "utility": int(proof) - PENALTY * (initial - left)}
        bit = quote_bit(action, ledger, price)
        seen = bit if patch is None else patch(action, bit)
        if seen not in (0, 1):
            raise ValueError("a quote observation must have a binary value")
        left -= fees[action]
        trace.append({"action": action, "charge": fees[action], "actual": bit, "seen": seen})
        history = (*history, (action, seen))


def structure(case, regime_name, *, one_step=False, always_quote=False, reserve_proof_effort=False):
    """Exact likelihood ceiling given the tariff generator and a uniform private world."""
    regime = REGIMES[regime_name]
    plans, ledger = case["plans"], case["ledger"]
    worlds = [scenario(plans, ledger, case["other"], regime, world) for world in range(4)]

    @lru_cache(maxsize=None)
    def values(history, left, allow_quote):
        support = [tariffs for tariffs in worlds
                   if all(quote_bit(action, ledger, tariffs) == bit for action, bit in history)]
        # The intervention can create a history that is impossible under the real prices.
        # Fall back to the prior on that history, without repeating an observation.
        support = support or worlds
        scores = {action: sum(sum(tariffs[move] for move in plan) <= left for tariffs in support) / len(support)
                  for action, plan in zip(COMMITS, plans, strict=True)}
        if allow_quote:
            for action in available(case["quote_order"], fees, history, left)[len(COMMITS):]:
                chance = sum(quote_bit(action, ledger, tariffs) for tariffs in support) / len(support)
                child = 0.
                for bit, weight in ((0, 1 - chance), (1, chance)):
                    if weight:
                        successor = values((*history, (action, bit)), left - fees[action], not one_step)
                        child += weight * max(successor.values())
                scores[action] = child - PENALTY * fees[action]
        return scores

    fees = dict(zip(QUOTES, regime.fees, strict=True))

    def choose(menu, history, left, public_fees):
        if public_fees != fees:
            raise ValueError("the registered quote charges changed")
        scores = values(history, left, True)
        reserve = public_view(case, regime_name)["min_proof_effort"] if reserve_proof_effort else 0
        quotes = [action for action in menu[len(COMMITS):] if fees[action] <= left - reserve]
        options = quotes if always_quote and quotes else menu
        return max(options, key=scores.__getitem__)

    return choose
