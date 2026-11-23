"""Source-only supervision from checked fragments and paid native prefixes."""

from collections import defaultdict
from math import fsum

from .data import digest
from .live_quote_fragments import NativeFragments, NativeQuoteBackup
from .private_quotes import PENALTY, QUOTES


def witness_prefixes(log):
    """First recorded native path to each snapshot, independent of billing."""
    paths = {log["root"]: ()}
    suffixes = defaultdict(list)
    for fragment in log["fragments"]:
        prefix = paths[fragment["start"]]
        for index, step in enumerate(fragment["steps"]):
            suffixes[step["before"]].append(tuple(fragment["steps"][index:]))
            prefix = (*prefix, step["action"])
            if step["status"] == "progress":
                paths.setdefault(step["after"], prefix)
    return paths, suffixes


def observed_anchors(model, budget, prefix, purchases):
    """Condition only on paid quote bits and executed tactic charges."""
    grouped = defaultdict(list)
    for world in model.initial_belief:
        quoted = tuple((move, model.quote_bit(world, move)) for move in purchases)
        charged = tuple((move, model.tactic_price(world, move)) for move in prefix)
        left = budget - sum(price for _, price in charged) - sum(model.fees[move] for move in purchases)
        if left > 0 and len(prefix) < 8:
            # Replay the prefix first, then these quotes, using the same allowance.
            grouped[quoted, charged, left].append(world)
    return [(quotes, charges, left, tuple(worlds)) for (quotes, charges, left), worlds in grouped.items()]


def empirical_suffix(world, model, steps, left, ticks):
    """Evaluate a suffix as recorded and censor ordinary cuts before closure."""
    for step in steps:
        price = model.tactic_price(world, step["action"])
        status = step["status"]
        if price > left or not ticks:
            return 0.
        left -= price
        ticks -= 1
        if status == "closed":
            return 1.
        if status == "resource_limit":
            return None
        if status in ("rejected", "no_change"):
            return 0.
        if not model.native.states[step["after"]]["menu"]:
            return 0.
    return None if left and ticks else 0.


def suffix_returns(model, suffixes, belief, left, ticks, move):
    fee = model.fees[move] if move in QUOTES else 0
    # For a new quote the behavior suffix is fixed before seeing its answer.
    tapes = suffixes if fee else [tape for tape in suffixes if tape[0]["action"] == move]
    observed = []
    for tape in tapes:
        for world in belief:
            value = empirical_suffix(world, model, tape, left - fee, ticks)
            if value is not None:
                observed.append(value - PENALTY * fee)
    return {"mean": fsum(observed) / len(observed) if observed else None,
            "observed": len(observed), "offered": len(tapes) * len(belief)}


def build_targets(case):
    native = NativeFragments(case["native"])
    paths, suffixes = witness_prefixes(case["native"])
    rows = []
    for regime, view in case["views"].items():
        model = NativeQuoteBackup(native, view["fees"], case["billing"])
        for key, prefix in paths.items():
            if len(prefix) >= 8 or not native.states[key]["menu"]:
                continue
            phase = int(digest([case["module"], key, regime])[:8], 16) % len(QUOTES)
            # Always retain zero and one quote context, with a rotating pair for credit assignment.
            patterns = [(), *((action,) for action in QUOTES),
                        (QUOTES[phase], QUOTES[(phase + 1) % len(QUOTES)])]
            for purchases in patterns:
                bought = sum(1 << QUOTES.index(move) for move in purchases)
                ticks, quotes = 8 - len(prefix), len(QUOTES) - len(purchases)
                for paid, charges, left, belief in observed_anchors(model, view["budget"], prefix, purchases):
                    labels = {}
                    for move in model.menu(key, left, ticks, quotes, bought):
                        lo, hi = model.action(key, left, ticks, belief, quotes, bought, move)
                        labels[move] = {"bellman": [lo, hi], "suffix": suffix_returns(
                            model, suffixes[key], belief, left, ticks, move)}
                    rows.append({"regime": regime, "state": key, "charges": charges, "quotes": paid,
                                 "remaining": left, "ticks": ticks, "targets": labels})
        model.action.cache_clear()
        model.value.cache_clear()
    return rows
