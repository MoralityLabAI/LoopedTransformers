"""Learn the observed billing vocabulary and model the public Horn tactic grammar."""

from collections import Counter, deque
from types import SimpleNamespace

from lcgrt import fragment

from .data import digest
from .private_quotes import QUOTES


def bindings(case):
    links = []
    for name, channel in case["channels"].items():
        primary = channel["rule"]
        inverse = channel.get("counterpart")
        if name == case["late_link"]["quote"]:
            kind, inverse = "late", case["late_link"]["counterpart"]
        elif inverse:
            idx = int(primary.removeprefix("apply r")) if primary.startswith("apply r") else None
            arity = len(case["graph"]["rules"][idx][0]) if idx is not None else 0
            kind = f"band_{arity}"
        else:
            kind = "flag"
        links.append((name, primary, inverse, kind))
    return sorted(links, key=lambda link: QUOTES.index(link[0]))


def fit_meter(source):
    invoices, words = {}, Counter()
    for case in source:
        for world in case["billing"]:
            bits, prices = world["bits"], world["prices"]
            words[tuple(bits[name] for name in QUOTES)] += 1
            for quote, primary, inverse, kind in bindings(case):
                known = [(primary, bits[quote])]
                if inverse:
                    known.append((inverse, 1 - bits[quote]))
                for tactic, bit in known:
                    if tactic in prices:
                        invoices.setdefault(kind, (Counter(), Counter()))[bit][prices[tactic]] += 1
    if len(set(words.values())) != 1:
        raise ValueError("the learned billing worlds require balanced source observations")
    if any(not counts for conditional in invoices.values() for counts in conditional):
        raise ValueError("the billed source has an unobserved price class")
    return {"words": tuple(sorted(words)), "costs": {kind: tuple(counts.most_common(1)[0][0] for counts in conditional)
                                                     for kind, conditional in invoices.items()}}


def predict_billing(case, meter):
    links = bindings(case)
    costs = meter["costs"]
    outcomes = []
    for word in meter["words"]:
        bits = dict(zip(QUOTES, word, strict=True))
        tariffs = {}
        for name, primary, inverse, kind in links:
            if kind not in costs and kind.startswith("band_"):
                arity = int(kind.split("_")[1])
                kind = min((key for key in costs if key.startswith("band_")),
                           key=lambda key: abs(int(key.split("_")[1]) - arity))
            conditional = costs[kind]
            bit = bits[name]
            tariffs[primary] = conditional[bit]
            if inverse:
                tariffs[inverse] = conditional[1 - bit]
        outcomes.append({"bits": bits, "prices": tariffs})
    return outcomes


def grammar(theorem):
    producers = {}
    for i, atom in enumerate(theorem.facts):
        producers.setdefault(atom, {})[f"apply h{i}"] = ()
    for i, (premises, atom) in enumerate(theorem.rules):
        producers.setdefault(atom, {})[f"apply r{i}"] = tuple(premises)
    return producers


def successors(producers, agenda):
    first, *tail = agenda
    if isinstance(first, tuple):
        return {"constructor": (*first, *tail)}
    return {action: (*wanted, *tail) for action, wanted in producers.get(first, {}).items()}


def public_graph(case):
    """Enumerate predicted public proof states, with no access to held Lean edges."""
    theorem = fragment.Theorem.from_json(case["graph"])
    producers = grammar(theorem)
    subject = (tuple(theorem.goal),) if len(theorem.goal) > 1 else tuple(theorem.goal)
    root = digest(subject)
    pending = deque([(subject, 0)])
    states, edges = {}, {}
    while pending:
        agenda, depth = pending.popleft()
        key = digest(agenda)
        if key in states:
            continue
        possibilities = successors(producers, agenda)
        states[key] = {"menu": tuple(possibilities)}
        edges[key] = {}
        if depth >= 8:
            continue
        for action, after in possibilities.items():
            following = digest(after) if after else None
            edges[key][action] = {"status": "progress" if after else "closed", "after": following}
            if after:
                pending.append((after, depth + 1))
    return SimpleNamespace(root=root, states=states, edges=edges)
