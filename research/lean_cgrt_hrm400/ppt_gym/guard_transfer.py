"""Public selection of same-depth guard pairs with disjoint closed expressions."""

from collections import Counter

from .computational_guards import FAMILIES
from .data import digest
from .guard_recovery import cases


NAMESPACE = "probe-transfer-unseen-expressions-1127"
ROOTS_PER_FAMILY = 32
CANDIDATE_LIMIT = 50_000
PUBLIC_KEYS = ("module", "pair", "family", "statement", "world", "guards")


def expressions(rows):
    return {guard.split(" = ", 1)[0] for row in rows for witness in row["guards"] for guard in witness}


def guard_menu(row):
    return tuple(sorted(tuple(sorted(witness)) for witness in row["guards"]))


def public_digest(rows):
    return digest([{key: row[key] for key in PUBLIC_KEYS} for row in rows])


def unseen_pairs(previous):
    blocked = expressions(previous)
    seen = {guard_menu(row) for row in previous}
    accepted, examined, plan = Counter(), Counter(), []
    used = Counter()
    stream = cases("development", CANDIDATE_LIMIT, namespace=NAMESPACE)
    for root in zip(stream, stream, strict=True):
        family = root[0]["family"]
        if accepted[family] == ROOTS_PER_FAMILY:
            continue
        examined[family] += 1
        signature = guard_menu(root[0])
        terms = expressions(root)
        if signature in seen or terms & blocked:
            continue
        seen.add(signature)
        plan.extend(root)
        used.update(terms)
        accepted[family] += 1
        if all(accepted[key] == ROOTS_PER_FAMILY for key in FAMILIES):
            break
    if accepted != {key: ROOTS_PER_FAMILY for key in FAMILIES}:
        raise ValueError("the fixed public candidate stream did not yield the registered number of roots")
    for key in ("pair", "module", "statement"):
        if {row[key] for row in plan} & {row[key] for row in previous}:
            raise ValueError(f"the transfer cohort overlaps prior rows on {key}")
    return plan, {"roots": dict(accepted), "examined": dict(examined), "excluded_closed_expressions": len(blocked),
                  "new_closed_expressions": len(used), "maximum_roots_per_expression": max(used.values())}
