"""LBF-05 bounded hybrid Lean-public-state serializer.

Fuses explicit candidate-premise edges with a reserved suffix of the Lean
server's *public* pretty-printed state. Strict input allowlist; no verifier
receipts, labels, reference paths, or model outcomes are accepted.

Called only via ppt_gym.typed_state.encode_public_packet(view="hybrid").
"""
from __future__ import annotations

import re
from typing import Sequence

SCHEMA = "lbf05.hybrid_public_state.v1"
PP_RESERVE_MAX = 112


def _utf8_suffix(source: str, budget: int) -> str:
    """Return a valid, complete-codepoint UTF-8 suffix within byte budget."""
    if budget <= 0:
        return ""
    raw = source.encode("utf-8")
    if len(raw) <= budget:
        return source
    # Slicing may start inside a multibyte character. Ignore only the
    # incomplete prefix; the remaining characters are complete.
    return raw[-budget:].decode("utf-8", errors="ignore")


def hybrid_public_packet(packet: str, candidates: Sequence[str], max_bytes: int = 384) -> str:
    """Encode header + typed dependency rules + public Lean pretty-text suffix.

    The registered relative allocation is 2/3 structured facts and up to 1/3
    readable context, capped at 112 bytes. Mandatory header fits or fails.
    Candidate rules precede background rules. Missing optional rules are
    omitted completely; no partial rule is emitted.
    """
    from .typed_state import _public, _formula, _type_entries, _dependencies

    if type(max_bytes) is not int or not 32 <= max_bytes <= 384:
        raise ValueError("hybrid byte budget must be between 32 and 384")
    if not 1 <= len(candidates) <= 64 or len(set(candidates)) != len(candidates):
        raise ValueError("hybrid requires 1..64 distinct public candidates")

    goals, remaining, failed = _public(packet)
    for goal in goals:
        if set(goal) - {"target", "hypotheses", "pp"}:
            raise ValueError("hybrid input contains non-public goal fields")
        if "pp" in goal and not isinstance(goal["pp"], str):
            raise ValueError("public pretty-print must be text")
    first = goals[0]
    entries = _type_entries(first)
    names = {name: (premises, head) for name, premises, head in entries}
    selected = []
    for candidate in candidates:
        if candidate == "constructor":
            if set(first["target"]) != {"and"}:
                raise ValueError("constructor candidate not public")
        elif candidate.startswith("apply ") and candidate[6:] in names:
            selected.append(candidate[6:])
        else:
            raise ValueError("candidate missing from public Lean hypothesis context")

    tasks = ",".join(_formula(g["target"]) for g in goals)
    header = f"B{remaining}|G:{tasks}|"
    if failed:
        # Failures are already public; preserve only legal public tactic names.
        names_failed = []
        for tactic in failed:
            if tactic == "constructor":
                names_failed.append(tactic)
            elif tactic.startswith("apply ") and re.fullmatch(r"[hr][0-9]{1,4}", tactic[6:]):
                names_failed.append(tactic[6:])
            else:
                raise ValueError("unrecognized public failed tactic")
        header += "F:" + ",".join(names_failed) + "|"
    if "constructor" in candidates:
        header += "A:constructor|"
    if len(header.encode("utf-8")) > max_bytes:
        raise ValueError("hybrid mandatory goal header exceeds byte budget")

    # The actual Lean 'pp' field is a public server-returned observation.
    # Whitespace normalization preserves source token order; raw suffix is
    # preferred because the legacy raw model sees the end of this field.
    pretty = first.get("pp", "")
    pretty = re.sub(r"\s+", " ", pretty).strip().replace("|", "/")
    reserve = min(PP_RESERVE_MAX, max_bytes // 3) if pretty else 0

    # Keep candidate-named rules in the public menu order; other clauses are
    # sorted by goal-relevant syntactic graph distance then name.
    distance = _dependencies(entries, _formula(first["target"]))
    ordered = list(dict.fromkeys(selected))
    ordered += [
        name for name, _, head in sorted(entries, key=lambda e: (distance.get(e[2], 100_000), e[0]))
        if name not in ordered
    ]
    output = header
    for name in ordered:
        premises, head = names[name]
        rule = name + ":" + ("+".join(premises) + ">" if premises else "") + head + ";"
        if len((output + rule).encode("utf-8")) <= max_bytes - reserve:
            output += rule

    # The pretty-print suffix uses the remaining complete-codepoint budget.
    # If no room is left after the mandatory header, the structured view
    # remains valid and reports no pretty channel.
    remaining_bytes = max_bytes - len(output.encode("utf-8"))
    if pretty and remaining_bytes >= 7:
        suffix = _utf8_suffix(pretty, remaining_bytes - 3)
        if suffix:
            output += "P:" + suffix + "|"
    assert len(output.encode("utf-8")) <= max_bytes
    return output
