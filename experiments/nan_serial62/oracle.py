"""Independent rendered-table replay and strict reply verification."""
import re

CODE = r"[0-9]{2}"
EDGE = re.compile(rf"({CODE}) -> ({CODE})")
QUERY = re.compile(rf"Start at ({CODE})\. Lookups: ([1-9][0-9]*)\.")
FINAL = re.compile(rf"FINAL=({CODE});END")
TRACE = re.compile(rf"PATH=({CODE}(?:>{CODE})*);FINAL=({CODE});END")


def read_table(prompt):
    lines = iter(prompt.splitlines())
    for line in lines:
        if line == "Table:":
            break
    else:
        raise ValueError("Lookup table delimiter is missing")
    edges = {}
    erased = False
    for line in lines:
        query = QUERY.fullmatch(line)
        if query:
            if erased == bool(edges):
                raise ValueError("Expected either a complete table or an erased marker")
            return (None if erased else edges), query[1], int(query[2])
        if line == "[table withheld]" and not erased and not edges:
            erased = True
            continue
        match = EDGE.fullmatch(line)
        if not match or erased or match[1] in edges:
            raise ValueError("Invalid or duplicate lookup edge")
        edges[match[1]] = match[2]
    raise ValueError("Lookup query is missing")


def replay(prompt):
    edges, start, steps = read_table(prompt)
    if edges is None:
        return None
    path = [start]
    for _ in range(steps):
        path.append(edges[path[-1]])
    return path


def parse(text, readout, hops):
    text = text.strip(" \t\r\n")
    if readout == "direct":
        match = FINAL.fullmatch(text)
        return (None, match[1]) if match else None
    if readout != "trace":
        raise ValueError(f"Unknown lookup readout {readout}")
    match = TRACE.fullmatch(text)
    if not match:
        return None
    path = match[1].split(">")
    return (path, match[2]) if len(path) == hops else None


def grade(row, text):
    parsed = parse(text, row["readout"], row["hops"])
    gold = replay(row["prompt"])
    if gold is None:
        gold = row["path"]  # Same hidden labels as the twins with tables.
    elif gold != row["path"]:
        raise ValueError("Rendered-table replay differs from the generated label")
    final_correct = parsed is not None and parsed[1] == gold[-1]
    path_correct = None
    if row["readout"] == "trace":
        path_correct = parsed is not None and parsed[0] == gold[1:]
    return {"format_valid": parsed is not None, "final_exact": final_correct,
            "trace_exact": path_correct, "joint_exact": final_correct and path_correct is not False}
