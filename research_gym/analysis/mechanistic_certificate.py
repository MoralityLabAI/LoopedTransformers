"""MC-01: exact relations for a declared pointer-state abstraction.

Standard-library reference. Passing R1CS proves only the abstract trace relation.
Source linkage requires a separately trusted observation, and causal fidelity
requires interventions. No cryptographic prover or neural decoder is implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

BN254 = 21888242871839275222246405745257275088548364400416034343698204186575808495617
M31 = 2147483647
FIELDS = (BN254, M31)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def integer(value: object, lower: int, upper: int, name: str) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name}: expected integer in [{lower}, {upper}]")
    return value


@dataclass(frozen=True)
class Statement:
    graph: tuple[int, ...]
    start: int
    steps: int
    finish: int

    def __post_init__(self) -> None:
        if not isinstance(self.graph, tuple):
            raise ValueError("graph must be an immutable tuple")
        integer(len(self.graph), 2, 32, "node count")
        integer(self.steps, 1, 64, "steps")
        for i, value in enumerate((*self.graph, self.start, self.finish)):
            integer(value, 0, len(self.graph) - 1, f"node[{i}]")

    def public(self) -> list[int]:
        n = len(self.graph)
        return [int(self.graph[i] == j) for i in range(n) for j in range(n)] + [
            int(self.start == j) for j in range(n)] + [
            int(self.finish == j) for j in range(n)]

    def identity(self) -> dict:
        return {"schema": "mc01.pointer.v1", "graph": list(self.graph),
                "start": self.start, "steps": self.steps, "finish": self.finish}


def oracle(graph: Sequence[int], start: int, steps: int) -> tuple[int, ...]:
    # Validate before indexing, including Python's otherwise valid negative indices.
    Statement(tuple(graph), start, steps, start)
    out = [start]
    for _ in range(steps):
        out.append(graph[out[-1]])
    return tuple(out)


# A linear expression is a tuple of (wire index, integer coefficient).
# Wire 0 is the constant 1; the remaining wires begin with public values.
Linear = tuple[tuple[int, int], ...]
Constraint = tuple[str, Linear, Linear, Linear]


@dataclass(frozen=True)
class Relation:
    n: int
    steps: int
    public_count: int
    names: tuple[str, ...]
    constraints: tuple[Constraint, ...]

    def failures(self, witness: Sequence[int], field: int = BN254) -> list[str]:
        if type(field) is not int or field not in FIELDS:
            raise ValueError("only the two audited prime fields are supported")
        if len(witness) != len(self.names):
            raise ValueError("witness length differs from the circuit layout")
        for i, value in enumerate(witness):
            integer(value, 0, field - 1, f"wire[{i}]")
        if witness[0] != 1:
            raise ValueError("constant wire must be one")
        def evaluate(expr: Linear) -> int:
            return sum(witness[i] * coefficient for i, coefficient in expr) % field
        return [label for label, a, b, c in self.constraints
                if (evaluate(a) * evaluate(b) - evaluate(c)) % field]

    def circom(self) -> str:
        """Emit an explicit witness-checking circuit; compilation is a separate gate."""
        def signal(i: int) -> str:
            if i == 0:
                return "1"
            if i <= self.public_count:
                return f"public_values[{i - 1}]"
            return f"private_values[{i - self.public_count - 1}]"
        def expression(expr: Linear) -> str:
            return " + ".join(f"({coefficient})*{signal(i)}" for i, coefficient in expr) or "0"
        lines = ["pragma circom 2.1.6;", "// MC-01 abstract relation only. Compile with the default BN254 field.",
                 "// Public: graph rows, initial one-hot, final one-hot. Private: states and products.",
                 "// Source hashes and neural provenance are deliberately absent from this relation.",
                 "template PointerTrace() {", f"  signal input public_values[{self.public_count}];",
                 f"  signal input private_values[{len(self.names) - 1 - self.public_count}];"]
        for label, a, b, c in self.constraints:
            lines += [f"  // {label}", f"  ({expression(a)}) * ({expression(b)}) === ({expression(c)});"]
        lines += ["}", "component main {public [public_values]} = PointerTrace();", ""]
        return "\n".join(lines)

    def description(self) -> dict:
        return {"schema": "mc01.r1cs.v1", "n": self.n, "steps": self.steps,
                "public_count": self.public_count, "names": self.names,
                "constraints": self.constraints,
                "source_binding": "outside_relation", "compiled": False}


def relation(n: int, steps: int) -> Relation:
    integer(n, 2, 32, "node count")
    integer(steps, 1, 64, "steps")
    names = ["one"]
    def allocate(name: str) -> int:
        names.append(name)
        return len(names) - 1
    edge = [[allocate(f"edge_{i}_{j}") for j in range(n)] for i in range(n)]
    initial = [allocate(f"initial_{j}") for j in range(n)]
    final = [allocate(f"final_{j}") for j in range(n)]
    public_count = len(names) - 1
    state = [[allocate(f"q_{t}_{j}") for j in range(n)] for t in range(steps + 1)]
    product = [[[allocate(f"product_{t}_{i}_{j}") for j in range(n)]
                for i in range(n)] for t in range(steps)]
    rows: list[Constraint] = []
    one: Linear = ((0, 1),)
    def equal(label: str, left: Linear, right: Linear) -> None:
        rows.append((label, left, one, right))
    def one_hot(label: str, indices: list[int]) -> None:
        for j, index in enumerate(indices):
            rows.append((f"{label}.bit.{j}", ((index, 1),), ((index, 1), (0, -1)), ()))
        equal(f"{label}.sum", tuple((i, 1) for i in indices), one)
    for i in range(n):
        one_hot(f"edge.{i}", edge[i])
    for t in range(steps + 1):
        one_hot(f"state.{t}", state[t])
    for j in range(n):
        equal(f"initial.{j}", ((state[0][j], 1),), ((initial[j], 1),))
        equal(f"final.{j}", ((state[steps][j], 1),), ((final[j], 1),))
    for t in range(steps):
        for i in range(n):
            for j in range(n):
                rows.append((f"product.{t}.{i}.{j}", ((state[t][i], 1),),
                             ((edge[i][j], 1),), ((product[t][i][j], 1),)))
        for j in range(n):
            equal(f"transition.{t}.{j}", ((state[t + 1][j], 1),),
                  tuple((product[t][i][j], 1) for i in range(n)))
    return Relation(n, steps, public_count, tuple(names), tuple(rows))


def encode(statement: Statement, labels: Sequence[int]) -> list[int]:
    n = len(statement.graph)
    if len(labels) != statement.steps + 1:
        raise ValueError("labels must include the input boundary and every visit")
    for i, label in enumerate(labels):
        integer(label, 0, n - 1, f"label[{i}]")
    q = [[int(label == j) for j in range(n)] for label in labels]
    edge = [[int(statement.graph[i] == j) for j in range(n)] for i in range(n)]
    return [1] + statement.public() + [v for row in q for v in row] + [
        q[t][i] * edge[i][j] for t in range(statement.steps) for i in range(n) for j in range(n)]


def check(statement: Statement, labels: Sequence[int], *, expected: Statement | None = None,
          trusted_observed: Sequence[int] | None = None, field: int = BN254) -> dict:
    """Return separate abstract and source-linked verdicts, failing closed on missing evidence.

    expected and trusted_observed must originate outside the candidate witness.
    Equality to caller-supplied copies has no authentication meaning.
    """
    circuit = relation(len(statement.graph), statement.steps)
    failures = circuit.failures(encode(statement, labels), field)
    statement_bound = expected is not None and statement == expected
    if trusted_observed is not None:
        encode(statement, trusted_observed)  # Strictly validate shape and labels.
    observations_bound = trusted_observed is not None and tuple(labels) == tuple(trusted_observed)
    return {"abstract_valid": not failures, "constraint_failures": failures,
            "statement_bound": statement_bound, "observations_bound": observations_bound,
            "trusted_capture_accept": not failures and statement_bound and observations_bound,
            "causal_fidelity": "unmeasured", "cryptographic_source_binding": False,
            "statement_sha256": digest(statement.identity())}


def audit_finite(transitions: Sequence[Sequence[int]], alpha: Sequence[int],
                 abstract_transitions: Sequence[Sequence[int]], output: Sequence[int],
                 abstract_output: Sequence[int], domain: Sequence[int] | None = None) -> dict:
    """Exhaustively audit a finite candidate under every declared operation.

    Operations include any intended interventions. Domain closure is checked;
    a certificate on a subset never silently covers states reached outside it.
    """
    h, q = len(alpha), len(abstract_output)
    if h == 0 or q == 0 or not transitions or len(transitions) != len(abstract_transitions):
        raise ValueError("nonempty, matching concrete/abstract operation alphabets required")
    if len(output) != h or any(len(row) != h for row in transitions) or any(
            len(row) != q for row in abstract_transitions):
        raise ValueError("transition or readout dimensions disagree")
    for value in alpha:
        integer(value, 0, q - 1, "abstract label")
    for row in transitions:
        for value in row:
            integer(value, 0, h - 1, "concrete successor")
    for row in abstract_transitions:
        for value in row:
            integer(value, 0, q - 1, "abstract successor")
    for value in (*output, *abstract_output):
        if type(value) is not int:
            raise ValueError("finite outputs must be integer symbols")
    states = tuple(range(h)) if domain is None else tuple(domain)
    if not states or len(set(states)) != len(states):
        raise ValueError("domain must be nonempty and unique")
    for state in states:
        integer(state, 0, h - 1, "domain state")
    allowed = set(states)
    defects = []
    for state in states:
        if output[state] != abstract_output[alpha[state]]:
            defects.append({"kind": "readout", "state": state})
        for op, row in enumerate(transitions):
            target = row[state]
            if target not in allowed:
                defects.append({"kind": "domain_escape", "state": state, "op": op})
            if alpha[target] != abstract_transitions[op][alpha[state]]:
                defects.append({"kind": "commutation", "state": state, "op": op})
    return {"exact_on_declared_domain": not defects, "states": len(states),
            "operations": len(transitions), "defects": defects,
            "scope": "finite declared transition tables, including listed interventions"}
