"""Original exact-integer recurrent cells for the MIC-18 delayed-query task.

All affine entries execute, including zeros. Public write/query gates carry
addresses. The query head accepts only the committed state and query gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def bits(values: Iterable[int], width: int) -> tuple[int, ...]:
    value = tuple(values)
    if len(value) != width or any(type(x) is not int or x not in (0, 1) for x in value):
        raise ValueError("expected a binary integer vector of the declared width")
    return value


def one_hot(width: int, index: int) -> tuple[int, ...]:
    if type(width) is not int or width < 1 or type(index) is not int or not 0 <= index < width:
        raise ValueError("invalid gate width or index")
    return tuple(int(i == index) for i in range(width))


def word_bits(word: int, width: int) -> tuple[int, ...]:
    if type(width) is not int or width < 1 or type(word) is not int or not 0 <= word < 2**width:
        raise ValueError("word lies outside the declared binary domain")
    return tuple((word >> (width - 1 - i)) & 1 for i in range(width))


def pack(values: Iterable[int]) -> int:
    values = tuple(values)
    bits(values, len(values))
    result = 0
    for value in values:
        result = 2 * result + value
    return result


@dataclass(frozen=True)
class ReluNetwork:
    """Two affine maps with an integer ReLU between them."""

    w_in: tuple[tuple[int, ...], ...]
    b_hidden: tuple[int, ...]
    w_out: tuple[tuple[int, ...], ...]
    b_out: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.w_in or not self.w_out or not self.w_in[0]:
            raise ValueError("empty affine layer")
        input_width = len(self.w_in[0])
        hidden_width = len(self.w_in)
        if any(len(row) != input_width for row in self.w_in):
            raise ValueError("ragged input matrix")
        if any(len(row) != hidden_width for row in self.w_out):
            raise ValueError("ragged output matrix")
        if len(self.b_hidden) != hidden_width or len(self.b_out) != len(self.w_out):
            raise ValueError("bias width mismatch")
        entries = (*self.b_hidden, *self.b_out, *(x for row in self.w_in for x in row), *(x for row in self.w_out for x in row))
        if any(type(x) is not int for x in entries):
            raise ValueError("integer coefficients required")

    def forward(self, inputs: Iterable[int]) -> tuple[int, ...]:
        inputs = tuple(inputs)
        if len(inputs) != len(self.w_in[0]) or any(type(x) is not int for x in inputs):
            raise ValueError("integer affine input of the declared width required")
        hidden = tuple(max(0, sum(w * x for w, x in zip(row, inputs)) + bias)
                       for row, bias in zip(self.w_in, self.b_hidden))
        return tuple(sum(w * x for w, x in zip(row, hidden)) + bias
                     for row, bias in zip(self.w_out, self.b_out))

    def as_dict(self) -> dict:
        return {"w_in": self.w_in, "b_hidden": self.b_hidden,
                "w_out": self.w_out, "b_out": self.b_out}

    def costs(self) -> dict[str, int]:
        products = sum(map(len, self.w_in)) + sum(map(len, self.w_out))
        coefficients = (*self.b_hidden, *self.b_out,
                        *(x for row in self.w_in for x in row),
                        *(x for row in self.w_out for x in row))
        return {"dense_products": products, "relu_evaluations": len(self.w_in),
                "dense_coefficient_slots": len(coefficients),
                "nonzero_coefficients": sum(x != 0 for x in coefficients),
                "max_abs_coefficient": max(map(abs, coefficients))}


def make_cell(width: int, mode: str = "register") -> ReluNetwork:
    """Same dense shape and executed arithmetic for all three memory programs.

    Register: h_i' = relu(h_i-g_i) + relu(b+g_i-1).
    Padded parity: h_0' = relu(h_0+b) - 2 relu(h_0+b-1).
    Padded last-bit: h_0' = relu(b); other coordinates are zero.
    """
    one_hot(width, 0)
    if mode not in ("register", "parity", "last_bit"):
        raise ValueError("unknown memory program")
    w_in = [[0] * (2 * width + 1) for _ in range(2 * width)]
    b_hidden = [0] * (2 * width)
    w_out = [[0] * (2 * width) for _ in range(width)]
    if mode == "register":
        for i in range(width):
            w_in[2 * i][i] = 1
            w_in[2 * i][width + 1 + i] = -1
            w_in[2 * i + 1][width] = 1
            w_in[2 * i + 1][width + 1 + i] = 1
            b_hidden[2 * i + 1] = -1
            w_out[i][2 * i] = w_out[i][2 * i + 1] = 1
    elif mode == "parity":
        for unit in (0, 1):
            w_in[unit][0] = w_in[unit][width] = 1
        b_hidden[1] = -1
        w_out[0][0], w_out[0][1] = 1, -2
    else:
        w_in[0][width] = w_out[0][0] = 1
    return ReluNetwork(tuple(map(tuple, w_in)), tuple(b_hidden),
                       tuple(map(tuple, w_out)), (0,) * width)


def make_head(width: int, mode: str = "register", task: str = "query") -> ReluNetwork:
    """Fixed query heads with matched dense shapes and activation counts."""
    one_hot(width, 0)
    if mode not in ("register", "parity", "last_bit") or task not in ("query", "parity"):
        raise ValueError("unknown readout contract")
    if task == "parity" and mode != "parity":
        raise ValueError("this parity head reads the parity memory program")
    w_in = [[0] * (2 * width) for _ in range(width)]
    bias = [0] * width
    if task == "parity":
        w_in[0][0] = 1
    elif mode == "register":
        for i in range(width):
            w_in[i][i] = w_in[i][width + i] = 1
            bias[i] = -1
    elif mode == "last_bit":
        w_in[0][0] = w_in[0][2 * width - 1] = 1
        bias[0] = -1
    elif width == 1:
        w_in[0][0] = 1
    # For J>=2, parity and each queried coordinate are independent under the
    # complete uniform word census. Constant zero is an optimal such decoder.
    return ReluNetwork(tuple(map(tuple, w_in)), tuple(bias),
                       ((1,) * width,), (0,))


@dataclass(frozen=True)
class MemoryProgram:
    width: int
    mode: str
    cell: ReluNetwork
    head: ReluNetwork
    task: str = "query"

    @classmethod
    def build(cls, width: int, mode: str = "register", task: str = "query") -> "MemoryProgram":
        return cls(width, mode, make_cell(width, mode), make_head(width, mode, task), task)

    def update(self, state: Iterable[int], incoming: int, phase: int) -> tuple[int, ...]:
        state = bits(state, self.width)
        bits((incoming,), 1)
        gate = one_hot(self.width, phase)
        return bits(self.cell.forward((*state, incoming, *gate)), self.width)

    def query(self, committed_state: Iterable[int], query: int = 0) -> int:
        state = bits(committed_state, self.width)
        gate = one_hot(self.width, query)
        return bits(self.head.forward((*state, *gate)), 1)[0]

    def commit(self, observations: Iterable[int]) -> tuple[int, ...]:
        observations = bits(observations, self.width)
        state = (0,) * self.width
        for phase, incoming in enumerate(observations):
            state = self.update(state, incoming, phase)
        return state

    def resource_account(self) -> dict:
        update, read = self.cell.costs(), self.head.costs()
        return {"allocated_binary_state_bits": self.width,
                "terminal_reachable_states": 2**self.width if self.mode == "register" else 2,
                "public_phase_categories": self.width + 1,
                "public_phase_bits_if_stored": self.width.bit_length(),
                "public_write_gate_coordinates": self.width,
                "public_query_gate_coordinates": self.width,
                "updates_per_word": self.width,
                "update": update, "one_query": read,
                "episode_dense_products": self.width * update["dense_products"] + read["dense_products"],
                "episode_relu_evaluations": self.width * update["relu_evaluations"] + read["relu_evaluations"],
                "codebook_lookup_entries": 0,
                "state_storage_unit": "logical binary coordinates; Python object overhead unmeasured"}


def observer_before_query(committed_state: Iterable[int]) -> int:
    """An engineered constant observer. Blindness is part of the fixture."""
    state = tuple(committed_state)
    bits(state, len(state))
    return 0
