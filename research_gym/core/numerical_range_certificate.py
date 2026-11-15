"""NRF-01: exact rational 2x2 Jordan numerical-range loop certificate.

A finite-dimensional toy benchmark. No neural checkpoint is certified.
All distances and margins use Euclidean norm with exact fractions.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from math import prod

Vector = tuple[Q, Q]


def project_box(x: Vector) -> Vector:
    """Euclidean metric projection onto [-1,1]^2."""
    return tuple(max(Q(-1), min(Q(1), v)) for v in x)


def orthogonal_gate(name: str, x: Vector) -> Vector:
    a, b = x
    if name == "rot90":
        return (b, -a)
    if name == "rot270":
        return (-b, a)
    if name == "identity":
        return (a, b)
    if name == "negate":
        return (-a, -b)
    raise ValueError(f"unknown gate: {name}")


def sqdist(x: Vector, y: Vector) -> Q:
    return sum(((a - b) ** 2 for a, b in zip(x, y)), Q(0))


@dataclass(frozen=True)
class Phase:
    """A=center I+shear E12; p(z)=sum coeff[k]*(z-center)^k.

    W(A) is the disk of radius |shear|/2 about center. This scalar
    polynomial is the 1x1-coefficient case of complete Crouzeix.
    """
    center: Q
    shear: Q
    coeff: tuple[Q, ...]
    eta: Q
    gate: str
    bias: Vector = (Q(0), Q(0))

    def __post_init__(self) -> None:
        if not self.coeff:
            raise ValueError("nonempty polynomial required")
        if not Q(0) < self.eta <= Q(1):
            raise ValueError("eta must belong to (0,1]")
        orthogonal_gate(self.gate, (Q(0), Q(0)))

    @property
    def q(self) -> Q:
        radius = abs(self.shear) / 2
        return sum((abs(c) * radius**k for k, c in enumerate(self.coeff)), Q(0))

    @property
    def gain(self) -> Q:
        return 1 - self.eta + 2 * self.eta * self.q

    def polynomial(self, x: Vector) -> Vector:
        """Exact p(A)x, since (A-center I)^2=0."""
        c0 = self.coeff[0]
        c1 = self.coeff[1] if len(self.coeff) > 1 else Q(0)
        return (c0 * x[0] + c1 * self.shear * x[1], c0 * x[1])

    def step(self, x: Vector) -> Vector:
        old = orthogonal_gate(self.gate, x)
        raw = self.polynomial(x)
        proposed = project_box((raw[0] + self.bias[0], raw[1] + self.bias[1]))
        return ((1 - self.eta) * old[0] + self.eta * proposed[0],
                (1 - self.eta) * old[1] + self.eta * proposed[1])

    def unregularized_residual_sq(self, x: Vector) -> Q:
        return sqdist(self.step(x), orthogonal_gate(self.gate, x))


def cycle_gain(phases: tuple[Phase, ...]) -> Q:
    if not phases:
        raise ValueError("nonempty cycle required")
    return prod((p.gain for p in phases), start=Q(1))


def require_contracting_cycle(phases: tuple[Phase, ...]) -> Q:
    """A full cycle can contract despite an individually expansive phase."""
    gamma = cycle_gain(phases)
    if gamma >= 1:
        raise ValueError("product of phase bounds must be below 1")
    return gamma


def tail_error_radius_sq(phases: tuple[Phase, ...], x: Vector, n: int) -> Q:
    """Squared a posteriori bound from the cycle residual at seed x."""
    if n < 0:
        raise ValueError("negative iteration count")
    gamma = require_contracting_cycle(phases)
    return gamma**(2 * n) * sqdist(cycle(phases, x), x) / (1 - gamma)**2


def cycle(phases: tuple[Phase, ...], x: Vector) -> Vector:
    for phase in phases:
        x = phase.step(x)
    return x


def iterate_cycles(phases: tuple[Phase, ...], x: Vector, count: int) -> Vector:
    if count < 0:
        raise ValueError("negative iteration count")
    for _ in range(count):
        x = cycle(phases, x)
    return x


def noise_budget(phases: tuple[Phase, ...], per_step_noise: Q) -> Q:
    """One-cycle weighted disturbance budget, each perturbation <= delta."""
    if per_step_noise < 0:
        raise ValueError("negative noise")
    bound = Q(0)
    for phase in phases:
        bound = phase.gain * bound + per_step_noise
    return bound


def cumulative_noise_bound(phases: tuple[Phase, ...], noise: Q, count: int) -> Q:
    if count < 0:
        raise ValueError("negative iteration count")
    gamma = require_contracting_cycle(phases)
    return noise_budget(phases, noise) * (1 - gamma**count) / (1 - gamma)


def perturbed_cycle(phases: tuple[Phase, ...], x: Vector, noise: Vector) -> Vector:
    """Project perturbations into the same domain after each phase."""
    for phase in phases:
        y = phase.step(x)
        x = project_box((y[0] + noise[0], y[1] + noise[1]))
    return x


def example_phases() -> tuple[Phase, ...]:
    return (
        Phase(Q(0), Q(2), (Q(1, 8), Q(1, 16), Q(1, 64)),
              Q(1, 3), "rot90", (Q(1, 20), Q(-1, 30))),
        Phase(Q(1, 5), Q(-1), (Q(1, 5), Q(1, 10), Q(1, 100)),
              Q(1, 4), "rot270", (Q(-1, 16), Q(1, 22))),
    )
