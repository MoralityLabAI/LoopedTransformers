"""FP64 extrema of a quadratic on the equality sphere.

The objective is ``v.T @ A @ v + 2 * b.T @ v`` with ``||v|| == 1``.
The reported multiplier obeys ``(A - multiplier * I) @ v == -b``
for both extrema. Its sign is unrestricted. A minimum requires
``A - multiplier * I`` to be positive semidefinite; a maximum requires
the opposite sign.

Non-hard cases use exactly 64 bisections of the secular equation in the
logarithm of the positive distance from its extreme eigenvalue pole.
This resolves near-hard poles that arithmetic bisection from an O(1)
bracket would miss. Hard cases include nonzero b orthogonal to the
extreme eigenspace, with its pseudoinverse solution inside the sphere.

The bounds include conservative floating-point residual and eigensystem
allowances. They are numerical certificates, not interval-arithmetic
proofs. ``certificate_ok`` reports the explicit normalized tolerance
contract in ``SphereExtremum``. Only source-free tensor algebra lives here.
"""

from dataclasses import dataclass
import math

import torch


SECULAR_ITERATIONS = 64
_EPS = torch.finfo(torch.float64).eps
_MIN_POSITIVE = math.nextafter(0.0, 1.0)


@dataclass(frozen=True)
class SphereExtremum:
    """One attained extremum and its floating certificate.

    ``dual_bound`` is a lower bound for a minimum and an upper bound for
    a maximum. ``psd_gap`` is the smallest oriented eigenvalue estimate:
    A-lambda*I for a minimum, lambda*I-A for a maximum. ``psd_lower_bound``
    subtracts the measured eigensystem error and an arithmetic allowance.

    Objective, multiplier, residual, PSD gap, and dual bound use the
    original units. Fields beginning with ``normalized_`` divide those
    units by ``scale = max(||A||_2, ||b||_2)``. ``norm_error`` is
    dimensionless. ``certificate_tolerance`` is dimensionless as well:
    certificate_ok requires norm_error and normalized stationarity
    residual below it, normalized PSD lower bound at least its negative,
    and normalized bound gap at most 32 times it. Exact zero input has
    scale zero and an exact zero certificate.

    ``hard_case`` describes the computed eigensystem. A rotated exact
    hard case can acquire a tiny pole component through rounding and
    then be solved by the secular branch. Both branches have the same
    residual and bound checks.
    """

    sense: str
    vector: torch.Tensor
    value: float
    multiplier: float
    stationarity_residual: float
    normalized_stationarity_residual: float
    norm_error: float
    psd_gap: float
    normalized_psd_gap: float
    psd_lower_bound: float
    dual_bound: float
    dual_multiplier: float
    bound_gap: float
    normalized_bound_gap: float
    certificate_ok: bool
    certificate_tolerance: float
    hard_case: bool
    secular_iterations: int
    scale: float
    eigensystem_error: float
    normalized_eigensystem_error: float
    secular_pole_offset: float

    @property
    def extremizer(self) -> torch.Tensor:
        return self.vector


@dataclass(frozen=True)
class SphereExtrema:
    minimum: SphereExtremum
    maximum: SphereExtremum


def _norm(tensor: torch.Tensor) -> float:
    """Euclidean/Frobenius norm without avoidable squaring underflow."""
    if tensor.numel() == 0:
        return 0.0
    peak = float(tensor.abs().max())
    if peak == 0.0 or not math.isfinite(peak):
        return peak
    return peak * float(torch.linalg.vector_norm(tensor / peak))


def _canonical_direction(basis: torch.Tensor) -> torch.Tensor:
    """Select an eigenspace direction through its coordinate projector.

    Project the coordinate axis with largest projector diagonal, taking
    the first index on exact ties. The largest-magnitude vector entry is
    positive. This depends on the eigenspace projector, including for a
    repeated eigenvalue, instead of a chosen eigenvector basis.
    """
    diagonal = basis.square().sum(dim=1)
    index = int(torch.argmax(diagonal))
    vector = basis @ basis[index]
    vector = vector / _norm(vector)
    pivot = int(torch.argmax(vector.abs()))
    if float(vector[pivot]) < 0.0:
        vector = -vector
    return vector


def _minimum_candidate(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, float, float, bool, int]:
    """Solve the normalized minimum in a supplied ascending eigensystem."""
    bottom = float(eigenvalues[0])
    gaps = eigenvalues - bottom
    pole = gaps == 0.0
    linear = eigenvectors.T @ b
    pole_norm = _norm(linear[pole])
    regular = torch.zeros_like(linear)
    regular[~pole] = -linear[~pole] / gaps[~pole]
    regular_norm = _norm(regular)

    if pole_norm == 0.0 and regular_norm <= 1.0:
        # This is the full hard case, including a nonzero complementary b.
        amplitude = math.sqrt(max(0.0, (1.0 - regular_norm) * (1.0 + regular_norm)))
        vector = eigenvectors @ regular
        vector = vector + amplitude * _canonical_direction(eigenvectors[:, pole])
        return vector, bottom, 0.0, True, 0

    # At a root the pole term alone has norm <= 1, so t >= ||b_pole||.
    # Also ||b/(gap+t)|| <= ||b||/t, giving t <= ||b||. A logarithmic
    # bracket covers every positive float64 pole scale in 64 bisections.
    lower = pole_norm if pole_norm > 0.0 else _MIN_POSITIVE
    upper = _norm(linear)
    if pole_norm > 0.0 and regular_norm < 1.0:
        # The complementary part decreases with t. This tighter upper
        # bracket is useful when a very small pole term fills a hard-case
        # sphere radius of order one.
        radius = math.sqrt((1.0 - regular_norm) * (1.0 + regular_norm))
        upper = min(upper, pole_norm / radius)
    upper = math.nextafter(max(lower, upper), math.inf)
    log_lower, log_upper = math.log(lower), math.log(upper)
    for _ in range(SECULAR_ITERATIONS):
        log_midpoint = 0.5 * (log_lower + log_upper)
        offset = max(_MIN_POSITIVE, math.exp(log_midpoint))
        length = _norm(linear / (gaps + offset))
        if length > 1.0:
            log_lower = log_midpoint
        else:
            log_upper = log_midpoint
    offset = max(_MIN_POSITIVE, math.exp(0.5 * (log_lower + log_upper)))
    coordinates = -linear / (gaps + offset)
    vector = eigenvectors @ coordinates
    return vector, bottom - offset, offset, False, SECULAR_ITERATIONS


def _certificate(
    A: torch.Tensor,
    b: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    scale: float,
    sense: str,
    eigensystem_error: float,
    roundoff: float,
) -> SphereExtremum:
    """Certify a minimum, reversing its signs when requested as a maximum."""
    vector, multiplier, offset, hard_case, iterations = _minimum_candidate(
        eigenvalues, eigenvectors, b
    )
    length = _norm(vector)
    if length == 0.0 or not math.isfinite(length):
        raise FloatingPointError("The secular solver produced a nonfinite or zero vector.")
    vector = vector / length
    length = _norm(vector)
    norm_error = abs(length - 1.0)
    squared_norm = float(vector @ vector)
    value = float(vector @ (A @ vector) + 2.0 * (b @ vector))
    residual = _norm(A @ vector - multiplier * vector + b)
    psd_gap = float(eigenvalues[0]) - multiplier
    psd_lower = psd_gap - eigensystem_error

    # Let M=A-lambda_d I >= mu I, r=Mv+b. The Lagrangian dual is
    # d(lambda_d)=q(v)+lambda_d*(1-||v||^2)-r.T M^{-1} r.
    # Therefore d(lambda_d) >= q(v)+lambda_d*(1-||v||^2)-||r||^2/mu.
    # This remains valid for a singular stationary hard case because its
    # separate dual multiplier moves strictly below the extreme spectrum.
    eigen_lower = float(eigenvalues[0]) - eigensystem_error
    shift = residual + roundoff
    dual_multiplier = min(multiplier, eigen_lower) - shift
    mu = math.nextafter(eigen_lower - dual_multiplier, 0.0)
    if mu <= 0.0:
        raise FloatingPointError("A positive shifted dual PSD margin could not be resolved.")
    dual_residual_upper = residual + abs(multiplier - dual_multiplier) * length + roundoff
    dual_bound = (
        value
        + dual_multiplier * (1.0 - squared_norm)
        - dual_residual_upper * dual_residual_upper / mu
        - roundoff * (1.0 + abs(dual_multiplier))
    )
    gap = max(0.0, value - dual_bound)
    tolerance = max(4096.0 * _EPS * (A.shape[0] + 1), 16.0 * roundoff)
    certificate_ok = (
        norm_error <= tolerance
        and residual <= tolerance
        and psd_lower >= -tolerance
        and 0.0 <= value - dual_bound <= 32.0 * tolerance
    )

    orientation = 1.0 if sense == "minimum" else -1.0
    physical_value = orientation * value * scale
    physical_bound = orientation * dual_bound * scale
    physical_bound = math.nextafter(
        physical_bound, -math.inf if sense == "minimum" else math.inf
    )
    physical_multiplier = orientation * multiplier * scale
    physical_dual_multiplier = orientation * dual_multiplier * scale
    physical_gap = max(0.0, orientation * (physical_value - physical_bound))
    physical_numbers = (
        physical_value, physical_bound, physical_multiplier,
        physical_dual_multiplier, physical_gap, residual * scale,
        psd_gap * scale, psd_lower * scale, eigensystem_error * scale,
        offset * scale,
    )
    if not all(math.isfinite(number) for number in physical_numbers):
        raise FloatingPointError("The extrema or their certificates exceed float64 range.")
    return SphereExtremum(
        sense=sense,
        vector=vector,
        value=physical_value,
        multiplier=physical_multiplier,
        stationarity_residual=residual * scale,
        normalized_stationarity_residual=residual,
        norm_error=norm_error,
        psd_gap=psd_gap * scale,
        normalized_psd_gap=psd_gap,
        psd_lower_bound=psd_lower * scale,
        dual_bound=physical_bound,
        dual_multiplier=physical_dual_multiplier,
        bound_gap=physical_gap,
        normalized_bound_gap=gap,
        certificate_ok=certificate_ok,
        certificate_tolerance=tolerance,
        hard_case=hard_case,
        secular_iterations=iterations,
        scale=scale,
        eigensystem_error=eigensystem_error * scale,
        normalized_eigensystem_error=eigensystem_error,
        secular_pole_offset=offset * scale,
    )


def _zero_extremum(A: torch.Tensor, sense: str) -> SphereExtremum:
    vector = torch.zeros(A.shape[0], dtype=A.dtype, device=A.device)
    vector[0] = 1.0
    return SphereExtremum(
        sense=sense, vector=vector, value=0.0, multiplier=0.0,
        stationarity_residual=0.0, normalized_stationarity_residual=0.0,
        norm_error=0.0, psd_gap=0.0, normalized_psd_gap=0.0,
        psd_lower_bound=0.0, dual_bound=0.0, dual_multiplier=0.0,
        bound_gap=0.0, normalized_bound_gap=0.0, certificate_ok=True,
        certificate_tolerance=4096.0 * _EPS * (A.shape[0] + 1),
        hard_case=True, secular_iterations=0, scale=0.0,
        eigensystem_error=0.0, normalized_eigensystem_error=0.0,
        secular_pole_offset=0.0,
    )


@torch.no_grad()
def sphere_extrema(A: torch.Tensor, b: torch.Tensor) -> SphereExtrema:
    """Return global min/max candidates and numerical certificates.

    A must be a finite, real, symmetric float64 matrix with n >= 1, and b
    a finite float64 vector of length n on the same device. Asymmetry at
    a dimension-scaled rounding level is symmetrized; larger asymmetry
    is rejected. No size threshold turns a small nonzero problem into
    a zero problem. The eigensystem is shared by minimum and maximum.

    Bad input raises ValueError. Unrepresentable scales or extrema and
    failed numerical operations raise FloatingPointError/RuntimeError.
    A returned candidate that misses the documented residual contract
    has certificate_ok=False and must not support an admission decision.
    """
    if not isinstance(A, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise ValueError("A and b must be torch tensors.")
    if A.dtype != torch.float64 or b.dtype != torch.float64:
        raise ValueError("Sphere extrema require torch.float64 A and b.")
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a nonempty square matrix.")
    if b.ndim != 1 or b.shape[0] != A.shape[0] or b.device != A.device:
        raise ValueError("b must match A's dimension and device.")
    if not bool(torch.isfinite(A).all()) or not bool(torch.isfinite(b).all()):
        raise ValueError("A and b must be finite.")

    entry_scale = float(A.abs().max())
    b_norm = _norm(b)
    if not math.isfinite(b_norm):
        raise FloatingPointError("The norm of b exceeds float64 range.")
    if entry_scale == 0.0:
        matrix_unit = torch.zeros_like(A)
        eigenvalues_unit = torch.zeros_like(b)
        eigenvectors = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
        matrix_norm = 0.0
    else:
        matrix_unit = A / entry_scale
        asymmetry = _norm(matrix_unit - matrix_unit.T)
        symmetry_tolerance = 64.0 * _EPS * (A.shape[0] + 1) * _norm(matrix_unit)
        if asymmetry > symmetry_tolerance:
            raise ValueError("A must be symmetric within float64 rounding tolerance.")
        matrix_unit = 0.5 * (matrix_unit + matrix_unit.T)
        eigenvalues_unit, eigenvectors = torch.linalg.eigh(matrix_unit)
        matrix_norm = entry_scale * float(eigenvalues_unit.abs().max())
    scale = max(matrix_norm, b_norm)
    if not math.isfinite(scale):
        raise FloatingPointError("The spectral scale exceeds float64 range.")
    if scale == 0.0:
        return SphereExtrema(_zero_extremum(A, "minimum"), _zero_extremum(A, "maximum"))

    matrix_factor = entry_scale / scale
    matrix = matrix_unit * matrix_factor
    eigenvalues = eigenvalues_unit * matrix_factor
    linear = b / scale
    identity = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    orthogonality_error = _norm(eigenvectors.T @ eigenvectors - identity)
    reconstruction_error = _norm(matrix - (eigenvectors * eigenvalues) @ eigenvectors.T)
    roundoff = 64.0 * _EPS * (A.shape[0] + 1) * max(1.0, _norm(matrix) + _norm(linear))
    eigensystem_error = (
        reconstruction_error
        + float(eigenvalues.abs().max()) * orthogonality_error
        + roundoff
    )
    minimum = _certificate(
        matrix, linear, eigenvalues, eigenvectors, scale,
        "minimum", eigensystem_error, roundoff,
    )
    maximum = _certificate(
        -matrix, -linear, -eigenvalues.flip(0), eigenvectors.flip(1), scale,
        "maximum", eigensystem_error, roundoff,
    )
    return SphereExtrema(minimum, maximum)
