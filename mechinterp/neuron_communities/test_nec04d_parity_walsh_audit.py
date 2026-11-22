"""Source-only zero-training NEC-04D fifth-order Walsh spectral checks."""
import math

import pytest

from mechinterp.neuron_communities.nec04d_parity_walsh_audit import (
    ROOTS, ARMS, CHECKPOINTS, context_spectrum, walsh_coefficients,
)


def input_rows(c=0):
    return [{
        "focal_mask": mask,
        "context_group": c,
        "mux_selector": 1,
        "composed_label": mask.bit_count() % 2,
    } for mask in range(32)]


def binary_logits(margins):
    return [[0.0, float(v)] for v in margins]


def test_existing_native_roster_is_exactly_fixed():
    assert ROOTS == (6, 7, 8, 9, 10, 11)
    assert ARMS == ("vanilla", "parity_curriculum")
    assert CHECKPOINTS == (240, 900)


def test_perfect_five_focal_parity_is_pure_degree_five():
    truth = [(mask.bit_count() % 2) for mask in range(32)]
    margin = [2 * y - 1 for y in truth]
    score = context_spectrum(input_rows(), binary_logits(margin))
    assert score["cases"] == 32
    assert score["correct"] == 32
    assert score["truth_positive"] == 16
    assert score["predicted_majority_fraction"] == 0.5
    assert score["parity_margin_cosine_to_oracle"] == pytest.approx(1.0)
    assert score["degree5_fraction_of_centered_walsh_power"] == pytest.approx(1.0)
    assert score["walsh_power_by_degree"][:4] == [0., 0., 0., 0.]
    assert score["walsh_power_by_degree"][-1] == pytest.approx(1.0)


def test_low_degree_signal_has_zero_five_focal_parity_energy():
    # A first-order input bit is orthogonal to the five-way XOR teacher.
    margins = [1.0 if mask & 1 else -1.0 for mask in range(32)]
    score = context_spectrum(input_rows(), binary_logits(margins))
    assert score["correct"] == 16
    assert score["parity_margin_cosine_to_oracle"] == pytest.approx(0.0)
    assert score["degree5_fraction_of_centered_walsh_power"] == pytest.approx(0.0)
    assert score["walsh_power_by_degree"][0] == pytest.approx(1.0)
    assert score["walsh_power_by_degree"][-1] == pytest.approx(0.0)


def test_constant_logits_have_zero_variance_and_undefined_fraction():
    score = context_spectrum(input_rows(), binary_logits([3.0] * 32))
    assert score["correct"] == 16
    assert score["zero_variance_context"]
    assert score["degree5_fraction_of_centered_walsh_power"] is None
    assert score["predicted_majority_fraction"] == 1.0
    assert score["oracle_aligned_parity_margin"] == pytest.approx(0.0)


def test_logit_magnitudes_scale_but_spectral_fractions_do_not():
    parity = [2 * (mask.bit_count() % 2) - 1 for mask in range(32)]
    marginal = [parity[mask] + (0.3 if mask & 1 else -0.3) + 0.2
                for mask in range(32)]
    a = context_spectrum(input_rows(), binary_logits(marginal))
    b = context_spectrum(input_rows(), binary_logits([2.5 * v for v in marginal]))
    assert a["degree5_fraction_of_centered_walsh_power"] == pytest.approx(
        b["degree5_fraction_of_centered_walsh_power"]
    )
    assert a["parity_margin_cosine_to_oracle"] == pytest.approx(
        b["parity_margin_cosine_to_oracle"]
    )
    assert b["oracle_aligned_parity_margin"] == pytest.approx(
        2.5 * a["oracle_aligned_parity_margin"]
    )


def test_context_label_sign_change_reverses_alignment():
    margins = [1.0 if (mask.bit_count() % 2) else -1.0 for mask in range(32)]
    rows = input_rows()
    original = context_spectrum(rows, binary_logits(margins))
    rows_flipped = [{**r, "composed_label": 1 - r["composed_label"]} for r in rows]
    flipped = context_spectrum(rows_flipped, binary_logits(margins))
    assert original["oracle_aligned_parity_margin"] == pytest.approx(
        -flipped["oracle_aligned_parity_margin"]
    )
    assert original["degree5_fraction_of_centered_walsh_power"] == pytest.approx(
        flipped["degree5_fraction_of_centered_walsh_power"]
    )


def test_rejects_misordered_masks_and_nonfinite_logits():
    inputs = input_rows()
    with pytest.raises(AssertionError, match="mask ordering"):
        context_spectrum(inputs[::-1], binary_logits([1.0] * 32))
    with pytest.raises(AssertionError, match="nonfinite"):
        walsh_coefficients([float("nan")] + [0.0] * 31)
    with pytest.raises(AssertionError, match="exactly 32"):
        walsh_coefficients([0.] * 31)
