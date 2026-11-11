from __future__ import annotations

import pytest

# `rla` is not part of this repo or any declared dependency; these tests were
# recovered with the baseline and only run where that package is installed.
pytest.importorskip("rla")

from rla.metta_srt_long_trace_distillation import pair_macro_expression


def test_pair_macro_expression_preserves_arbitrary_even_order() -> None:
    assert pair_macro_expression("SEED", (0, 1, 3, 2, 1, 0), group_index=7) == (
        "(mac-007-1-0 (mac-007-3-2 (mac-007-0-1 SEED)))"
    )
    with pytest.raises(ValueError):
        pair_macro_expression("SEED", (0, 1, 2), group_index=7)
