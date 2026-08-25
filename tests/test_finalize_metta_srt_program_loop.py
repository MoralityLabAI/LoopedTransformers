from __future__ import annotations

from rla.finalize_metta_srt_program_loop import _winner


def test_exact_group_winner_handles_ties_and_direction() -> None:
    assert _winner(0.5, 0.25) == "A"
    assert _winner(0.25, 0.5) == "B"
    assert _winner(0.5, 0.5) == "TIE"
