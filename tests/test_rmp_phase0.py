from __future__ import annotations

from research_gym.interp.phase0 import cell_summary, gate_summary, resolve_seed


MANIFEST = {"families": {"pointer_chase": {"validation": {"best_constant": 0.07}}}}


def _result(arm: str, seed: int, *, exact: float, loss: float, passed: bool) -> dict:
    report = {
        "validation": {"pointer_chase": {"exact": exact, "by_depth": {"1": 1.0, "2": exact}}},
        "ood_stress": {"pointer_chase": {"by_depth": {"9": 0.1}}},
    }
    return {
        "cell_id": f"{arm}-base-s{seed}",
        "config": {"arm": arm, "variant": "base", "seed": seed, "families": ["pointer_chase"]},
        "status": "completed" if passed else "budget_exhausted",
        "optimizer_steps": 1000,
        "examples_seen": 64000,
        "events": {"C3": 500} if passed else {},
        "gate_passed": passed,
        "last_interval_train_loss": {"pointer_chase": loss},
        "target_entropy_nats": {"pointer_chase": 2.77},
        "c3": report if passed else None,
        "final": {
            "validation": {"pointer_chase": {"exact": 0.5, "by_depth": {"1": 0.5, "2": 0.5}}},
            "ood_stress": {"pointer_chase": {"by_depth": {"9": 0.0}}},
        },
    }


def test_cell_summary_reports_the_gate_checkpoint_and_learning_checks() -> None:
    passed = cell_summary(_result("LOOP-T", 1, exact=0.99, loss=0.05, passed=True), MANIFEST)
    failed = cell_summary(_result("LOOP-T", 2, exact=0.99, loss=2.90, passed=False), MANIFEST)

    assert passed["gate_passed"] and passed["learned"] and passed["c3_step"] == 500
    assert passed["families"]["pointer_chase"]["validation_exact"] == 0.99
    assert passed["families"]["pointer_chase"]["ood_by_depth"] == {"9": 0.1}
    assert failed["families"]["pointer_chase"]["validation_exact"] == 0.5
    assert not failed["families"]["pointer_chase"]["loss_below_entropy"]
    assert not failed["learned"] and failed["c3_step"] is None


def _staged(status: str, *, passed: bool = False) -> dict:
    result = _result("LOOP-T", 1, exact=0.99, loss=0.05, passed=passed)
    return {**result, "status": status}


def test_resolve_seed_walks_the_stages_in_ladder_order() -> None:
    exhausted = _staged("budget_exhausted")
    passed = _staged("gate_passed_c4_pending", passed=True)
    paused = _staged("paused")

    cases = {
        "not_started": {},
        "in_progress": {"": paused},
        "passed": {"": exhausted, "-ds": passed},
        "failed": {"": exhausted, "-ds": exhausted, "-ds-inj": exhausted},
    }
    for expected, stages in cases.items():
        assert resolve_seed("LOOP-T", 1, stages, MANIFEST)["verdict"] == expected
    waiting = resolve_seed("LOOP-T", 1, {"": exhausted, "-ds": exhausted}, MANIFEST)
    assert waiting["verdict"] == "in_progress"
    first = resolve_seed("LOOP-T", 1, {"": passed, "-ds": exhausted}, MANIFEST)
    assert first["verdict"] == "passed" and [s["stage"] for s in first["stages"]] == ["plain"]
    assert resolve_seed("LOOP-T", 1, cases["passed"], MANIFEST)["deciding"]["stage"] == "-ds"


def test_gate_needs_three_passing_seeds_in_every_arm() -> None:
    cells = [
        cell_summary(_result(arm, seed, exact=0.99, loss=0.05, passed=ok), MANIFEST)
        for arm, seed, ok in (
            ("LOOP-T", 1, True),
            ("LOOP-T", 2, True),
            ("LOOP-T", 3, True),
            ("FF-U", 1, True),
            ("FF-U", 2, True),
            ("FF-U", 3, False),
        )
    ]
    summary = gate_summary(cells)

    assert summary["arms"]["LOOP-T"]["g0_passed"]
    assert summary["arms"]["FF-U"]["passing_seeds"] == [1, 2]
    assert not summary["arms"]["FF-U"]["g0_passed"]
    assert not summary["g0_passed"]
    assert not gate_summary([])["g0_passed"]
