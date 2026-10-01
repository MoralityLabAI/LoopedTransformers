"""Gate G0 summary over completed RMP training cells."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


MINIMUM_SEEDS = 3


def cell_summary(result: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Per-family learning checks and gate verdict for one cell.

    The reported accuracy is the C3 checkpoint's when the gate passed, otherwise the last model's.
    """

    report = result["c3"] if result["gate_passed"] else result["final"]
    families = {}
    for name in result["config"]["families"]:
        shard = manifest["families"][name]["validation"]
        loss = (result["last_interval_train_loss"] or {}).get(name)
        entropy = result["target_entropy_nats"][name]
        validation = report["validation"][name] if report else None
        families[name] = {
            "train_loss": loss,
            "target_entropy_nats": entropy,
            "loss_below_entropy": loss is not None and loss < entropy,
            "best_constant": shard["best_constant"],
            "validation_exact": validation["exact"] if validation else None,
            "above_best_constant": bool(validation) and validation["exact"] > shard["best_constant"],
            "validation_by_depth": validation["by_depth"] if validation else None,
            "ood_by_depth": (report["ood_stress"].get(name) or {}).get("by_depth") if report else None,
        }
    return {
        "cell_id": result["cell_id"],
        "arm": result["config"]["arm"],
        "variant": result["config"]["variant"],
        "seed": result["config"]["seed"],
        "status": result["status"],
        "optimizer_steps": result["optimizer_steps"],
        "examples_seen": result["examples_seen"],
        "c3_step": result["events"].get("C3"),
        "gate_passed": bool(result["gate_passed"]),
        "learned": all(
            family["loss_below_entropy"] and family["above_best_constant"] for family in families.values()
        ),
        "families": families,
    }


def gate_summary(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """G0 holds for an arm when at least three of its seeds pass the gate on every family."""

    arms: dict[str, dict[str, Any]] = {}
    for cell in cells:
        arm = arms.setdefault(cell["arm"], {"seeds": [], "passing_seeds": []})
        arm["seeds"].append(cell["seed"])
        if cell["gate_passed"]:
            arm["passing_seeds"].append(cell["seed"])
    for arm in arms.values():
        arm["seeds"].sort()
        arm["passing_seeds"].sort()
        arm["g0_passed"] = len(arm["passing_seeds"]) >= MINIMUM_SEEDS
    return {"arms": arms, "g0_passed": bool(arms) and all(arm["g0_passed"] for arm in arms.values())}
