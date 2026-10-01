from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from research_gym.interp.bundle import BundleConfig, materialize_bundle, train_example  # noqa: E402
from research_gym.interp.models import build_model  # noqa: E402
from research_gym.interp.training import (  # noqa: E402
    COMPLETED,
    EXHAUSTED,
    PAUSED,
    CellConfig,
    cell_loss,
    draw_indices,
    load_pool,
    pool_from_rows,
    pool_rows,
    run_cell,
    run_visits,
    save_pool,
)


ROOT = Path(__file__).resolve().parents[1]
FAMILY = "pointer_chase"
CELL = CellConfig(
    arm="LOOP-T",
    seed=7,
    families=(FAMILY,),
    hidden_size=32,
    visits=4,
    batch_per_family=16,
    warmup_steps=4,
    max_steps=40,
    eval_every=8,
    eval_per_depth=4,
)


def _bundle_config() -> BundleConfig:
    raw = json.loads((ROOT / "configs" / "rmp_task_bundle_v0.json").read_text(encoding="utf-8"))
    counts = {"validation": 32, "monitor_train": 8, "monitor_calibration": 8, "locked": 8, "ood_stress": 16}
    return BundleConfig.from_mapping({**raw, "families": [FAMILY], "counts": counts})


def _setup(tmp_path: Path):
    config = _bundle_config()
    bundle_dir = tmp_path / "datasets"
    manifest = materialize_bundle(config, bundle_dir, write_splits=["validation", "ood_stress"])
    pools = {FAMILY: pool_from_rows(pool_rows(FAMILY, 0, 128, config))}
    return pools, {
        "bundle_dir": bundle_dir,
        "result_dir": tmp_path / "runs",
        "artifact_dir": tmp_path / "artifacts",
        "provenance": {"bundle_hash": manifest["bundle_hash"]},
        "device": torch.device("cpu"),
    }


def test_pool_is_the_training_stream_and_is_hash_checked(tmp_path: Path) -> None:
    config = _bundle_config()
    pool = pool_from_rows(pool_rows(FAMILY, 0, 24, config))
    receipt = save_pool(pool, tmp_path / "pool.pt")
    loaded = load_pool(tmp_path / "pool.pt", receipt["sha256"])

    for index in (0, 11, 23):
        example = train_example(FAMILY, index, config)
        assert loaded.tokens[index].tolist() == list(example.tokens)
        assert int(loaded.targets[index]) == example.target_token
        assert int(loaded.depths[index]) == example.difficulty
    assert 0.0 < loaded.target_entropy() <= torch.log(torch.tensor(16.0)).item() + 1e-6
    with pytest.raises(ValueError, match="hash mismatch"):
        load_pool(tmp_path / "pool.pt", "0" * 64)


def test_cell_keeps_named_hash_attested_checkpoints_and_reports_both_splits(tmp_path: Path) -> None:
    pools, paths = _setup(tmp_path)
    result = run_cell(CELL, pools, **paths)
    stored = json.loads((paths["result_dir"] / CELL.cell_id / "result.json").read_text(encoding="utf-8"))

    assert json.loads(json.dumps(result)) == stored
    assert result["status"] == EXHAUSTED and not result["gate_passed"]
    assert result["optimizer_steps"] == 40 and result["examples_seen"] == 640
    assert [entry["label"] for entry in result["checkpoints"]] == ["C0", "step_16", "step_32"]
    for entry in result["checkpoints"]:
        path = paths["artifact_dir"] / CELL.cell_id / "checkpoints" / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert torch.load(path, weights_only=False)["step"] == entry["step"]
    assert [record["step"] for record in result["history"]] == [0, 8, 16, 24, 32, 40]
    assert result["history"][0]["train_loss"][FAMILY] is None
    assert result["last_interval_train_loss"][FAMILY] > 0.0
    final = result["final"]
    assert set(final["validation"][FAMILY]["by_depth"]) == {str(depth) for depth in range(1, 9)}
    assert set(final["ood_stress"][FAMILY]["by_depth"]) == {"9", "10", "11", "12"}
    assert final["validation"][FAMILY]["examples"] == 32


def test_paused_and_resumed_cell_matches_an_uninterrupted_one(tmp_path: Path) -> None:
    config = replace(CELL, max_steps=24)
    pools, paths = _setup(tmp_path / "whole")
    whole = run_cell(config, pools, **paths)
    pools, paths = _setup(tmp_path / "pieces")
    statuses = []
    while True:
        pieces = run_cell(config, pools, max_new_steps=10, **paths)
        statuses.append(pieces["status"])
        if pieces["status"] != PAUSED:
            break

    assert statuses == [PAUSED, PAUSED, EXHAUSTED]
    assert pieces["resumed"] and not whole["resumed"]
    assert pieces["history"] == whole["history"]
    assert pieces["final"] == whole["final"]
    assert [e["sha256"] for e in pieces["checkpoints"]] == [e["sha256"] for e in whole["checkpoints"]]


def test_budget_extension_resumes_the_same_cell_and_other_changes_are_refused(tmp_path: Path) -> None:
    pools, paths = _setup(tmp_path)
    first = run_cell(replace(CELL, max_steps=16), pools, **paths)
    extended = run_cell(replace(CELL, max_steps=24), pools, **paths)
    events = (paths["result_dir"] / CELL.cell_id / "events.jsonl").read_text(encoding="utf-8")

    assert first["status"] == extended["status"] == EXHAUSTED
    assert first["cell_hash"] == extended["cell_hash"]
    assert extended["resumed"] and extended["optimizer_steps"] == 24
    assert '"event":"budget_extended"' in events
    with pytest.raises(ValueError, match="another cell"):
        run_cell(replace(CELL, learning_rate=3e-4), pools, **paths)


def test_passing_the_gate_records_c3_then_c4(tmp_path: Path) -> None:
    pools, paths = _setup(tmp_path)
    result = run_cell(replace(CELL, gate=0.0), pools, **paths)

    assert result["status"] == COMPLETED and result["gate_passed"]
    assert result["events"]["C3"] == 0
    assert {"C0", "C3", "C4"} <= {entry["label"] for entry in result["checkpoints"]}
    assert result["c3"]["validation"][FAMILY]["examples"] == 32


def test_microbatches_accumulate_to_the_same_update_as_one_pass(tmp_path: Path) -> None:
    pools, paths = _setup(tmp_path)
    whole = run_cell(replace(CELL, variant="whole", max_steps=8), pools, **paths)
    split = run_cell(replace(CELL, variant="split", max_steps=8, microbatch=4), pools, **paths)

    assert split["history"][-1]["train_loss"][FAMILY] == pytest.approx(
        whole["history"][-1]["train_loss"][FAMILY], rel=1e-4
    )
    assert split["final"]["validation"][FAMILY]["exact"] == whole["final"]["validation"][FAMILY]["exact"]


def test_batches_are_drawn_uniformly_over_unlocked_depths_only() -> None:
    by_depth = {0: torch.arange(0, 10), 1: torch.arange(10, 20), 2: torch.arange(20, 30)}
    generator = torch.Generator().manual_seed(3)
    shallow = draw_indices(by_depth, 1, 400, generator)
    everything = draw_indices(by_depth, 2, 600, generator)

    assert shallow.shape == (400,) and int(shallow.max()) < 20
    assert 150 < int((shallow < 10).sum()) < 250
    assert all(150 < int(((everything >= low) & (everything < low + 10)).sum()) < 250 for low in (0, 10, 20))

    weighted = draw_indices(by_depth, 2, 400, generator, frontier_share=0.5)
    assert int((weighted >= 20).sum()) == 200
    assert 70 < int((weighted < 10).sum()) < 130
    assert int(draw_indices(by_depth, 0, 40, generator, frontier_share=0.5).max()) < 10


def test_curriculum_starts_shallow_and_unlocks_on_interim_accuracy(tmp_path: Path) -> None:
    pools, paths = _setup(tmp_path)
    staircase = pool_from_rows(pool_rows(FAMILY, 0, 32, _bundle_config(), staircase=True))
    pools = {FAMILY: pools[FAMILY].merged(staircase)}
    assert set(staircase.depths.tolist()) == {0}

    locked = run_cell(replace(CELL, variant="locked", curriculum=True, max_steps=8), pools, **paths)
    opened = run_cell(
        replace(CELL, variant="opened", curriculum=True, unlock_accuracy=0.0, max_steps=24), pools, **paths
    )
    events = (paths["result_dir"] / "LOOP-T-opened-s7" / "events.jsonl").read_text(encoding="utf-8")

    assert locked["deepest_unlocked"] == {FAMILY: 1}
    assert opened["deepest_unlocked"] == {FAMILY: 5}
    assert events.count('"event":"unlock"') == 4
    assert run_cell(replace(CELL, variant="flat", max_steps=8), pools, **paths)["deepest_unlocked"] == {FAMILY: 8}


def test_variants_change_the_loss_as_specified() -> None:
    torch.manual_seed(5)
    model = build_model("LOOP-T", hidden_size=32, visits=4, vocab_size=64)
    tokens = torch.randint(0, 64, (6, 10))
    targets = torch.randint(0, 64, (6,))
    plain = replace(CELL, families=("x",))
    output = model(tokens)

    per_visit = torch.stack(
        [
            torch.nn.functional.cross_entropy(model.lens(state[:, -1]), targets, reduction="none")
            for state in output.visit_outputs
        ]
    )
    assert torch.allclose(cell_loss(model, tokens, targets, plain), per_visit[-1])
    assert torch.allclose(
        cell_loss(model, tokens, targets, replace(plain, deep_supervision=True)), per_visit.mean(dim=0)
    )
    injected = model.embed(tokens)
    state = injected
    for visit in range(4):
        state = model.block_at(visit)(state) + injected
    assert torch.allclose(
        run_visits(model, tokens, replace(plain, input_injection=True)).logits, model.lens(state), atol=1e-6
    )
