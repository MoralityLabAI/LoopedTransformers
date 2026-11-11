from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from research_gym.interp.analysis import AnalysisModel
from research_gym.interp.bundle import BundleConfig, iter_family_examples
from research_gym.interp.models import build_model
from research_gym.interp.probes import (
    answer_states,
    control_labels,
    mlp_sites,
    online_code_ratio,
    probe_sites,
    shuffled_within,
    trace_labels,
)
from research_gym.interp.training import CellConfig


ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cpu")


def _synthetic(count: int, seed: int) -> tuple[torch.Tensor, np.ndarray]:
    """Two sites: the label is linear in site 1 and absent from site 0."""

    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 4, (count,), generator=generator)
    noise = torch.randn(count, 2, 8, generator=generator)
    noise[:, 1, :4] += 3.0 * torch.nn.functional.one_hot(labels, 4).float()
    return noise, labels.numpy()


def test_linear_probe_finds_the_site_that_holds_the_label_and_transfers_nowhere_else() -> None:
    train, y_train = _synthetic(600, 0)
    select, y_select = _synthetic(200, 1)
    report, y_report = _synthetic(200, 2)
    y_report[:10] = -1  # rows without the variable are skipped
    result = probe_sites(train, y_train, select, y_select, report, y_report, device=DEVICE, transfer=True)

    assert result["accuracy"][1] > 0.95 and result["accuracy"][0] < 0.4
    assert result["transfer"].shape == (2, 2) and result["transfer"][1, 0] < 0.4
    assert mlp_sites(train, y_train, report, y_report, device=DEVICE)[1] > 0.9
    ratio = online_code_ratio(train, y_train, device=DEVICE)
    assert ratio[1] < 0.5 < ratio[0]


def test_baseline_labels_keep_their_distribution() -> None:
    labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    strata = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    shuffled = shuffled_within(labels, strata, seed=0)
    for value in (1, 2):
        assert sorted(shuffled[strata == value]) == sorted(labels[strata == value])
    control = control_labels(np.array([3, 3, 5, 0]), classes=4, seed=0)
    assert control[0] == control[1] and control.max() < 4


def test_trace_labels_and_answer_states_on_bundle_rows() -> None:
    raw = json.loads((ROOT / "configs" / "rmp_task_bundle_v0.json").read_text(encoding="utf-8"))
    families = ["pointer_chase", "gated_pointer_chase", "reachability", "sudoku4"]
    counts = {"validation": 16, "monitor_train": 8, "monitor_calibration": 8, "locked": 8, "ood_stress": 8}
    config = BundleConfig.from_mapping({**raw, "families": families, "counts": counts})
    for family in families:
        rows = [row for row in iter_family_examples(family, config) if row.sibling is None][:16]
        labels = trace_labels(family, rows)
        for steps in labels.values():
            for step, values in steps.items():
                assert values.shape == (len(rows),)
                # A row has a label at a step exactly when its trace is long enough.
                assert ((values >= 0) == np.array([step < len(next(iter(row.trace.values()))) for row in rows])).all()

    torch.manual_seed(0)
    model = build_model("LOOP-T", hidden_size=16, num_heads=2, visits=3, beta=1.0).eval()
    analysis = AnalysisModel("tiny", "tiny", "C0", 0, CellConfig(arm="LOOP-T", seed=0, families=(), visits=3), model)
    tokens = torch.tensor([row.tokens for row in rows])
    states, pooled = answer_states(analysis, tokens, batch=5)
    assert states.shape == (len(rows), 4, 16) and pooled.shape == (len(rows), 16)
    output = model(tokens)
    torch.testing.assert_close(states[:, -1], output.visit_outputs[-1][:, -1])
