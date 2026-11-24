"""Verify direct carry capture through real random Nanbeige CPU decoder layers."""
import argparse
from importlib.metadata import version
from itertools import permutations
import json
import os
from pathlib import Path

from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import bound, digest, write_new

from .account import TERMS, account, row, shapley
from .capture import CarryCapture, SITES, check_additions, same_bf16

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PARENT = REPO / "experiments/nan_readpath37/registration.json"
BUDGET = {"minimum_start_available_RAM_bytes": 5 * 2**30, "minimum_available_RAM_bytes": 4 * 2**30,
          "maximum_private_bytes": 768 * 2**20, "maximum_seconds": 220}
PACKAGES = {"torch": "2.6.0+cu124", "transformers": "4.46.3", "numpy": "2.5.5"}
CONFIGURATION = "98abb9faa303717b234b3d381a9650e8cb656ccf287a12f6ad6215a3bbf8be10"
BASE_SITES = ("inputs", "native_writes", "residual")
SEED = 64017


def seal():
    paths = [HERE / name for name in (".gitattributes", "__init__.py", "account.py", "capture.py",
                                     "verify.py", "PROTOCOL.md")]
    paths += [PARENT] + [REPO / name for name in (
        "experiments/nan_phase35/run.py", "experiments/nan_ram28/resources.py",
        "experiments/nan_ranksite13/resources.py", "experiments/nan_readpath37/run.py",
        "experiments/nan_recomp09/common.py", "experiments/nan_carry63/algebra.py",
        "experiments/nan_lattice53/run.py", "experiments/nan_dose51/native.py",
        "experiments/nan_gated46/gated.py", "experiments/nan_delay39/instrument.py",
        "mechinterp/nanbeige_bpd_gym/chunked_cache.py",
        "mechinterp/nanbeige_bpd_gym/streamed_vocab_head.py",
        "mechinterp/dark_fiber_gyms/arg03_nanbeige.py")]
    return {p.relative_to(REPO).as_posix(): digest(p) for p in paths}


class FixtureHead:
    """Replay coalition states through the same tiny CPU norm and streamed head."""

    def __init__(self, norm, head):
        self.norm, self.head = norm, head
        self.gamma = norm.weight.detach().float().numpy()
        self.eps = norm.variance_epsilon

    def row(self, token):
        return self.head.weight[token].detach().float().numpy()

    def logits(self, states, tokens):
        import torch

        with torch.inference_mode():
            hidden = torch.as_tensor(states, dtype=torch.bfloat16)
            return self.head(self.norm(hidden))[:, tokens].numpy()


def verify_shapley():
    from fractions import Fraction
    from math import factorial

    n = len(TERMS)
    scores = [3 * bool(mask & 1) + 2 * bool(mask & 2) - 7 * bool(mask & 16)
              + 8 * ((mask & 9) == 9) for mask in range(2**n)]
    reference = [Fraction(0)] * n
    for order in permutations(range(n)):
        active = 0
        for i in order:
            new = active | (1 << i)
            reference[i] += Fraction(scores[new] - scores[active], factorial(n))
            active = new
    assert list(shapley(scores).values()) == reference == [7, 2, 0, 4, -7]
    return factorial(n)


def hook_count(loop):
    layers = list(loop.layers)
    if loop.stage_layer is not None:
        layers.append(loop.stage_layer)
    return sum(len(module._forward_hooks) + len(module._forward_pre_hooks)
               for layer in layers for module in layer.modules())


def check_reference(reference, enhanced, torch):
    assert torch.equal(reference["logits"], enhanced["logits"])
    assert reference["semantic_lengths"] == enhanced["semantic_lengths"]
    assert reference["physical_length"] == enhanced["physical_length"]
    assert reference["applied"] == enhanced["applied"]
    assert all(same_bf16(reference[site][key], enhanced[site][key])
               for site in BASE_SITES for key in reference[site])


def fixture(snapshot, resident, check, np, torch):
    from experiments.nan_delay39.instrument import clean_row_closes, prepare
    from experiments.nan_dose51.native import RouteCapture
    from experiments.nan_gated46.gated import preservation
    from experiments.nan_recomp09.common import target_digest
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import build_nanbeige_model_and_loop
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    check()
    torch.manual_seed(SEED)
    model, loop = build_nanbeige_model_and_loop(
        snapshot, exec_device=torch.device("cpu"), dtype=torch.bfloat16,
        n_gpu_layers=resident, random_init=True,
        config_overrides={"vocab_size": 97, "num_hidden_layers": 3, "hidden_size": 64,
                          "intermediate_size": 96, "num_attention_heads": 4, "num_key_value_heads": 2,
                          "head_dim": 16, "kv_channels": 16})
    model.requires_grad_(False)
    if loop.stage_layer is not None:
        loop.stage_layer.requires_grad_(False)
    placement = "staged" if loop.stage_layer is not None else "resident"
    head = StreamedVocabularyHead(model.lm_head.weight, loop.device, 13)
    head_view = FixtureHead(loop.base.norm, head)
    before = target_digest(loop)
    hooks = hook_count(loop)
    prompt = [1, 4, 32, 5, 29, 6, 42]
    plans = ([prompt, [8, 39, 77]], [prompt, prompt])
    tokens = (2, 1)
    cfg = {"maximum_head_delta": 0.25, "maximum_margin_delta": 0.25}
    original = RouteCapture(loop, head, layer=2, chunk=3)
    try:
        expected = [original.capture(prompts) for prompts in plans]
        starting = expected[-1]["residual"][1, 1].float().numpy().astype(np.float64)
        direction = np.random.default_rng(SEED).normal(size=starting.shape)
        patch = prepare(starting, direction, .025, .05, .9, np)
        assert patch.eligible
        planned = original.planned_head(expected[-1], patch)
        baseline = expected[-1]["logits"][0]
        assert preservation(baseline, planned, tokens, cfg)["passed"]
        gate = {"planned": planned, "baseline": baseline, "tokens": tokens, "cfg": cfg}
        expected_edit = original.capture(plans[-1], patch, **gate)
    finally:
        original.close()
    assert hook_count(loop) == hooks

    observed = CarryCapture(loop, head, layer=2, chunk=3)
    try:
        twins, clean = [observed.capture(prompts) for prompts in plans]
        check_reference(expected[0], twins, torch)
        check_reference(expected[1], clean, torch)
        edited = observed.capture(plans[-1], patch, **gate)
        check_reference(expected_edit, edited, torch)
        clean_row_closes(clean, edited, expect_edit=True)
        assert all(same_bf16(clean[k][phase, 0], edited[k][phase, 0]) for k in SITES for phase in (1, 2))
        assert all(same_bf16(clean[k][1, 1], edited[k][1, 1]) for k in (*SITES, *BASE_SITES))
        assert not same_bf16(clean["pre_mlp"][2, 1], edited["pre_mlp"][2, 1])
        assert all(same_bf16(clean[k][phase, 0], clean[k][phase, 1]) for k in SITES for phase in (1, 2))

        affected = {**twins, "pre_mlp": dict(twins["pre_mlp"])}
        value = affected["pre_mlp"][1, 1]
        affected["pre_mlp"][1, 1] = value + torch.ones_like(value)
        try:
            check_additions(affected)
        except ValueError as exc:
            assert "native attention add" in str(exc)
        else:
            raise AssertionError("A corrupted carry operand passed")
        assert observed.cache.stage_materializations == (4 if placement == "staged" else 0)

        accounts = []
        for phase in (1, 2):
            account0, account1 = row(twins, phase, 0), row(twins, phase, 1)
            value = account(account0, account1, head_view, tokens, np)
            # A patch affects the phase-1 reached head after native capture.
            # The unedited pair's endpoint is authoritative for this replay comparison.
            logit = twins["logits"][phase - 1, :, list(tokens)].double().numpy()
            native_delta = float(logit[1, 0] - logit[1, 1] - logit[0, 0] + logit[0, 1])
            assert value["native_swing"] == native_delta
            assert abs(sum(value["finite"].values()) - native_delta) < 1e-12
            accounts.append(value)
        delayed = account(row(clean, 2, 1), row(edited, 2, 1), head_view, tokens, np)
        logits = edited["logits"][1, 1].double().numpy() - clean["logits"][1, 1].double().numpy()
        assert delayed["native_swing"] == float(logits[tokens[0]] - logits[tokens[1]])
        accounts.append(delayed)
        assert target_digest(loop) == before
        check()
        return {
            "placement": placement, "seed": SEED, "native_layers": loop.n, "matched_shapes": [list(map(len, p)) for p in plans],
            "unchanged_original_head_comparisons": 3, "last_token_additions_checked": 3 * 2 * 2 * 2,
            "edited_phase2_pre_mlp_changed": True, "tampered_carry_refused": True,
            "head_vocabulary_size": int(head.weight.shape[0]), "native_head_blocks": head.last_chunks,
            "independent_head_games_with_exact_replayed_swing": len(accounts),
            "max_attention_rounding_delta_l2": max(v["attention_rounding_delta_l2"] for v in accounts),
            "max_mlp_rounding_delta_l2": max(v["mlp_rounding_delta_l2"] for v in accounts),
        }
    finally:
        observed.close()
        head.close()
        assert hook_count(loop) == hooks


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--snapshot", required=True, type=Path)
    cli.add_argument("--work", required=True, type=Path)
    args = cli.parse_args()
    if not __debug__:
        cli.error("Enable Python assertions")
    manifest = seal()
    base = json.loads(PARENT.read_bytes())
    if args.snapshot.name != base["revision"]:
        cli.error("Use the pinned Nanbeige snapshot")
    with lease(BUDGET) as (check, samples):
        args.work.mkdir(parents=True, exist_ok=False)
        os.environ.update(USE_TORCH="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          HF_HOME=str((args.work / "hf_home").resolve()),
                          HF_MODULES_CACHE=str((args.work / "hf_modules").resolve()))
        os.environ.pop("TRANSFORMERS_CACHE", None)
        for key in ("config.json", "modeling_nanbeige.py"):
            bound(args.snapshot / key, base["snapshot_pins"][key])
        bound(args.snapshot / "configuration_nanbeige.py", CONFIGURATION)
        assert {p: version(p) for p in PACKAGES} == PACKAGES
        import numpy as np
        import torch

        torch.set_num_threads(1)
        assert torch.get_num_threads() == 1
        permutations_checked = verify_shapley()
        rows = [fixture(args.snapshot, resident, check, np, torch) for resident in (3, 1)]
        assert not torch.cuda.is_initialized()
    if seal() != manifest:
        raise ValueError("Verified source changed during the fixture")
    result = {
        "experiment": "NAN-PREMLP-64", "status": "passed", "source_sha256": manifest,
        "revision": base["revision"], "snapshot_code_sha256": {
            **{name: base["snapshot_pins"][name] for name in ("config.json", "modeling_nanbeige.py")},
            "configuration_nanbeige.py": CONFIGURATION},
        "resources": {**BUDGET, "maximum_sampled_private_bytes": max(s["private_bytes"] for s in samples),
                      "minimum_sampled_available_ram_bytes": min(s["available_physical_bytes"] for s in samples)},
        "fixtures": rows, "synthetic_shapley_orders_checked": permutations_checked,
        "cpu_threads": 1, "pretrained_forwards": 0, "formal_comparisons": 0,
        "cuda_initialized": False, "sealed_test_accessed": False,
    }
    write_new(args.work / "verification.json", result)
    print(result["status"], [(r["placement"], r["last_token_additions_checked"]) for r in rows], flush=True)


if __name__ == "__main__":
    main()
