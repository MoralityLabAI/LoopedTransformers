"""Cross-check the BF16 inverse with directly observed random Nanbeige operands."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path

from experiments.nan_phase35.run import lease
from experiments.nan_premlp64.capture import CarryCapture
from experiments.nan_premlp64.verify import BUDGET, CONFIGURATION, PACKAGES
from experiments.nan_readpath37.run import bound, digest, write_new

from .lattice import invert, lattice

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "nan_readpath37/registration.json"


def fixture(snapshot, check, np, torch):
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import build_nanbeige_model_and_loop
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    torch.manual_seed(65102)
    model, loop = build_nanbeige_model_and_loop(
        snapshot, exec_device=torch.device("cpu"), dtype=torch.bfloat16,
        n_gpu_layers=1, random_init=True,
        config_overrides={"vocab_size": 97, "num_hidden_layers": 3, "hidden_size": 64,
                          "intermediate_size": 96, "num_attention_heads": 4, "num_key_value_heads": 2,
                          "head_dim": 16, "kv_channels": 16})
    model.requires_grad_(False)
    loop.stage_layer.requires_grad_(False)
    head = StreamedVocabularyHead(model.lm_head.weight, loop.device, 13)
    capture = CarryCapture(loop, head, layer=2, chunk=3)
    try:
        packet = capture.capture([[1, 4, 32, 5, 29, 6, 42], [8, 39, 77]])
    finally:
        capture.close()
        head.close()

    checked = 0
    uniques = [0, 0]
    grid = lattice(np)
    for key in sorted(packet["residual"]):
        check()
        output, attn, write, incoming, carry = (packet[site][key].float().numpy() for site in
            ("residual", "attention_write", "native_writes", "attention_input", "pre_mlp"))
        lo, hi, carry_lo, carry_hi = invert(grid, output, attn, write, np)
        assert (lo <= hi).all()
        assert ((grid[lo] <= incoming) & (incoming <= grid[hi])).all()
        assert ((carry_lo <= carry) & (carry <= carry_hi)).all()
        assert np.array_equal(grid[lo][lo == hi], incoming[lo == hi])
        assert np.array_equal(carry_lo[carry_lo == carry_hi], carry[carry_lo == carry_hi])
        checked += incoming.size
        uniques[0] += int((lo == hi).sum())
        uniques[1] += int((carry_lo == carry_hi).sum())
    return {"direct_coordinates_inside_each_interval": checked,
            "direct_unique_incoming_matches": uniques[0], "direct_unique_pre_mlp_matches": uniques[1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        parser.error("Enable Python assertions")
    base = json.loads(PARENT.read_bytes())
    assert args.snapshot.name == base["revision"]
    with lease(BUDGET) as (check, samples):
        args.work.mkdir(parents=True, exist_ok=False)
        os.environ.update(USE_TORCH="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          HF_HOME=str((args.work / "hf_home").resolve()),
                          HF_MODULES_CACHE=str((args.work / "hf_modules").resolve()))
        os.environ.pop("TRANSFORMERS_CACHE", None)
        for name in ("config.json", "modeling_nanbeige.py"):
            bound(args.snapshot / name, base["snapshot_pins"][name])
        bound(args.snapshot / "configuration_nanbeige.py", CONFIGURATION)
        assert {p: version(p) for p in PACKAGES} == PACKAGES
        import numpy as np
        import torch

        torch.set_num_threads(1)
        result = fixture(args.snapshot, check, np, torch)
        assert not torch.cuda.is_initialized()
    write_new(args.work / "verification.json", result | {
        "status": "passed", "pretrained_forwards": 0, "cuda_initialized": False,
        "source_sha256": {p.name: digest(p) for p in (HERE / "lattice.py", HERE / "native_check.py")},
        "archive_manifest_sha256": digest(HERE / "SOURCE_MANIFEST.json"),
        "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
        "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
    })
    print(result, flush=True)


if __name__ == "__main__":
    main()
