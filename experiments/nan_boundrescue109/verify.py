"""Commission the gated residual and K rescue on random pinned decoder weights."""

import argparse
from contextlib import ExitStack
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

from experiments.nan_centersham108 import run as center
from experiments.nan_centersham108.search import midpoint_round
from experiments.nan_phase35.run import lease
from experiments.nan_premlp64.verify import BUDGET, CONFIGURATION, PACKAGES, hook_count
from experiments.nan_radial76.native import WRITER
from experiments.nan_readpath37.run import bf16, bound, digest, write_new

from .core import PARENT, REPO, run_root
from .native import BoundaryCapture, HeadPlan


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SOURCE_MANIFEST.json"


def registration():
    if center.registration() != json.loads(center.MANIFEST.read_bytes()):
        raise ValueError("The centered sham source changed")
    paths = [HERE / name for name in (".gitattributes", "__init__.py", "native.py", "core.py", "verify.py", "PROTOCOL.md")]
    paths += [center.MANIFEST, PARENT] + [REPO / name for name in (
        "experiments/nan_gated46/gated.py", "experiments/nan_radial76/native.py",
        "experiments/nan_reader71/native.py", "experiments/nan_writer77/pipeline.py",
        "experiments/nan_premlp64/verify.py", "mechinterp/nanbeige_bpd_gym/chunked_cache.py",
        "mechinterp/nanbeige_bpd_gym/streamed_vocab_head.py")]
    return {"experiment": "NAN-BOUNDRESCUE-109", "budget": BUDGET, "packages": PACKAGES,
            "files_sha256": {p.relative_to(REPO).as_posix(): digest(p) for p in paths}}


def exercise(snapshot, check, np, torch):
    from experiments.nan_recomp09.common import target_digest
    from experiments.nan_relabel68.pipeline import paired_heads
    from experiments.nan_stream36.batch_engine import BatchedDepthEngine
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import build_nanbeige_model_and_loop
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    torch.manual_seed(10931)
    model, loop = build_nanbeige_model_and_loop(snapshot, exec_device=torch.device("cpu"), dtype=torch.bfloat16,
        n_gpu_layers=2, random_init=True, config_overrides={"vocab_size": 97, "num_hidden_layers": 3, "hidden_size": 192,
        "intermediate_size": 80, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "kv_channels": 16})
    model.requires_grad_(False)
    loop.stage_layer.requires_grad_(False)
    original, hooks = target_digest(loop), hook_count(loop)
    prompts = [[1, 4, 32, 5, 29, 6, 42], [8, 39, 3, 4, 65, 77, 91]]
    labels, design = (7, 19), np.arange(1, 96, 11)
    commits, edits = [], []
    p2_calls = [0]

    class ObservedCapture(BoundaryCapture):
        def capture(self, ids, **kwargs):
            edit = kwargs.get("edit")
            if edit is not None and not torch.equal(*edit):
                if not commits or commits[-1]["status"] != "qualified":
                    raise AssertionError("Root edits started before preflight was accepted")
                edits.append(kwargs.get("rescue") is not None)
            return super().capture(ids, **kwargs)

    def synthetic(amp, refuse=False):
        def plans(anchor, root, check):
            result = {}
            for i, world in enumerate(("A", "B")):
                raw = anchor["writer_raw"][1, i].float().numpy().astype(np.float64)
                for sign in (1, -1):
                    v, w = np.zeros_like(raw), np.zeros_like(raw)
                    v[12], w[13] = sign*amp, sign*amp
                    a, b = bf16(raw-v, np).astype(np.float64), bf16(raw+v, np).astype(np.float64)
                    center = (a+b)/2
                    left, right = midpoint_round(center, w[None, :], a, b, np)
                    result[world, sign] = a, b, left[0], right[0]
            if refuse:
                result.pop(("B", -1))
            return result, {"source": "synthetic"}
        return plans

    def count_phase2(_module, _args):
        if loop.phase == 2:
            p2_calls[0] += 1

    with ExitStack() as stack:
        head = StreamedVocabularyHead(model.lm_head.weight, loop.device, 13)
        stack.callback(head.close)
        engine = BatchedDepthEngine(loop, head, chunk=3)
        try:
            reference = torch.stack(paired_heads(engine, prompts, lambda *_: check()))
        finally:
            engine.close()
        cap = ObservedCapture(loop, head, chunk=3)
        stack.callback(cap.close)
        observed = loop.layers[0].register_forward_pre_hook(count_phase2)
        stack.callback(observed.remove)
        assert torch.equal(cap.capture(prompts)["logits"], reference)
        refused = run_root(cap, "missing", prompts, labels, synthetic(.004, True), design, commits.append, check, np)
        assert refused["status"] == "geometry_unqualified" and not edits
        refused = run_root(cap, "unsafe", prompts, labels, synthetic(250.), design, commits.append, check, np)
        assert refused["status"] == "head_unqualified" and not edits and refused["edited_pass2_runs"] == 0

        duplicate = cap.capture([prompts[0]]*2)
        base = duplicate["writer_raw"][WRITER]
        prediction = cap.predict(duplicate, base)
        stale = HeadPlan(torch.zeros_like(prediction.normalized), prediction.logits)
        before = p2_calls[0]
        try:
            cap.capture([prompts[0]]*2, edit=(base, base), planned=stale)
        except RuntimeError as error:
            assert "before pass 2" in str(error)
        else:
            raise AssertionError("The stale preflight was accepted")
        assert p2_calls[0] == before and cap.cache.boundary_gate is None and not cap.instrument.active

        closed = run_root(cap, "accepted", prompts, labels, synthetic(.004), design, commits.append, check, np)
        assert closed["status"] == "closed" and closed["edited_pass2_runs"] == len(edits) == 32
        assert edits.count(True) == 16 and len(commits) == 3
        arms = [a for w in closed["worlds"].values() for receipt in w.values() for a in receipt["arms"].values()]
        assert len(arms) == 16 and all(a["zeroed_k_delta"] for a in arms)
        k_changes = sum(a["edited"]["raw_projection"]["k"]["l2_change"] > 0 for a in arms)
        assert k_changes > 0
        assert all(a["k_rescue"]["reached_projection"]["k"]["l2_change"] == 0 for a in arms)
    assert not torch.cuda.is_initialized() and target_digest(loop) == original and hook_count(loop) == hooks
    return {"staged_both_heads_exact": True, "geometry_refusal_prevents_all_edits": True,
            "head_refusal_prevents_all_edits": True, "stale_plan_prevents_pass2": True,
            "edited_pass2_forward_calls": len(edits), "exact_k_rescues": edits.count(True),
            "nonzero_pre_rescue_k": k_changes, "unchanged_checkpoint_and_hooks": True,
            "cuda_initialized": False, "pretrained_decoder_forwards": 0}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--work", type=Path)
    args = cli.parse_args()
    lock = registration()
    if args.freeze:
        write_new(MANIFEST, lock)
        return
    if not args.snapshot or not args.work or lock != json.loads(MANIFEST.read_bytes()):
        cli.error("Require the frozen source, pinned snapshot and a fresh work path")
    base = json.loads((REPO / "experiments/nan_readpath37/registration.json").read_bytes())
    if args.snapshot.name != base["revision"] or "torch" in sys.modules:
        raise ValueError("The pinned CPU fixture must own its model imports")
    with lease(BUDGET) as (check, samples):
        args.work.mkdir(parents=True, exist_ok=False)
        os.environ.update(USE_TORCH="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          HF_HOME=str((args.work/"hf_home").resolve()),
                          HF_MODULES_CACHE=str((args.work/"hf_modules").resolve()))
        os.environ.pop("TRANSFORMERS_CACHE", None)
        for name in ("config.json", "modeling_nanbeige.py"):
            bound(args.snapshot/name, base["snapshot_pins"][name])
        bound(args.snapshot/"configuration_nanbeige.py", CONFIGURATION)
        assert {name:version(name) for name in PACKAGES} == PACKAGES
        import numpy as np
        import torch

        torch.set_num_threads(1)
        details = exercise(args.snapshot, check, np, torch)
    assert lock == registration()
    write_new(args.work/"verification.json", details | {
        "status": "passed", "source_manifest_sha256": digest(MANIFEST),
        "mode": "random_staged_cpu_bf16", "formal_comparisons": 0, "sealed_test_accessed": False,
        "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
        "max_sampled_working_set_bytes": max(s["working_set_bytes"] for s in samples),
        "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
    })


if __name__ == "__main__":
    main()
