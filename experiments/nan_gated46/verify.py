"""Commission the real phase-boundary gate on tiny synthetic CPU BF16 weights."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from experiments.nan_delay39.contrasts import numpy_state
from experiments.nan_delay39.instrument import clean_row_closes, prepare
from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha

from .gated import GateCapture, preservation, run_world
from .host import qualify, score_world
from .source import CONFIG, HERE, bound_rows


def fixture_loop(torch, nn):
    class Norm(nn.Module):
        variance_epsilon = 1e-6

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(64, dtype=torch.bfloat16), requires_grad=False)

        def forward(self, hidden):
            scaled = hidden.float()
            scaled *= torch.rsqrt(scaled.square().mean(-1, keepdim=True) + self.variance_epsilon)
            return self.weight * scaled.to(hidden.dtype)

    class Layer(nn.Module):
        def __init__(self, loop, index):
            super().__init__()
            self.loop = loop
            self.self_attn = nn.Module()
            self.self_attn.layer_idx = index
            self.self_attn.o_proj = nn.Identity()
            self.mlp = nn.Module()
            self.mlp.down_proj = nn.Identity()

        def forward(self, h, *, attention_mask, past_key_value, cache_position, **_):
            self.loop.second_calls += int(self.loop.phase == 2)
            k = h[:, None].clone()
            _, values = past_key_value.update(k, k.clone(), self.self_attn.layer_idx)
            keys = torch.arange(values.shape[2])
            mask = (keys[None, None, :] <= cache_position[None, :, None]) & attention_mask[:, None, :keys.numel()].bool()
            average = (mask.float() @ values[:, 0].float()) / mask.sum(-1, keepdim=True).clamp_min(1)
            update = self.mlp.down_proj(self.self_attn.o_proj(average.to(h.dtype)))
            return ((h.float() + .25 * update.float()).to(h.dtype),)

    loop = SimpleNamespace(n=2, n_gpu_layers=1, device=torch.device("cpu"), second_calls=0)
    loop.base = nn.Module()
    loop.base.embed_tokens = nn.Embedding(32, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        loop.base.embed_tokens.weight.copy_((.6 + .2 * torch.randn(32, 64)).to(torch.bfloat16))
    loop.base.norm = Norm()
    loop.base._update_causal_mask = lambda attention, *_: attention
    loop.layers = nn.ModuleList([Layer(loop, 0), Layer(loop, 1)])
    loop.stage_layer = Layer(loop, 1)
    loop.model = loop.base.requires_grad_(False)
    loop._load_stage = lambda _layer: loop.stage_layer
    return loop


def refused(operation, expect, error_type):
    try:
        operation()
    except error_type as err:
        if expect not in str(err):
            raise
    else:
        raise AssertionError(f"Expected refusal containing {expect}")


def cpu_gates(cfg, check):
    import numpy as np
    import torch
    from torch import nn
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    torch.set_num_threads(1)
    torch.manual_seed(46)
    random = np.random.default_rng(46)
    loop = fixture_loop(torch, nn)
    weights = (.015 * torch.randn(32, 64)).to(torch.bfloat16)
    weights[0].fill_(.125)
    head = StreamedVocabularyHead(weights, loop.device, 7)
    engine = GateCapture(loop, head, layer=1, chunk=2)
    prompt = [1, 3, 8, 14, 22]
    toy = cfg | {"phase1_dose_fraction": .05, "minimum_donor_cosine": 0.0, "maximum_head_delta": .5}
    tokens = (2, 1)
    try:
        check()
        baseline = engine.capture([prompt, prompt])
        h = numpy_state(baseline["residual"][1, 1])
        patch = prepare(h, random.normal(size=len(h)), toy["phase1_dose_fraction"], .03, .98, np)
        assert patch.eligible
        planned = engine.planned_head(baseline, patch)
        native = baseline["logits"][0]
        assert preservation(native, planned, tokens, toy)["passed"]
        start = loop.second_calls
        bad = planned.clone()
        bad[1, 12] += 10
        refused(lambda: engine.capture([prompt, prompt], patch, planned=bad, baseline=native, tokens=tokens, cfg=toy),
                "refused head plan", ValueError)
        failed_plan_calls = loop.second_calls - start
        falsified = planned.clone()
        falsified[1, 12] += .001
        assert preservation(native, falsified, tokens, toy)["passed"]
        refused(lambda: engine.capture([prompt, prompt], patch, planned=falsified, baseline=native,
                                       tokens=tokens, cfg=toy), "second pass refused", RuntimeError)
        failed_boundary_calls = loop.second_calls - start
        assert failed_boundary_calls == failed_plan_calls == 0
        accepted = engine.capture([prompt, prompt], patch, planned=planned, baseline=native, tokens=tokens, cfg=toy)
        clean_row_closes(baseline, accepted, expect_edit=True)
        assert loop.second_calls > start and not torch.equal(baseline["inputs"][2, 1], accepted["inputs"][2, 1])
        clean_row_closes(baseline, engine.capture([prompt, prompt]), expect_edit=False)
        bank = np.eye(64)
        writes = np.linalg.qr(random.normal(size=(64, 8)), mode="reduced")[0].T
        # The instrument uses the same sign-paired registered arms; toy radius and task label are synthetic.
        runbank = bank[:, :8]
        tok = SimpleNamespace(decode=lambda ids, **_: "A" if ids[0] == 0 else "B")
        record = run_world(engine, baseline, prompt, random.normal(size=8), runbank, writes,
                           tokens, "fixture", toy, tok, check, np)
        assert record["phase2_edit_count"] == 4 and score_world(record, "A", "B", toy) is not None
        restrictive = toy | {"maximum_head_delta": 0.0}
        start = loop.second_calls
        blocked = run_world(engine, baseline, prompt, random.normal(size=8), runbank, writes,
                            tokens, "fixture", restrictive, tok, check, np)
        assert blocked["status"] == "phase1_gate_unqualified" and blocked["phase2_edit_count"] == 0
        assert loop.second_calls == start and score_world(blocked, "A", "B", restrictive) is None
    finally:
        engine.close()
        head.close()
    assert not torch.cuda.is_initialized()
    return {"bad_plan_edited_pass2_calls": failed_plan_calls, "live_mismatch_edited_pass2_calls": failed_boundary_calls,
            "four_matching_edits_scored": record["phase2_edit_count"], "failed_world_phase2_calls": 0,
            "native_recovery_exact": True, "cuda_initialized": False, "new_pretrained_forwards": 0}


def independent_gate(folder, out, cfg):
    gold = bound_rows(folder, "gold")
    records = [{"id": row["id"], "token_id": ord(show) - 65, "text": show}
               for row in gold for show in ["A" if row["arm"] == "focused_erased" else row["answer"]]]
    paths = out / "native_fixture.json", out / "native_gate_fixture.json"
    write_new(paths[0], records)
    command = ["C:/Python312/python.exe", "-X", "utf8", "-B", "-S", "-m", "experiments.nan_gated46.host",
               "--source-dir", str(folder), "--native-records", str(paths[0]), "--output", str(paths[1])]
    subprocess.run(command, check=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
    saved = json.loads(paths[1].read_bytes())
    assert saved == qualify(records, gold, cfg) and saved["passed"] and len(saved["qualified_focused_roots"]) == 8
    first_copy = next(row["id"] for row in gold if row["arm"] == "copy")
    failed = [row | {"text": "X"} if row["id"] == first_copy else row for row in records]
    assert not qualify(failed, gold, cfg)["passed"]
    return {"independent_native_fixture_rows": len(records), "failed_copy_removed_native_admission": True}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-dir", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    cfg = json.loads(CONFIG.read_bytes())
    limits = {"minimum_start_available_RAM_bytes": 5 * 2**30, "minimum_available_RAM_bytes": 4 * 2**30,
              "maximum_private_bytes": 768 * 2**20, "maximum_seconds": 120}
    args.output.mkdir(parents=True, exist_ok=False)
    with lease(limits) as (check, samples):
        proof = independent_gate(args.source_dir.resolve(), args.output, cfg)
        os.environ.update(USE_TORCH="1", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1")
        result = cpu_gates(cfg, check)
    names = ("gated.py", "source.py", "host.py", "worker.py", "verify.py")
    receipt = {"status": "passed", **result, **proof, "source_sha256": {n: sha(HERE / n) for n in names},
               "max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
               "max_sampled_working_set_bytes": max(v["working_set_bytes"] for v in samples),
               "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples)}
    write_new(args.output / "verification.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
