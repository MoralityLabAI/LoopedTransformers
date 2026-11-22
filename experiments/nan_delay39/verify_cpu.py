"""Tiny BF16 DynamicCache fixture for the actual delayed-capture and contrast path."""
import argparse
import json
import os
from pathlib import Path

from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha


def verify(check):
    import numpy as np
    import torch
    from torch import nn

    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead
    from .contrasts import measure_pair, numpy_state
    from .instrument import DelayedCapture, clean_row_closes, prepare

    torch.set_num_threads(1)
    rng = np.random.default_rng(39)
    torch.manual_seed(39)

    class Layer(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.layer_idx = index
            self.self_attn.o_proj = nn.Identity()
            self.mlp = nn.Module()
            self.mlp.down_proj = nn.Identity()

        def forward(self, h, *, attention_mask, past_key_value, cache_position, **unused):
            kv = h[:, None].clone()
            if past_key_value is not None:
                _, kv = past_key_value.update(kv, kv.clone(), self.self_attn.layer_idx)
            keys = torch.arange(kv.shape[2])
            mask = (keys[None, None, :] <= cache_position[None, :, None]) & attention_mask[:, None, :kv.shape[2]].bool()
            values = kv[:, 0].float()
            avg = ((mask.float() @ values) / mask.sum(-1, keepdim=True).clamp_min(1)).to(h.dtype)
            update = self.mlp.down_proj(self.self_attn.o_proj(avg))
            return ((h.float() + update.float() * 0.25).to(h.dtype),)

    class Loop:
        def __init__(self):
            self.base = nn.Module()
            self.base.embed_tokens = nn.Embedding(32, 8, dtype=torch.bfloat16)
            self.base.norm = nn.Identity()
            self.base._update_causal_mask = lambda attention, *_args: attention
            self.layers = nn.ModuleList([Layer(0), Layer(1)])
            self.model = self.base.requires_grad_(False)
            self.device = torch.device("cpu")
            self.n = 2
            self.n_gpu_layers = 1
            self.stage_layer = Layer(1)

        def _load_stage(self, source):
            self.stage_layer.self_attn.layer_idx = source.self_attn.layer_idx
            return self.stage_layer

    def explicit_reference(loop, head, prompts, patch):
        """No cache or patch hook. Apply a single edit to the full phase-1 tensor."""
        width = max(map(len, prompts))
        attention = torch.zeros((2, width), dtype=torch.long)
        hidden = torch.zeros((2, width, 8), dtype=torch.bfloat16)
        for row, ids in enumerate(prompts):
            attention[row, :len(ids)] = 1
            hidden[row, :len(ids)] = loop.base.embed_tokens(torch.tensor(ids))
        boundaries = []
        for phase in (1, 2):
            for layer in loop.layers:
                hidden = layer(hidden, attention_mask=attention, past_key_value=None,
                               cache_position=torch.arange(width))[0]
            if phase == 1:
                hidden[1, len(prompts[1]) - 1] = torch.as_tensor(patch.after, dtype=torch.bfloat16)
            boundaries.append(head(torch.stack([hidden[i, len(ids) - 1] for i, ids in enumerate(prompts)])))
        return torch.stack(boundaries)

    cfg = {"phase1_components": list(range(1, 8)), "phase1_dose_fraction": .25,
           "dose_relative_tolerance": .03, "minimum_direction_cosine": .98, "phase2_readout_dose_fraction": .25,
           "maximum_phase2_readout_replay_logit_delta": .125,
           "max_contrast_dose_gap_fraction": .03}
    loop = Loop()
    head = StreamedVocabularyHead(torch.randn(32, 8, dtype=torch.bfloat16), loop.device, 7)
    engine = DelayedCapture(loop, head, layer=1, chunk=2)
    first = [1, 3, 8, 14, 22]
    other = [1, 3, 7, 12, 20]
    phase_proof = {}
    expect = {}
    try:
        with torch.inference_mode():
            check()
            baseline = engine.capture([first, first])
            candidate = prepare(numpy_state(baseline["residual"][1, 1]), rng.normal(size=8), .25, .03, .98, np)
            assert candidate.eligible
            changed = engine.capture([first, first], candidate)
            clean_row_closes(baseline, changed, expect_edit=True)
            expect["max_full_tensor_reference_logit_difference"] = float(
                (explicit_reference(loop, head, [first, first], candidate) - changed["logits"]).abs().max())
            assert expect["max_full_tensor_reference_logit_difference"] == 0
            expect["phase1_projection_input_exactly_preserved"] = torch.equal(
                baseline["inputs"][1, 1], changed["inputs"][1, 1])
            expect["phase2_projection_input_changed"] = not torch.equal(
                baseline["inputs"][2, 1], changed["inputs"][2, 1])
            assert all(expect[k] for k in ("phase1_projection_input_exactly_preserved", "phase2_projection_input_changed"))

            # One physical staged layer and ragged data exercise a padding-hole CAL capture.
            ragged = engine.capture([first, other[:-1]])
            assert ragged["physical_length"] == 5 and ragged["semantic_lengths"] == [5, 4]
            V = np.eye(8)
            U = np.linalg.qr(rng.normal(size=(8, 8)))[0].T
            outcome = measure_pair(engine, first, other, V, U, list(range(16)), cfg, "SYNTHETIC-CAL", np, check)
            edges = outcome["signed_phase2_edges_per_l2"]
            sink = outcome["phase2_terminal"]
            assert sink["selected_head_replay_max_delta"] == 0
            assert all(v["signed_label_slopes_per_read_coefficient"] is not None
                       for v in sink["phase2_write_responses"].values())
            offered = [name for name, edit in outcome["cases"].items() if edit["eligible"]]
            assert outcome["self_closes"] and len(edges) == 7 and len(offered) == 17
            assert all(edge is not None and len(edge) == 8 for edge in edges.values())
            assert max(abs(e) for edge in edges.values() for e in edge) > 0
            assert all(outcome["donor_sham_pairs"].values())
            altered_base = numpy_state(baseline["residual"][1, 1] + 1)
            bad = prepare(altered_base, rng.normal(size=8), .25, .03, .98, np)
            assert bad.eligible
            try:
                engine.capture([first, first], bad)
            except RuntimeError as error:
                assert "Pre-edit bytes" in str(error)
            else:
                raise AssertionError("A stale patch unexpectedly reached the decoder")
            clean_row_closes(baseline, engine.capture([first, first]), expect_edit=False)
            phase_proof = {"offered_synthetic_edits": len(offered), "signed_edges_rows": len(edges),
                           "signed_read_coefficients_per_edge_row": 8, "matched_donor_shams": 2, "qualified_phase2_sink_edges": 8,
                           "stale_patch_refused_and_following_native_capture_exact": True}
    finally:
        engine.close()
        head.close()
    assert not torch.cuda.is_initialized()
    return {**phase_proof, **expect}


def main():
    if not __debug__:
        raise RuntimeError("Assertion-enabled test required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    limits = {"minimum_start_available_RAM_bytes": 5 * 2**30, "minimum_available_RAM_bytes": 4 * 2**30,
              "maximum_private_bytes": 768 * 2**20, "maximum_seconds": 120}
    with lease(limits) as (check, samples):
        os.environ.update(USE_TORCH="1", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1")
        proof = verify(check)
    here = Path(__file__).resolve().parent
    result = {"status": "passed", "scope": "actual two-phase cache, staged layer and head on synthetic BF16 CPU weights",
              **proof, "new_pretrained_forwards": 0, "cuda_initialized": False,
              "source_sha256": {name: sha(here / name) for name in ("instrument.py", "contrasts.py", "sink_edges.py", "verify_cpu.py")},
              "max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
              "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples)}
    write_new(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
