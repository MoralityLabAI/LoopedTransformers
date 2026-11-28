"""CPU unit tests for InkyNano GGUF conversion, streaming writer, resume and LDT loader."""
from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from safetensors.torch import save_file

from gguf_backend import (
    GGMLCBackend,
    SyntheticRefBackend,
    discover_ggml_lib_dir,
    resolve_ggml_type,
)
from gguf_convert import (
    BACKBONE_RELPATH,
    MANIFEST_RELPATH,
    build_master_manifest,
    convert_layer_shard,
    convert_text_backbone,
    render_hf_readme,
    run_all_resumable,
    run_softcapped_swiglu,
    shard_relpath,
)
from gguf_dequant import (
    E2M1_VALUES,
    LocalSafetensorsSource,
    dequantize_expert,
    dequantize_projection_standalone,
)
from gguf_format import (
    GGMLType,
    GGUFBundleReader,
    StreamingGGUFWriter,
    TensorSpec,
    ggml_nbytes,
)
from gguf_loader import BackboneBundleReader, LDTResidencyManager, SelectiveExpertStore
from gguf_profiles import (
    EXPERT_PROFILES,
    classify_backbone_tensor,
    is_sensitive_backbone_tensor,
    is_text_backbone_tensor,
    rtx3050_residency_table,
)
from native_expert import expert as native_expert_fn


def _make_synthetic_checkpoint(path: Path, n_experts: int = 4, d: int = 64, m: int = 32) -> Path:
    """Create a tiny valid safetensors file exercising both NVFP4 (layer 0) and MXFP8 (layer 20) + backbone."""
    rng = torch.Generator().manual_seed(42)
    tensors = {}

    # Layer 0: NVFP4 (w13 packed [E, 2m, d//2] U8 + F8_E4M3 scale + F32 amax + F32 denom)
    tensors["model.llm.layers.0.mlp.experts.w13_weight"] = torch.randint(
        0, 256, (n_experts, 2 * m, d // 2), generator=rng, dtype=torch.uint8
    )
    tensors["model.llm.layers.0.mlp.experts.w13_weight_scale"] = (
        torch.rand((n_experts, 2 * m, d // 16), generator=rng) * 0.5 + 0.25
    ).to(torch.float8_e4m3fn)
    tensors["model.llm.layers.0.mlp.experts.w13_weight_amax"] = (
        torch.rand((n_experts,), generator=rng, dtype=torch.float32) * 0.8 + 0.2
    )
    tensors["model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator"] = torch.tensor(
        [448.0 * 6.0], dtype=torch.float32
    )

    tensors["model.llm.layers.0.mlp.experts.w2_weight"] = torch.randint(
        0, 256, (n_experts, d, m // 2), generator=rng, dtype=torch.uint8
    )
    tensors["model.llm.layers.0.mlp.experts.w2_weight_scale"] = (
        torch.rand((n_experts, d, m // 16), generator=rng) * 0.5 + 0.25
    ).to(torch.float8_e4m3fn)
    tensors["model.llm.layers.0.mlp.experts.w2_weight_amax"] = (
        torch.rand((n_experts,), generator=rng, dtype=torch.float32) * 0.8 + 0.2
    )
    tensors["model.llm.layers.0.mlp.experts.w2_weight_outer_scale_denominator"] = torch.tensor(
        [448.0 * 6.0], dtype=torch.float32
    )

    # Layer 20: MXFP8 (w13 [E, 2m, d] F8_E4M3 + U8 e8m0 scale per 32 values)
    tensors["model.llm.layers.20.mlp.experts.w13_weight"] = (
        (torch.randn((n_experts, 2 * m, d), generator=rng) * 0.5).clamp(-3.0, 3.0).to(torch.float8_e4m3fn)
    )
    tensors["model.llm.layers.20.mlp.experts.w13_weight_scale"] = torch.full(
        (n_experts, 2 * m, d // 32), 125, dtype=torch.uint8
    )
    tensors["model.llm.layers.20.mlp.experts.w2_weight"] = (
        (torch.randn((n_experts, d, m), generator=rng) * 0.5).clamp(-3.0, 3.0).to(torch.float8_e4m3fn)
    )
    tensors["model.llm.layers.20.mlp.experts.w2_weight_scale"] = torch.full(
        (n_experts, d, m // 32), 125, dtype=torch.uint8
    )

    # Text-only backbone tensors (large eligible + sensitive routers, recurrent controllers, sconv, attn_res, norms)
    tensors["model.llm.embed.weight"] = (torch.randn((64, 256), generator=rng) * 0.02).to(torch.bfloat16)
    tensors["model.llm.unembed.weight"] = (torch.randn((64, 256), generator=rng) * 0.02).to(torch.bfloat16)
    tensors["model.llm.norm.weight"] = torch.ones((256,), dtype=torch.bfloat16)
    tensors["model.llm.layers.0.attn.wq_du.weight"] = (torch.randn((64, 256), generator=rng) * 0.02).to(torch.bfloat16)
    tensors["model.llm.layers.0.attn.wo_ud.weight"] = (torch.randn((64, 256), generator=rng) * 0.02).to(torch.bfloat16)
    tensors["model.llm.layers.0.mlp.gate.weight"] = (torch.randn((16, 256), generator=rng) * 0.02).to(torch.bfloat16)
    tensors["model.llm.layers.0.mlp.gate.bias"] = torch.randn((16,), generator=rng, dtype=torch.float32)
    tensors["model.llm.layers.6.controllers.attn.down_weight"] = (torch.randn((16, 64), generator=rng) * 0.02).to(torch.bfloat16)
    tensors["model.llm.layers.6.controllers.mlp.gain_weight"] = (torch.randn((64, 16), generator=rng) * 0.02).to(torch.bfloat16)
    tensors["model.llm.middle_cycle_cross_pass_queries"] = torch.randn((4, 64), generator=rng, dtype=torch.float32)
    tensors["model.llm.attn_res.attn_queries.0"] = torch.randn((64,), generator=rng, dtype=torch.float32)
    tensors["model.llm.layers.0.mlp_sconv.weight"] = torch.randn((64, 1, 4), generator=rng).to(torch.bfloat16)
    # Vision tensor that MUST be excluded from text-only backbone
    tensors["model.visual.proj_in.weight"] = torch.randn((32, 32), generator=rng).to(torch.bfloat16)

    save_file(tensors, str(path))
    return path


class GGUFConversionTests(unittest.TestCase):
    def test_block_divisibility_and_llama_cpp_fallback(self):
        # gate/up ne0=2304 is divisible by 256 (Q4_K / Q6_K native)
        self.assertEqual(resolve_ggml_type(GGMLType.Q4_K, 2304, "llama_cpp_fallback"), (GGMLType.Q4_K, 2304))
        self.assertEqual(resolve_ggml_type(GGMLType.Q6_K, 2304, "llama_cpp_fallback"), (GGMLType.Q6_K, 2304))
        # down ne0=1152 = 36*32 (1152 % 256 = 128): canonical llama.cpp fallback Q4_K->Q5_0, Q6_K->Q8_0
        self.assertEqual(resolve_ggml_type(GGMLType.Q4_K, 1152, "llama_cpp_fallback"), (GGMLType.Q5_0, 1152))
        self.assertEqual(resolve_ggml_type(GGMLType.Q6_K, 1152, "llama_cpp_fallback"), (GGMLType.Q8_0, 1152))
        self.assertEqual(resolve_ggml_type(GGMLType.Q4_K, 1152, "q4_0_fallback"), (GGMLType.Q4_0, 1152))
        self.assertEqual(resolve_ggml_type(GGMLType.Q4_K, 1152, "pad_to_block"), (GGMLType.Q4_K, 1280))
        self.assertEqual(resolve_ggml_type(GGMLType.Q6_K, 1152, "pad_to_block"), (GGMLType.Q6_K, 1280))
        with self.assertRaises(ValueError):
            resolve_ggml_type(GGMLType.Q4_K, 1152, "strict")

    def test_profiles_and_rtx3050_budgets(self):
        pure = EXPERT_PROFILES["pure_q4_k"]
        mixed = EXPERT_PROFILES["mixed_q4_k_q6_k"]
        self.assertEqual(pure.expert_payload_bytes(0), pure.expert_payload_bytes(23))
        self.assertGreater(mixed.expert_payload_bytes(23), mixed.expert_payload_bytes(0))
        table = rtx3050_residency_table(mixed, backbone_bytes=1_200_000_000)
        # k=8 and k=16 must leave positive headroom under 4 GiB before KV/activations
        self.assertGreater(table["rtx3050_4gib_scenarios"]["k_8"]["headroom_before_kv_activations_gib"], 1.5)
        self.assertGreater(table["rtx3050_4gib_scenarios"]["k_16"]["headroom_before_kv_activations_gib"], 0.8)

    def test_backbone_sensitive_tensor_classification(self):
        self.assertFalse(is_text_backbone_tensor("model.visual.proj_in.weight"))
        self.assertFalse(is_text_backbone_tensor("model.llm.layers.0.mlp.experts.w13_weight"))
        sensitive_names = [
            "model.llm.layers.0.mlp.gate.weight",
            "model.llm.layers.0.mlp.gate.bias",
            "model.llm.layers.10.controllers.attn.forget_weight",
            "model.llm.layers.12.controllers.mlp.down_weight",
            "model.llm.middle_cycle_cross_pass_queries",
            "model.llm.middle_cycle_cross_pass_kv_queries",
            "model.llm.attn_res.attn_queries.4",
            "model.llm.layers.3.mlp_sconv.weight",
            "model.llm.layers.5.attn.k_sconv.weight_proj",
            "model.llm.layers.0.attn.attn_sink_proj.weight",
            "model.llm.layers.0.attn.q_l2_clip.scale",
            "model.llm.norm.weight",
        ]
        for name in sensitive_names:
            self.assertTrue(is_sensitive_backbone_tensor(name), name)
            src_dt = "F32" if ("bias" in name or "queries" in name) else "BF16"
            shape = (128, 2304) if "weight" in name else (2304,)
            qtype, cat = classify_backbone_tensor(name, shape, src_dt, "backbone_mixed_q4_k_q6_k")
            self.assertIn(qtype, (GGMLType.BF16, GGMLType.F32), name)
            self.assertTrue(cat.startswith("preserved_"), (name, cat))

        # Quantizable attention & embed/unembed in mixed profile
        self.assertEqual(
            classify_backbone_tensor("model.llm.unembed.weight", (201024, 2304), "BF16", "backbone_mixed_q4_k_q6_k")[0],
            GGMLType.Q6_K,
        )
        self.assertEqual(
            classify_backbone_tensor("model.llm.layers.0.attn.wq_du.weight", (3072, 2304), "BF16", "backbone_mixed_q4_k_q6_k")[0],
            GGMLType.Q4_K,
        )
        self.assertEqual(
            classify_backbone_tensor("model.llm.layers.0.attn.wo_ud.weight", (2304, 3072), "BF16", "backbone_mixed_q4_k_q6_k")[0],
            GGMLType.Q6_K,
        )

    def test_nvfp4_and_mxfp8_dequant_and_softcapped_swiglu(self):
        # Hand-verified NVFP4 single-block test with Interleave(dim=1)
        # Byte 0x21 -> low nibble 1 (0.5), high nibble 2 (1.0)
        w = torch.full((1, 4, 8), 0x21, dtype=torch.uint8)
        scale = torch.ones((1, 4, 1), dtype=torch.float8_e4m3fn)
        amax = torch.tensor([2.0], dtype=torch.float32)
        denom = torch.tensor([1.0], dtype=torch.float32)
        keys = ["w", "s", "a", "d"]
        out = dequantize_projection_standalone({"w": w, "s": scale, "a": amax, "d": denom}, keys, "w2", loader_cast=True)
        self.assertEqual(tuple(out.shape), (4, 16))
        # Even indices: 0.5 * 1.0 * 2.0 = 1.0; odd indices: 1.0 * 1.0 * 2.0 = 2.0
        self.assertTrue(torch.allclose(out[:, 0::2].float(), torch.ones((4, 8))))
        self.assertTrue(torch.allclose(out[:, 1::2].float(), torch.full((4, 8), 2.0)))

        # Verify Interleave(dim=1) separates even rows into gate and odd rows into up
        w_int = torch.zeros((1, 4, 8), dtype=torch.uint8)
        w_int[:, 0::2, :] = 0x11  # 0.5 -> scaled by 2.0 = 1.0 (gate)
        w_int[:, 1::2, :] = 0x22  # 1.0 -> scaled by 2.0 = 2.0 (up)
        gu = dequantize_projection_standalone({"w": w_int, "s": scale, "a": amax, "d": denom}, keys, "w13", loader_cast=True)
        gate, up = gu.chunk(2, dim=0)
        self.assertTrue(torch.allclose(gate.float(), torch.ones((2, 16))))
        self.assertTrue(torch.allclose(up.float(), torch.full((2, 16), 2.0)))

        # Verify MXFP8 dequantization (scale = 128 => 2**(128-127) = 2.0)
        wm = torch.full((1, 2, 32), 1.5, dtype=torch.float8_e4m3fn)
        sm = torch.full((1, 2, 1), 128, dtype=torch.uint8)
        om = dequantize_projection_standalone({"w": wm, "s": sm}, ["w", "s", "a", "d"], "w2", loader_cast=True)
        self.assertTrue(torch.allclose(om.float(), torch.full((2, 32), 3.0)))

        # Verify soft-capped SwiGLU matches native_expert.expert exactly in native dtype
        x = torch.randn((4, 16), dtype=torch.bfloat16)
        down = torch.randn((16, 2), dtype=torch.bfloat16) * 0.1
        ref = native_expert_fn(x, gu, down)
        got = run_softcapped_swiglu(x, gate, up, down)
        self.assertEqual((got.float() - ref.float()).abs().max().item(), 0.0)

    def test_streaming_gguf_writer_and_selective_reader(self):
        backend = SyntheticRefBackend()
        rng = np.random.default_rng(7)
        a = rng.standard_normal((16, 64)).astype(np.float32)
        b = rng.standard_normal((32, 32)).astype(np.float32)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bundle.gguf"
            specs = [
                TensorSpec("blk.0.exp.000.ffn_gate.weight", (16, 64), GGMLType.Q4_0),
                TensorSpec("blk.0.exp.000.ffn_down.weight", (32, 32), GGMLType.Q8_0),
                TensorSpec("model.llm.norm.weight", (64,), GGMLType.BF16),
            ]
            meta = {
                "general.architecture": "inkling2_loop_bundle",
                "general.type": "tensors",
                "inkynano.bundle_kind": "expert_layer_shard",
                "inkynano.llama_cpp_dropin_compatible": False,
            }
            qa = backend.quantize(a, GGMLType.Q4_0)
            qb = backend.quantize(b, GGMLType.Q8_0)
            qn = backend.quantize(np.ones((64,), dtype=np.float32), GGMLType.BF16)

            w = StreamingGGUFWriter(p, metadata=meta, tensors=specs, alignment=32)
            r0 = w.write_tensor("blk.0.exp.000.ffn_gate.weight", qa)
            r1 = w.write_tensor("blk.0.exp.000.ffn_down.weight", qb)
            r2 = w.write_tensor("model.llm.norm.weight", qn)
            info = w.finalize()

            reader = GGUFBundleReader(p)
            layout = reader.verify_layout()
            self.assertEqual(layout["tensor_count"], 3)
            self.assertEqual(layout["general_type"], "tensors")
            self.assertIs(layout["llama_cpp_dropin_compatible"], False)
            self.assertEqual(reader.read_tensor_bytes("blk.0.exp.000.ffn_gate.weight", verify_sha256=r0["sha256"]), qa)
            self.assertEqual(reader.read_tensor_bytes("blk.0.exp.000.ffn_down.weight", verify_sha256=r1["sha256"]), qb)
            self.assertEqual(reader.read_tensor_bytes("model.llm.norm.weight", verify_sha256=r2["sha256"]), qn)
            self.assertEqual(len(info["sha256"]), 64)

            # Corrupt hash must raise
            with self.assertRaises(ValueError):
                reader.read_tensor_bytes("blk.0.exp.000.ffn_gate.weight", verify_sha256="0" * 64)

    def test_synthetic_end_to_end_resume_manifest_and_ldt_loader(self):
        """Full synthetic pipeline: convert 2 layers + backbone -> resume skips all -> selective LDT cache."""
        backend = SyntheticRefBackend()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt = _make_synthetic_checkpoint(root / "tiny.safetensors", n_experts=4, d=64, m=32)
            out_dir = root / "out"

            # Patch dimensions/profiles for tiny synthetic shapes (d=64, m=32) using Q4_0/Q8_0
            from gguf_profiles import ExpertQuantProfile
            tiny_prof = ExpertQuantProfile(
                name="tiny_test",
                description="Tiny Q4_0/Q8_0 synthetic test profile",
                nvfp4_types=(GGMLType.Q4_0, GGMLType.Q4_0, GGMLType.Q4_0),
                mxfp8_types=(GGMLType.Q8_0, GGMLType.Q8_0, GGMLType.Q8_0),
                incompat_policy="strict",
            )
            tiny_bb_cfg = {
                "description": "Tiny Q8_0/BF16 synthetic backbone",
                "embed": "Q8_0",
                "unembed": "Q8_0",
                "attn_q_gate": "Q8_0",
                "attn_kv_r": "Q8_0",
                "attn_o": "Q8_0",
            }

            with (
                mock.patch("gguf_profiles.HIDDEN_SIZE", 64),
                mock.patch("gguf_profiles.INTERMEDIATE_SIZE", 32),
                mock.patch("gguf_profiles.NUM_EXPERTS_PER_LAYER", 4),
                mock.patch("gguf_convert.HIDDEN_SIZE", 64),
                mock.patch("gguf_convert.INTERMEDIATE_SIZE", 32),
                mock.patch("gguf_convert.NUM_EXPERTS_PER_LAYER", 4),
                mock.patch.dict("gguf_profiles.EXPERT_PROFILES", {"tiny_test": tiny_prof}),
                mock.patch.dict("gguf_profiles.BACKBONE_PROFILES", {"tiny_bb": tiny_bb_cfg}),
            ):
                with LocalSafetensorsSource(ckpt) as src:
                    # 1. First pass converts layers 0 and 20 + text-only backbone
                    s1 = run_all_resumable(
                        source=src,
                        backend=backend,
                        out_dir=out_dir,
                        profile_name="tiny_test",
                        backbone_profile="tiny_bb",
                        layers=[0, 20],
                        hf_repo=None,
                        keep_local_shards=True,
                        dequant_backend="standalone",
                        num_experts=4,
                    )
                    self.assertEqual(s1["converted_layers"], [0, 20])
                    self.assertEqual(s1["skipped_layers"], [])
                    self.assertEqual(s1["backbone_status"], "converted")

                    # 2. Resume pass MUST skip both layers and backbone without repeating work
                    s2 = run_all_resumable(
                        source=src,
                        backend=backend,
                        out_dir=out_dir,
                        profile_name="tiny_test",
                        backbone_profile="tiny_bb",
                        layers=[0, 20],
                        hf_repo=None,
                        keep_local_shards=True,
                        dequant_backend="standalone",
                        num_experts=4,
                    )
                    self.assertEqual(s2["converted_layers"], [])
                    self.assertEqual(s2["skipped_layers"], [0, 20])
                    self.assertEqual(s2["backbone_status"], "skipped_local")

            # 3. Verify text-only backbone excluded vision and preserved sensitive tensors verbatim
            bb_reader = BackboneBundleReader(out_dir / BACKBONE_RELPATH, backend=backend)
            self.assertNotIn("model.visual.proj_in.weight", bb_reader.reader.tensors)
            prec = bb_reader.verify_sensitive_precision()
            self.assertGreaterEqual(prec["preserved_sensitive_tensors"], 8)

            # 4. Verify manifest + selective byte-offset loading + LDT residency manager
            manifest_path = out_dir / MANIFEST_RELPATH
            manifest = json.loads(manifest_path.read_text(encoding="utf8"))
            self.assertEqual(manifest["total_experts"], 8)
            self.assertIs(manifest["llama_cpp_dropin_compatible"], False)
            readme = render_hf_readme(manifest)
            self.assertIn("NOT a drop-in", readme)

            store = SelectiveExpertStore(manifest_path, root_dir=out_dir, backend=backend)
            ldt = LDTResidencyManager(store, capacity_per_layer=2, verify_sha256=True, experts_per_layer=4)
            adm0 = ldt.admit_layer(layer=0, scores=np.array([0.9, 0.1, 0.7, 0.2]), swap_budget=2)
            self.assertEqual(sorted(adm0["admitted"]), [0, 2])
            self.assertEqual(ldt.resident_ids(0), frozenset({0, 2}))

            # Swap budget 1 replaces expert 2 (0.1) with expert 1 (0.8), keeping expert 0 (0.9)
            adm1 = ldt.admit_layer(layer=0, scores=np.array([0.9, 0.8, 0.1, 0.2]), swap_budget=1)
            self.assertEqual(adm1["admitted"], [1])
            self.assertEqual(adm1["evicted"], [2])
            self.assertEqual(ldt.resident_ids(0), frozenset({0, 1}))

            # Unadmitted expert raises KeyError
            with self.assertRaises(KeyError):
                ldt.get_resident(0, 3)

            # Forward weighted mixture over resident experts works with exact soft-capped SwiGLU
            x = torch.randn((3, 64), dtype=torch.bfloat16)
            y = ldt.forward_resident_mixture(0, x, selected_experts=[0, 1], weights=[0.37, 0.63])
            self.assertEqual(tuple(y.shape), (3, 64))
            self.assertTrue(torch.isfinite(y).all())

    def test_ggml_c_backend_q4k_and_q6k_if_available(self):
        lib_dir = discover_ggml_lib_dir()
        if lib_dir is None:
            self.skipTest("compiled libggml-base not present in this CI environment")
        backend = GGMLCBackend(lib_dir=lib_dir)
        rng = np.random.default_rng(123)
        w_gu = (rng.standard_normal((64, 512)) * 0.04).astype(np.float32)
        w_dn = (rng.standard_normal((64, 1152)) * 0.04).astype(np.float32)

        for qt, max_rel in ((GGMLType.Q4_K, 0.10), (GGMLType.Q6_K, 0.03)):
            q_st = backend.quantize(w_gu, qt, n_threads=1)
            q_mt = backend.quantize(w_gu, qt, n_threads=4)
            self.assertEqual(q_st, q_mt)
            self.assertEqual(len(q_mt), ggml_nbytes(w_gu.shape, qt))
            dq = backend.dequantize(q_mt, w_gu.shape, qt)
            rel = float(np.linalg.norm(dq - w_gu) / np.linalg.norm(w_gu))
            self.assertLess(rel, max_rel)

        # Verify fallback types for down_proj ne0=1152 work cleanly in C GGML
        for qt, max_rel in ((GGMLType.Q5_0, 0.06), (GGMLType.Q8_0, 0.01)):
            raw = backend.quantize(w_dn, qt, n_threads=4)
            dq = backend.dequantize(raw, w_dn.shape, qt)
            rel = float(np.linalg.norm(dq - w_dn) / np.linalg.norm(w_dn))
            self.assertLess(rel, max_rel)

    def test_colab_notebook_schema_and_syntax(self):
        nb_path = Path(__file__).resolve().parent / "inkynano_colab_gguf.ipynb"
        self.assertTrue(nb_path.is_file(), f"missing notebook: {nb_path}")
        nb = json.loads(nb_path.read_text(encoding="utf8"))
        self.assertEqual(nb.get("nbformat"), 4)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 7)
        for cell in code_cells:
            src = "".join(cell["source"])
            ast.parse(src)


if __name__ == "__main__":
    unittest.main()
