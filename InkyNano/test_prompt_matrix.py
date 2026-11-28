import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from matrix_cases import hilbert_storyworld_cases, render, score, sudoku_cases, sudoku_solutions
from matrix_compare import REPLAY_SETTINGS, compare, compare_backends, historical_storyworld
from matrix_expert_bank import ExpertBank, FORMAT
from matrix_gguf import ARCH, LAYERS, PublishedComposition, pack, plan_array
from prompt_matrix import PromptController, features, plans, select
from gym_manifest import sha
from stage1_memory import MemoryGuard


class PromptMatrixTests(unittest.TestCase):
    def test_aggregate_storyworld_baseline_preserves_cache_and_planning_outcomes(self):
        receipt = {"benchmark": "hilbert-storyworld-canvas-v1", "storyworld": "world", "run_id": "run",
                   "runtime": {"model": "inky"}, "conveyor_16k_turn_totals": {
                       "total_prompt_tokens_in": 80, "total_cached_prompt_tokens": 20,
                       "total_uncached_prompt_tokens": 60}, "sized_long_context_planning": {
                           "32768": {"accepted_long_context_plan": False},
                           "16384": {"accepted_long_context_plan": True}}, "canvas_edition": {"browser": {
                               "status": "verified", "visited_scenes": 3, "verified_secret_endings": ["secret"]}}}
        summary = historical_storyworld(receipt)
        self.assertEqual(summary["conveyor_16k"]["cached_prompt_token_fraction"], 0.25)
        self.assertEqual([row["accepted_long_context_plan"] for row in summary["planning"]], [True, False])
        self.assertEqual(summary["verified_canvas"]["verified_secret_endings"], ["secret"])

    def test_global_prior_initialization_and_checkpoint_replay(self):
        prior = torch.linspace(-8, -1, 2 * 24 * 128).reshape(2, 24, 128)
        x = torch.ones((1, 384))
        for architecture in ("ldt", "trm"):
            model = PromptController(architecture, log_prior=prior)
            torch.testing.assert_close(model(x)[0], prior)
            with torch.no_grad():
                model.output.bias.fill_(100)
            checkpoint = {"architecture": architecture, "state": model.state_dict()}
            restored = PromptController.from_checkpoint(checkpoint)
            torch.testing.assert_close(restored(x)[0], prior + 0.5)
            original = PromptController(architecture)
            legacy = PromptController.from_checkpoint({"architecture": architecture, "state": original.state_dict()})
            torch.testing.assert_close(legacy(x), original(x))

    def test_plan_uses_only_experts_present_in_each_physical_layer(self):
        allowed = tuple(tuple(range(i, i + 10)) for i in range(LAYERS))
        scores = np.broadcast_to(np.arange(128), (LAYERS, 128))
        selected = select(scores, 8, allowed)
        self.assertEqual(selected, [list(ids[2:]) for ids in allowed])

    def test_goose_comparison_requires_same_prompt_and_weight_digest(self):
        base = {"case_id": "root/infill", "root": "root", "task": "storyworld", "mode": "infill",
                "model_sha256": "model", "prompt_sha256": "prompt", "capacity": 128,
                "input_tokens": 10, "output_tokens": 4, "decode_tps": 2.0}
        native = {**base, "policy": "native"}
        selected = {**base, "policy": "ldt", "capacity": 8, "decode_tps": 3.0}
        historical = {**base, "model_sha256": "another-model"}
        result = compare([native, selected], [historical])
        self.assertEqual([row["decode_tps_delta"] for row in result["paired"]], [1.0])
        self.assertIn({"baseline": "goose", "reason": "different model_sha256",
                       "case_ids": ["root/infill"]}, result["unmatched"])
        historical_context, = result["same_prompt_goose"]
        self.assertEqual(historical_context["goose_model_sha256"], "another-model")
        self.assertEqual(historical_context["decode_tps_delta"], 1.0)

    def test_isolated_comparison_uses_same_capacity_global_control(self):
        base = {"case_id": "root/outfill", "root": "root", "task": "sudoku", "mode": "outfill",
                "model_sha256": "model", "prompt_sha256": "prompt", "capacity": 8,
                "input_tokens": 20, "output_tokens": 5, "runtime_revision": "local",
                "expert_manifest_sha256": "isolated", "decode_tps": 2.0, "quality": {"correct": False}}
        global_control = {**base, "policy": "global"}
        trm = {**base, "policy": "trm", "decode_tps": 2.5, "quality": {"correct": True}}
        result = compare([global_control, trm], [])
        matched, = result["paired_groups"]
        self.assertEqual(matched["baseline"], "global")
        self.assertEqual(matched["root_mean_decode_tps_delta"], 0.5)
        self.assertEqual(matched["root_mean_correct_delta"], 1)
        mismatch = compare([{**global_control, "capacity": 16}, trm], [])
        self.assertEqual(mismatch["paired"], [])

    def test_backend_comparison_pairs_partial_runs_and_checks_replay_identity(self):
        cpu_spec = {field: None for field in REPLAY_SETTINGS}
        cpu_spec.update(command="benchmark", model_sha256="model", runtime_revision="revision",
                        n_gpu_layers=0, runtime_binary_sha256="cpu-bin", plans_sha256="plans")
        cuda_spec = {**cpu_spec, "runtime_binary_sha256": "cuda-bin", "n_gpu_layers": 28, "vram_limit_bytes": 1024}
        host = {"case_id": "a/infill", "policy": "ldt", "root": "a", "task": "sudoku", "mode": "infill",
                "prompt_sha256": "prompt", "model_sha256": "model", "runtime_revision": "revision",
                "expert_manifest_sha256": None, "capacity": 8, "storage_mode": "ram", "quality_evaluated": False,
                "input_tokens": 20, "output_tokens": 5, "prefill_tps": 2, "decode_tps": 3, "end_to_end_seconds": 20,
                "peak_rss_bytes": 200, "stop": "limit", "quality": {}, "content": "first"}
        device = {**host, "end_to_end_seconds": 10, "content": "second",
                  "cuda": {"gpu_layers_requested": 28, "vram_cap_bytes": 1024}}
        result = compare_backends([host, {**host, "case_id": "b/infill"}], [device], cpu_spec, cuda_spec)
        self.assertEqual(result["cpu_only"], [["b/infill", "ldt"]])
        pair, = result["paired"]
        self.assertEqual(pair["cpu_to_cuda_wall_time_ratio"], 2)
        self.assertFalse(pair["same_output"])
        different_length = compare_backends([host], [{**device, "output_tokens": 4}], cpu_spec, cuda_spec)
        self.assertNotIn("cpu_to_cuda_wall_time_ratio", different_length["paired"][0])
        with self.assertRaisesRegex(ValueError, "plans_sha256"):
            compare_backends([host], [device], cpu_spec, {**cuda_spec, "plans_sha256": "other"})
        with self.assertRaisesRegex(ValueError, "prompt_sha256"):
            compare_backends([host], [{**device, "prompt_sha256": "different"}], cpu_spec, cuda_spec)

    def test_sudoku_modes_share_one_root_but_never_render_the_solution(self):
        outfill, infill, _, _ = sudoku_cases(1, 1, 17)
        compact, *_ = sudoku_cases(1, 1, 17, compact=True)
        self.assertEqual(outfill["root"], infill["root"])
        self.assertEqual(sudoku_solutions(outfill["label"]["givens"], 2), [outfill["label"]["grid"]])
        modified = {**infill, "label": {"grid": [[9]]}}
        self.assertEqual(render(infill), render(modified))
        np.testing.assert_array_equal(features(infill), features(modified))
        self.assertFalse(np.array_equal(features(outfill), features(infill)))
        self.assertEqual((compact["id"], compact["label"]), (outfill["id"], outfill["label"]))
        rows = compact["prefix"].splitlines()[1:-1]
        self.assertEqual([[int(cell) for cell in row] for row in rows], compact["label"]["givens"])

    def test_scores_all_arc_test_grids_with_up_to_two_attempts(self):
        case = {"task": "arc-agi-2", "label": [[[1]], [[2]]]}
        result = score(case, '{"attempt_1":[[[1]],[[0]]],"attempt_2":[[[0]],[[2]]]}')
        self.assertTrue(result["correct"])
        self.assertEqual(result["test_pass_at_2"], [True, True])
        self.assertFalse(score(case, '{"attempt_1":[[[true]],[[2]]]}')["correct"])

    def test_compact_grid_is_a_separate_exact_match_score(self):
        case = {"task": "arc-agi-2", "label": [[[1, 0], [0, 1]]]}
        result = score(case, "```text\n10\n01\n```")
        self.assertTrue(result["exact_grid_match"])
        self.assertFalse(result["correct"])
        self.assertFalse(score(case, "here is my answer\n10\n01")["exact_grid_match"])
        self.assertFalse(score({**case, "label": case["label"] * 2}, "10\n01")["exact_grid_match"])

    def test_scores_native_text_after_a_finished_thinking_message(self):
        case = {"task": "sudoku", "label": {"grid": [[1, 2], [2, 1]]}}
        grid = '{"grid":[[1,2],[2,1]]}'
        thinking = f"<|message_model|><|content_thinking|>{grid}"
        self.assertFalse(score(case, thinking)["correct"])
        self.assertFalse(score(case, thinking + "<|end_message|>")["correct"])
        final = "<|message_model|><|content_text|>" + grid + "<|end_message|><|content_model_end_sampling|>"
        self.assertTrue(score(case, final)["correct"])
        self.assertTrue(score(case, thinking + "<|end_message|>" + final)["correct"])

    def test_hilbert_authoring_preserves_encounter_rules_without_answer_prose(self):
        world = {"plan": {"title": "World", "premise": "Premise", "cast": [], "acts": [["Act"]]},
                 "math": {"operators": {"H0": []}, "clues": {"c0": "Clue"}}, "scenes": [
                     {"id": "enc_01", "title": "First", "text": "Unseen prose", "options": [[0, 0, 0, 0, 1]]},
                     {"id": "enc_03", "title": "Second", "text": "Unseen prose", "options": [
                         [0, 0, 0, 0, 1], [0, 0, 0, 0, 2]]}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "world.json"
            path.write_text(json.dumps(world), encoding="utf8")
            cases = list(hilbert_storyworld_cases(path, 1, 1, 29))
        self.assertEqual(len(cases), 4)
        self.assertEqual(len({c["root"] for c in cases}), 2)
        case = next(c for c in cases if c["root"].endswith("03"))
        self.assertNotIn("Unseen prose", render(case))
        value = {"id": "enc_03", "title": "Second", "text": "New start.\n\nNew ending.", "options": [
            ["Choice 1", "He replies.", "H0", None, 1], ["Choice 2", "She replies.", "H0", "c0", 2],
            ["Choice 3", "They reply.", "H0", None, 1]]}
        self.assertTrue(score(case, json.dumps(value))["storyworld_structure_ok"])
        self.assertTrue(score(case, json.dumps(value))["storyworld_paragraph_target_ok"])
        value["options"][1][4] = 1
        self.assertFalse(score(case, json.dumps(value))["storyworld_structure_ok"])

    def test_matched_controllers_train_and_select_a_single_physical_set(self):
        case, *_ = sudoku_cases(1, 1, 17)
        x = torch.from_numpy(features(case)).unsqueeze(0)
        output = {}
        count = []
        for architecture in ("ldt", "trm"):
            torch.manual_seed(17)
            model = PromptController(architecture)
            prediction = model(x)
            self.assertEqual(tuple(prediction.shape), (1, 2, 24, 128))
            prediction.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            count.append(sum(p.numel() for p in model.parameters()))
            output[architecture] = prediction.detach().softmax(-1).numpy()
        self.assertEqual(count[0], count[1])
        self.assertLess(count[0], 250_000)
        uniform = np.full((1, 2, 24, 128), 1 / 128)
        for row in plans([case], uniform, output, ("digest", "commit")):
            selected = plan_array(row["resident_by_layer"])
            self.assertEqual(selected.shape, (24, row["capacity"]))
            self.assertEqual(row["offline_mass"]["decode"], row["capacity"] / 128)


@unittest.skipUnless(importlib.util.find_spec("gguf"), "pinned llama.cpp gguf-py is unavailable")
class PackedGGUFTests(unittest.TestCase):
    def test_published_teacher_maps_native_slots_back_to_original_experts(self):
        import gguf

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher, student = root / "teacher.gguf", root / "student.gguf"
            support = [list(range(layer, layer + 24, 2)) for layer in range(LAYERS)]
            path = root / "teacher.receipt.json"
            path.write_text(json.dumps({"model_sha256": "a" * 64, "bank_manifest_sha256": "b" * 64,
                                        "capacity_per_layer": 12, "resident_by_layer": support}), encoding="utf8")
            composition = PublishedComposition.read(path)
            mass = np.zeros((LAYERS, 128))
            mass[:, 11] = 1
            phases = {"decode": {"mass": mass.tolist()}}
            composition.expand_mass(phases)
            np.testing.assert_array_equal(np.argmax(phases["decode"]["mass"], axis=1),
                                          [ids[-1] for ids in support])

            writer = gguf.GGUFWriter(teacher, ARCH)
            writer.add_block_count(LAYERS)
            writer.add_expert_count(12)
            writer.add_expert_used_count(8)
            writer.add_string("inkynano.bank_manifest_sha256", composition.manifest_sha256)
            router = np.arange(12 * 32, dtype=np.float32).reshape(12, 32)
            weight = router.reshape(12, 1, 32)
            for layer in range(LAYERS):
                writer.add_tensor(f"blk.{layer}.ffn_gate_inp.weight", router)
                for projection in ("down_exps", "gate_up_exps"):
                    writer.add_tensor(f"blk.{layer}.ffn_{projection}.weight", weight)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            pack(gguf, teacher, student, [ids[-8:] for ids in support], 8 << 20, composition=composition)
            result = gguf.GGUFReader(student)
            tensors = {tensor.name: tensor for tensor in result.tensors}
            np.testing.assert_array_equal(tensors["blk.23.ffn_gate_inp.weight"].data, router[-8:])
            np.testing.assert_array_equal(tensors["blk.23.ffn_down_exps.weight"].data, weight[-8:])
            del tensors, result

    def test_builds_ram_model_from_backbone_and_only_selected_shards(self):
        import gguf

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def finish(writer, path):
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}

            path = root / "backbone.gguf"
            writer = gguf.GGUFWriter(path, ARCH)
            writer.add_block_count(LAYERS)
            writer.add_expert_count(128)
            writer.add_expert_used_count(8)
            router = np.arange(128 * 32, dtype=np.float32).reshape(128, 32)
            writer.add_tensor("token_embd.weight", np.ones((4, 32), dtype=np.float32))
            for layer in range(LAYERS):
                writer.add_tensor(f"blk.{layer}.ffn_gate_inp.weight", router)
                writer.add_tensor(f"blk.{layer}.exp_probs_b.bias", router[:, 0].copy())
            backbone = finish(writer, path)

            rows = []
            qtype = gguf.GGMLQuantizationType.Q4_K
            # The packer transports native blocks without interpreting their contents.
            weight = np.arange(128 * 144, dtype=np.uint16).reshape(128, 1, 144).astype(np.uint8)
            for parity in (0, 1):
                ids = list(range(parity, 16, 2))
                path = root / f"experts-{parity}.gguf"
                writer = gguf.GGUFWriter(path, ARCH)
                for layer in range(LAYERS):
                    for projection in ("down_exps", "gate_up_exps"):
                        writer.add_tensor(f"blk.{layer}.ffn_{projection}.weight", weight[ids],
                                          raw_dtype=qtype)
                artifact = finish(writer, path)
                rows.extend({**artifact, "layer": layer, "expert_ids": ids} for layer in range(LAYERS))
            rows.append({"path": "unneeded.gguf", "bytes": 1, "sha256": "1" * 64,
                         "layer": 0, "expert_ids": [127]})
            manifest = root / "bank.json"
            manifest.write_text(json.dumps({"format": FORMAT, "source_model_sha256": "a" * 64,
                                            "backbone": backbone, "shards": rows}), encoding="utf8")
            bank = ExpertBank(manifest)
            selected = [list(range(8)) for _ in range(LAYERS)]
            target = root / "compact.gguf"
            receipt = bank.pack(gguf, target, selected, 8 << 20, MemoryGuard())
            self.assertEqual(receipt["expert_manifest_sha256"], sha(manifest))
            result = gguf.GGUFReader(target)
            self.assertEqual(result.fields[f"{ARCH}.expert_count"].contents(), 8)
            tensors = {tensor.name: tensor for tensor in result.tensors}
            for layer in range(LAYERS):
                np.testing.assert_array_equal(tensors[f"blk.{layer}.ffn_gate_inp.weight"].data, router[:8])
                np.testing.assert_array_equal(tensors[f"blk.{layer}.ffn_down_exps.weight"].data, weight[:8])
            del tensors, result

            bad = [{**row, "sha256": "0" * 64} if row["path"] == "experts-0.gguf" else row for row in rows]
            manifest.write_text(json.dumps({"format": FORMAT, "source_model_sha256": "a" * 64,
                                            "backbone": backbone, "shards": bad}), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                ExpertBank(manifest).pack(gguf, root / "bad.gguf", selected, 8 << 20, MemoryGuard())
            self.assertFalse((root / "bad.gguf").exists())

    def test_copies_exact_quantized_experts_and_router_rows_only(self):
        import gguf

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, compact = root / "source.gguf", root / "compact.gguf"
            selection = [list(range(i, i + 16, 2)) for i in range(LAYERS)]
            writer = gguf.GGUFWriter(model, ARCH)
            writer.add_block_count(LAYERS)
            writer.add_expert_count(128)
            writer.add_expert_used_count(8)
            writer.add_tensor("token_embd.weight", np.ones((4, 32), dtype=np.float32))
            router = np.arange(128 * 32, dtype=np.float32).reshape(128, 32)
            weight = gguf.quants.quantize(router.reshape(128, 1, 32), gguf.GGMLQuantizationType.Q4_0)
            for layer in range(LAYERS):
                writer.add_tensor(f"blk.{layer}.ffn_gate_inp.weight", router)
                writer.add_tensor(f"blk.{layer}.exp_probs_b.bias", router[:, 0].copy())
                for bank in ("down_exps", "gate_up_exps"):
                    writer.add_tensor(f"blk.{layer}.ffn_{bank}.weight", weight,
                                      raw_dtype=gguf.GGMLQuantizationType.Q4_0)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            receipt = pack(gguf, model, compact, selection, 8 << 20)
            result = gguf.GGUFReader(compact)
            self.assertEqual(result.fields[f"{ARCH}.expert_count"].contents(), 8)
            self.assertLess(receipt["compact_gguf_bytes"], model.stat().st_size)
            tensors = {tensor.name: tensor for tensor in result.tensors}
            np.testing.assert_array_equal(tensors["token_embd.weight"].data, np.ones((4, 32)))
            for layer in range(LAYERS):
                np.testing.assert_array_equal(tensors[f"blk.{layer}.ffn_gate_inp.weight"].data,
                                              router[selection[layer]])
                np.testing.assert_array_equal(tensors[f"blk.{layer}.exp_probs_b.bias"].data,
                                              router[selection[layer], 0])
                np.testing.assert_array_equal(tensors[f"blk.{layer}.ffn_down_exps.weight"].data,
                                              weight[selection[layer]])
            # Release Windows file mappings before TemporaryDirectory cleanup.
            del tensors, result


if __name__ == "__main__":
    unittest.main()
