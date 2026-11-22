"""Measure attention write reconstruction and endpoint readout by rank and site."""
import argparse
from collections import defaultdict
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
from statistics import fmean
import sys

from experiments.nan_carry63.algebra import account, basis, continuous, margins, self_test
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bf16, bound, digest, joined, write_new

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
READPATH = REPO / "experiments/nan_readpath37"
ARCHIVE = REPO / "runs/nan_comp14"
FIT = REPO / "runs/nan_ranksite13"
MANIFEST = HERE / "SOURCE_MANIFEST.json"
SITES = ("residual_after_layer", "o_proj_input", "o_proj_output")
FIT_SOURCE = REPO / "experiments/nan_ranksite13/SOURCE_MANIFEST.json"


def seal():
    files = [HERE / name for name in (".gitattributes", "PROTOCOL.md", "__init__.py", "registration.json", "run.py")]
    files += [FIT_SOURCE, READPATH / "registration.json"]
    files += [REPO / name for name in (
        "experiments/nan_carry63/algebra.py", "experiments/nan_lattice53/run.py",
        "experiments/nan_phase35/run.py", "experiments/nan_readpath37/run.py",
        "experiments/nan_ram28/resources.py", "experiments/nan_ranksite13/resources.py",
        "reports/nanbeige_bpd_gym/nan_inverse65_results.json")]
    return {"experiment": "NAN-ATTNSPAN-67",
            "files_sha256": {p.relative_to(REPO).as_posix(): digest(p) for p in files}}


def factors(path, expected, keys, rank, np):
    with np.load(BytesIO(bound(path, expected)), allow_pickle=False) as obj:
        v, u = (obj[k].astype(np.float64) for k in keys)
    if ((v.shape, u.shape) != ((6144, rank), (rank, 3072))
            or not np.isfinite(v).all() or not np.isfinite(u).all()
            or np.linalg.matrix_rank(v) != rank or np.linalg.matrix_rank(u) != rank):
        raise ValueError("Attention factors differ in dimension, rank or finiteness")
    return v, u


def read_banks(backend, layer, config, base, np):
    folder = REPO / "runs/nan_gym05" / backend / "linear_baselines"
    manifest = json.loads(bound(folder / "manifest.json", config["linear_manifest"][backend]))
    entry, = [r for r in manifest["receipts"] if (r["layer_zero_based"], r["module"]) == (layer, "o_proj")]
    if (entry["backend"], entry["checkpoint_revision"], entry["output_site"], entry["trained_BPD"],
            entry["trained_VPD"]) != (backend, base["revision"], "o_proj_output", False, False):
        raise ValueError("The randomized SVD control differs from its native condition")
    path, pin = folder / entry["file"], entry["file_sha256"]
    saved = {f"svd{rank}": factors(path, pin, (f"V{rank}", f"U{rank}"), rank, np) for rank in (8, 16)}
    v, u = saved["svd8"]
    saved["svd4"] = v[:, :4], u[:4]
    if sorted(int(k[3:]) for k in saved) != config["svd_ranks"]:
        raise ValueError("SVD rank roster changed")
    if backend == "bf16" and layer == 21:
        bound(FIT_SOURCE, config["ranksite_source"])
        for phase in (1, 2):
            name = f"t{phase}_layer21_o_proj_rank8"
            evidence = json.loads(bound(FIT / "audit" / f"{name}.json", config["ranksite_audits"][str(phase)]))
            parent = FIT / "conditions" / name
            receipt = json.loads(bound(parent / "receipt.json", evidence["fit_receipt_sha256"]))
            if (evidence["status"], evidence["source_manifest_sha256"], evidence["backend"], evidence["passes"],
                    receipt["status"], receipt["module"], receipt["rank"], receipt["layer_zero_based"],
                    receipt["optimizer_steps_completed"], receipt["target_parameter_sha256_unchanged"],
                    receipt["initial_svd_file_sha256"], receipt["native_all_one_full_head_bit_exact"]) != (
                    "passed", config["ranksite_source"], backend, phase, "completed", "o_proj", 8, 21, 8,
                    base["backends"][backend]["target"], pin, True):
                raise ValueError("Attention fit does not match its completed independent audit")
            for field, steps in (("initial_checkpoint", 0), ("final_checkpoint", 8)):
                ckpt = receipt[field]
                assert ckpt["optimizer_steps_completed"] == steps
                values = factors(parent / ckpt["file"], ckpt["sha256"], ("V", "U"), 8, np)
                if steps == 0:
                    assert all(np.array_equal(a, b) for a, b in zip(values, saved["svd8"], strict=True))
                else:
                    assert ckpt["sha256"] == evidence["checkpoint_sha256"]
                    saved[f"attn_bpd_t{phase}"] = values
    return saved


def read_archive(backend, layers, base, roster, head, check, np):
    folder = ARCHIVE / backend
    pin = base["backends"][backend]
    audit = json.loads(bound(ARCHIVE / "audits" / f"{backend}.json", pin["audit"]))
    manifest = json.loads(bound(folder / "manifest.json", pin["manifest"]))
    if ((audit["status"], audit["worker_status"], audit["missing_contexts"], audit["contexts_checked"])
            != ("passed", "completed", 0, 240) or len(audit["records"]) != 240
            or manifest["checkpoint_revision"] != base["revision"]
            or manifest["target_parameter_sha256"] != pin["target"]):
        raise ValueError("COMP14 archive lacks its expected native audit")
    grouped = defaultdict(dict)
    max_error = 0.
    expected = {(p, l, s) for p in (1, 2) for l in layers for s in SITES}
    widths = dict(zip(SITES, (3072, 6144, 3072), strict=True))
    for slot, row in roster.items():
        check()
        record = audit["records"][slot]
        case = json.loads(bound(folder / f"case_{slot:04d}.json", record["case_sha256"]))
        if any(record[k] != row[k] or case[k] != row[k] for k in ("id", "root", "split", "arm", "world")):
            raise ValueError("Native archive identity changed")
        filename = f"case_{slot:04d}_states.npz"
        if (case["state_file"], case["state_file_sha256"]) != (filename, record["state_sha256"]):
            raise ValueError("Native capture differs from its archive seal")
        entries = [(x["loop"], x["layer_zero_based"], x["site"], x["array_key"]) for x in case["state_index"]
                   if x["layer_zero_based"] in layers and x["site"] in SITES]
        if len(entries) != len(expected) or {x[:3] for x in entries} != expected:
            raise ValueError("Attention capture omitted a registered site")
        cells = {}
        with np.load(BytesIO(bound(folder / filename, record["state_sha256"])), allow_pickle=False) as pack:
            for p, l, s, name in entries:
                value = pack[name]
                if (value.dtype != np.float32 or value.shape != (widths[s],) or not np.isfinite(value).all()
                        or not np.array_equal(bf16(value, np), value)):
                    raise ValueError("Expected finite BF16 attention archive vectors")
                cells[p, l, s] = value
        assert [v["loop"] for v in case["initial_phases"]] == [1, 2]
        for item in case["initial_phases"]:
            score = item["score"]
            logits = {int(k): v for k, v in score["selected_logits"].items()}
            logits[score["free_top_token_id"]] = score["free_top_token_logit"]
            replay = head.logits(cells[item["loop"], 21, SITES[0]], list(logits))
            max_error = max(max_error, float(np.max(abs(replay - list(logits.values())))))
        group = grouped[row["split"], row["arm"], row["root"]]
        if row["world"] in group:
            raise ValueError("A twin world was repeated")
        group[row["world"]] = (row["answer"], case["initial_prompt_ids_sha256"], cells)
    assert len(grouped) * 2 == len(roster) and max_error <= base["native_logit_abs_tol"]
    return grouped, max_error


def make_pair(worlds, layer, ids, head, erased, np):
    assert set(worlds) == {"A", "B"}
    gold_a, hash_a, a = worlds["A"]
    gold_b, hash_b, b = worlds["B"]
    assert gold_a != gold_b
    tokens = [ids[gold_a], ids[gold_b]]
    direction = head.gamma.astype(np.float64) * (head.row(tokens[0]).astype(np.float64) - head.row(tokens[1]))
    norm = np.linalg.norm(direction)
    assert norm > 0
    if erased and (hash_a != hash_b or any(not np.array_equal(a[k], b[k]) for k in a)):
        raise ValueError("Erased control worlds have different source data")
    phase = []
    for p in (1, 2):
        h = np.stack([a[p, layer, SITES[0]], b[p, layer, SITES[0]]]).astype(np.float64)
        x, y = (a[p, layer, key].astype(np.float64) - b[p, layer, key] for key in SITES[1:])
        alpha = float(np.mean(1 / np.sqrt(np.mean(h*h, axis=1) + float(head.eps))))
        scaled, _, _ = continuous(h, direction, head.eps, np)
        np.testing.assert_allclose(scaled, direction*alpha, atol=1e-13, rtol=1e-13)
        a_margin, b_margin = margins(head, h, tokens, np)
        phase.append({"h": h, "x": x, "y": y, "alpha": alpha, "direction": scaled,
                      "native_swing": float(a_margin - b_margin), "direct_write": float(y @ scaled)})
    gain = phase[1]["y"]*phase[1]["alpha"] - phase[0]["y"]*phase[0]["alpha"]
    return {"tokens": tokens, "phase": phase, "cal_write": gain, "cal_direction": direction / norm}


def cal_bases(y, w, rank, np):
    # Economy QR gives the left singular vectors without forming a hidden-width-square matrix.
    q_y, r_y = np.linalg.qr(y.T)
    q_w, r_w = np.linalg.qr(w.T)
    energy = np.linalg.svd(r_y, full_matrices=False)[0]
    head = np.linalg.svd(r_w, full_matrices=False)[0]
    outputs = {"cal_energy4": q_y @ energy[:, :rank], "cal_head_only4": q_w @ head[:, :rank]}
    for name, label in (("cal_readout4", w), ("cal_shuffled4", np.roll(w, 1, axis=0))):
        if name == "cal_shuffled4":
            q, r = np.linalg.qr(label.T)
        else:
            q, r = q_w, r_w
        # Right multiplication by q.T is an orthonormal embedding.
        assert q.shape[1] == w.shape[0]
        u, s, _ = np.linalg.svd(r_y @ r.T, full_matrices=False)
        if s[rank-1] <= 1e-10:
            raise ValueError("The registered CAL cross-covariance lost rank")
        outputs[name] = q_y @ u[:, :rank]
    for q in outputs.values():
        np.testing.assert_allclose(q.T @ q, np.eye(rank), atol=1e-12, rtol=0)
    return outputs


def known_case(np):
    self_test(np)
    y = np.eye(8) * np.arange(1., 9.)[:, None]
    w = np.flip(np.eye(8), 0) * np.arange(8., 0., -1.)[:, None]
    output = cal_bases(y, w, 4, np)
    direct = np.linalg.svd(y.T @ w, full_matrices=False)[0][:, :4]
    close = output["cal_readout4"]
    np.testing.assert_allclose(close @ close.T, direct @ direct.T, rtol=0, atol=1e-12)
    for x in output.values():
        np.testing.assert_allclose(x.T @ x, np.eye(4), rtol=0, atol=1e-12)


def score_pair(pair, spans, weights, head, np):
    points = pair["phase"]
    conditions = {}
    for name, q in spans.items():
        rows = []
        for p in points:
            linear, finite, projection, _ = account(head, p["h"], p["y"], q, pair["tokens"],
                                                    p["direction"], p["native_swing"], np)
            terms = {"linear_in_span": linear["in_span"], "finite_in_span": finite["in_span"],
                     "finite_off_span": finite["off_span"], "finite_other": finite["effective_carry"],
                     "projected_energy": float(projection @ projection)}
            if name in weights:
                v, u = weights[name]
                predicted = (p["x"] @ v) @ u
                terms["predicted_linear"] = float(predicted @ p["direction"])
                terms["prediction_write_error"] = float(np.linalg.norm(p["y"] - predicted)**2)
            rows.append(terms)
        conditions[name] = rows
    return {"native": [p["native_swing"] for p in points], "direct_write": [p["direct_write"] for p in points],
            "write_energy": [float(p["y"] @ p["y"]) for p in points], "conditions": conditions}


def summarize(rows):
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    actual = [b-a for a,b in (r["direct_write"] for r in rows)]
    native = [b-a for a,b in (r["native"] for r in rows)]
    summed = sum(actual)
    denominator = sum(x*x for x in actual)
    energy = sum(r["write_energy"][1] for r in rows)
    quantities = {}
    for name in rows[0]["conditions"]:
        record = [r["conditions"][name] for r in rows]
        gains = {key: [b[key] - a[key] for a,b in record]
                 for key in ("linear_in_span", "finite_in_span", "finite_off_span", "finite_other")}
        direct = gains["linear_in_span"]
        result = {
            "pass2_projected_write_energy_fraction": ratio(sum(b["projected_energy"] for _,b in record), energy),
            "mean_signed_gain": {k: fmean(x) for k,x in gains.items()},
            "mean_abs_finite_in_span_gain": fmean(map(abs, gains["finite_in_span"])),
            "linear_gain_fraction_of_actual": ratio(sum(direct), summed),
            "linear_gain_squared_error_vs_zero": ratio(sum((x-y)**2 for x,y in zip(direct, actual)), denominator),
            "linear_gain_matching_sign_roots": sum(x*y > 0 for x,y in zip(direct, actual)),
        }
        if "predicted_linear" in record[0][0]:
            predicted = [b["predicted_linear"] - a["predicted_linear"] for a,b in record]
            result["predictor"] = {
                "mean_signed_gain": fmean(predicted),
                "linear_gain_squared_error_vs_zero": ratio(sum((x-y)**2 for x,y in zip(predicted, actual)), denominator),
                "pass2_relative_write_squared_error": ratio(sum(b["prediction_write_error"] for _,b in record), energy),
            }
        quantities[name] = result
    return {"roots": len(rows), "mean_signed_native_margin_gain": fmean(native),
            "mean_signed_full_attention_numerator_gain": fmean(actual), "spans": quantities}


def evaluate(backend, base, cfg, archive, labels, head, output, check, np):
    private = {}
    result = {}
    for layer in cfg["layers"]:
        banks = read_banks(backend, layer, cfg, base, np)
        spans = {name: basis(u, np) for name, (_,u) in banks.items()}
        pairs = {k: make_pair(worlds, layer, labels, head, k[1]=="focused_erased", np)
                 for k,worlds in sorted(archive.items())}
        cal = [p for (split, arm, _), p in pairs.items() if (split, arm) == ("calibration", "focused")]
        assert len(cal) == base["roots"]["calibration"] == 8 and cfg["cal_rank"] == 4
        spans.update(cal_bases(np.stack([r["cal_write"] for r in cal]),
                              np.stack([r["cal_direction"] for r in cal]), cfg["cal_rank"], np))
        for seed in cfg["random_seeds"]:
            q = basis(np.random.default_rng(seed).normal(size=(8, base["width"])), np)
            spans[f"random4_{seed}"] = q[:, :4]
            spans[f"random8_{seed}"] = q
        grouped = defaultdict(list)
        local = []
        for (split, arm, root), pair in pairs.items():
            check()
            row = score_pair(pair, spans, banks, head, np)
            if arm == "focused_erased":
                assert all(x == 0 for k in ("native", "direct_write", "write_energy") for x in row[k])
                assert all(x == 0 for p in row["conditions"].values() for c in p for x in c.values())
            grouped[split, arm].append(row)
            local.append({"root": root, "split": split, "arm": arm, **row})
        assert {k: len(v) for k,v in grouped.items()} == {
            (split, arm): count for split,count in base["roots"].items() for arm in base["arms"]}
        result[str(layer)] = {"/".join(k): summarize(rows) for k,rows in grouped.items()}
        private[str(layer)] = local
    write_new(output / f"{backend}_private.json", private)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if not __debug__:
        cli.error("Enable Python assertions")
    if args.freeze:
        write_new(MANIFEST, seal())
        return
    if not args.snapshot or not args.output or json.loads(MANIFEST.read_bytes()) != seal():
        cli.error("Require a frozen source and explicit snapshot and output")
    cfg = json.loads((HERE / "registration.json").read_bytes())
    base = json.loads((READPATH / "registration.json").read_bytes())
    if args.snapshot.name != cfg["revision"] or base["revision"] != cfg["revision"]:
        raise ValueError("Archive and checkpoint revision differ")
    with lease(cfg) as (check, samples):
        assert (version("numpy"), version("tokenizers")) == ("2.5.5", "0.20.3")
        import numpy as np
        from tokenizers import Tokenizer

        known_case(np)
        bound(args.snapshot / "tokenizer.json", base["snapshot_pins"]["tokenizer.json"])
        tokenizer = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
        bare = {label: tokenizer.encode(label, add_special_tokens=False).ids for label in "ABCDEFGHIJKLMNOP"}
        assert all(len(v) == 1 for v in bare.values()) and len({v[0] for v in bare.values()}) == 16
        labels = {k: v[0] for k,v in bare.items()}
        roster, _ = joined(base)
        args.output.mkdir(parents=True, exist_ok=False)
        aggregate = {}
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            for backend in base["backends"]:
                archive, error = read_archive(backend, cfg["layers"], base, roster, head, check, np)
                layers = evaluate(backend, base, cfg, archive, labels, head, args.output, check, np)
                aggregate[backend] = {"native_replay_max_abs_error": error, "condition_records": len(roster),
                                      "sampled_states": len(roster)*2*len(cfg["layers"]), "layers": layers}
                print(backend, aggregate[backend]["sampled_states"], flush=True)
        assert not any(k == "torch" or k.startswith("torch.") for k in sys.modules)
    write_new(args.output / "aggregate.json", {"experiment": cfg["experiment"], "status": "completed",
        "evidence": "descriptive_observational", "source_manifest_sha256": digest(MANIFEST),
        "formal_comparisons": 0, "new_pretrained_forwards": 0, "sealed_test_accessed": False,
        "random_seeds": cfg["random_seeds"], "independent_roots": base["roots"], "per_backend": aggregate,
        "resources": {"max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
                      "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples)}})


if __name__ == "__main__":
    main()
