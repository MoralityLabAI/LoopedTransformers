"""Audit calibrated and raw phase lenses on immutable COMP14 archives."""
import argparse
from collections import defaultdict
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import sys

from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, joined, write_new


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "SOURCE_MANIFEST.json"
ARCHIVE = REPO / "runs/nan_comp14"
READPATH = REPO / "experiments/nan_readpath37"
FINAL = (2, 21)
DEPENDENCIES = (
    "experiments/nan_readpath37/SOURCE_MANIFEST.json",
    "experiments/nan_readpath37/registration.json",
    "experiments/nan_readpath37/run.py",
    "experiments/nan_phase35/run.py",
    "experiments/nan_ram28/resources.py",
    "experiments/nan_ranksite13/resources.py",
)


def seal():
    sources = [HERE / name for name in (".gitattributes", "PROTOCOL.md", "__init__.py", "registration.json", "run.py")]
    sources += [REPO / path for path in DEPENDENCIES]
    return {"experiment": "NAN-LENS-40",
            "files_sha256": {path.relative_to(REPO).as_posix(): digest(path) for path in sources}}


def read_cases(backend, source_cfg, selected, stages, tokens, head, check, np):
    pin = source_cfg["backends"][backend]
    audit = json.loads(bound(ARCHIVE / "audits" / f"{backend}.json", pin["audit"]))
    location = ARCHIVE / backend
    manifest = json.loads(bound(location / "manifest.json", pin["manifest"]))
    if (audit["status"], audit["worker_status"], audit["missing_contexts"]) != ("passed", "completed", 0):
        raise ValueError("Archive must have a complete independent audit")
    if len(audit["records"]) != audit["contexts_checked"] or audit["contexts_checked"] != 240:
        raise ValueError("Archive audit roster mismatch")
    if (manifest["checkpoint_revision"], manifest["target_parameter_sha256"]) != (
            source_cfg["revision"], pin["target"]):
        raise ValueError("Archive checkpoint or target mismatch")

    metadata, residuals, owners = [], defaultdict(list), {}
    max_error = 0.0
    keys = ("id", "root", "split", "arm", "world")
    for slot, source in selected.items():
        check()
        verified = audit["records"][slot]
        case = json.loads(bound(location / f"case_{slot:04d}.json", verified["case_sha256"]))
        if any(verified[k] != source[k] or case[k] != source[k] for k in keys):
            raise ValueError("Archive case roster mismatch")
        state_name = f"case_{slot:04d}_states.npz"
        if (case["state_file"], case["state_file_sha256"]) != (state_name, verified["state_sha256"]):
            raise ValueError("Archive state binding mismatch")
        index = {(row["loop"], row["layer_zero_based"]): row["array_key"]
                 for row in case["state_index"] if row["site"] == "residual_after_layer"}
        if set(index) != set(stages):
            raise ValueError("Missing or unexpected phase/layer sites")
        with np.load(BytesIO(bound(location / state_name, verified["state_sha256"])),
                     allow_pickle=False) as states:
            for stage in stages:
                state = states[index[stage]].astype(np.float32)
                if state.shape != (head.width,) or not np.isfinite(state).all():
                    raise ValueError("Invalid archived residual")
                residuals[stage].append(state)

        native = {}
        for phase, captured in enumerate(case["initial_phases"], start=1):
            if captured["loop"] != phase:
                raise ValueError("Archive phase order mismatch")
            score = captured["score"]
            winner = score["free_top_token_id"]
            observed = {int(token): value for token, value in score["selected_logits"].items()}
            observed[winner] = score["free_top_token_logit"]
            replay = head.logits(residuals[phase, 21][-1], list(observed))
            max_error = max(max_error, float(np.max(abs(replay - list(observed.values())))))
            oracle_correct = verified["first_token_correct"][phase - 1]
            if type(oracle_correct) is not int or oracle_correct not in (0, 1):
                raise ValueError("Invalid native oracle verdict")
            native[phase] = {
                "winner_support_index": tokens.index(winner) if winner in tokens else None,
                "oracle_correct": oracle_correct,
            }
        if len(native) != 2 or max_error > source_cfg["native_logit_abs_tol"]:
            raise ValueError("Native layer-21 logit replay failed")
        row = {k: source[k] for k in keys}
        row.update(answer=source["answer"], native=native)
        metadata.append(row)
        if source["arm"] == "focused_erased":
            root = source["root"]
            prompt_digest = case["initial_prompt_ids_sha256"]
            if owners.setdefault(root, prompt_digest) != prompt_digest:
                raise ValueError("Erased counterfactual inputs are not identical")
    return metadata, {stage: np.stack(states) for stage, states in residuals.items()}, max_error


def center_rows(scores):
    return scores - scores.mean(axis=1, keepdims=True)


def features(residual, omitted, np):
    x = residual.astype(np.float64)
    if omitted is not None:
        x[:, omitted] = 0
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if (norm == 0).any():
        raise ValueError("Zero feature norm")
    return x / norm


def lenses(residual, raw, final, train, cfg, np):
    """Fit the correction against model logits with an explicit CAL row mask."""
    delta = final[train] - raw[train]
    shift = delta.mean(axis=0)
    result = {"raw": raw, "cal_mean_shift": raw + shift}
    for name, omitted in (("cal_linear", None),
                          ("cal_linear_without_2279", cfg["omitted_feature_coordinate"])):
        x = features(residual, omitted, np)
        x -= x[train].mean(axis=0)
        train_x = x[train]
        kernel = train_x @ train_x.T + cfg["ridge_penalty"] * np.eye(len(train_x))
        alpha = np.linalg.solve(kernel, delta - shift)
        result[name] = raw + shift + (x @ train_x.T) @ alpha
    return result


def strict_winners(scores, np):
    winner = scores.argmax(axis=1)
    tied = (scores == scores.max(axis=1, keepdims=True)).sum(axis=1) != 1
    return np.where(tied, -1, winner).tolist()


def log_softmax(scores, np):
    shifted = scores - scores.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def replay_native_group(rows, phase):
    native = [row["native"][phase] for row in rows]
    return {"registered": len(native),
            "winner_in_bare_support": sum(row["winner_support_index"] is not None for row in native),
            "oracle_correct": sum(row["oracle_correct"] for row in native)}


def summarize(rows, predictions, final, support, groups, np):
    target = strict_winners(final, np)
    gold = [support.index(row["answer"]) for row in rows]
    final_logp = log_softmax(final, np)
    probabilities = np.exp(final_logp)
    raw = strict_winners(predictions["raw"], np)
    estimates, private = {}, {}
    for lens, scores in predictions.items():
        choice = strict_winners(scores, np)
        kl = (probabilities * (final_logp - log_softmax(scores, np))).sum(axis=1)
        if not np.isfinite(scores).all() or not np.isfinite(kl).all() or float(kl.min()) < -1e-12:
            raise ValueError("Invalid lens scores")
        private[lens] = {"centered_logits": scores.tolist(), "choice_index": choice, "restricted_kl": kl.tolist()}
        estimates[lens] = {}
        for (split, arm), indices in groups.items():
            twins = defaultdict(list)
            for i in indices:
                twins[rows[i]["root"]].append(i)
            if not all({rows[i]["world"] for i in twin} == {"A", "B"} and len(twin) == 2
                       for twin in twins.values()):
                raise ValueError("Expected one matched world pair per root and arm")
            comparable = [i for i in indices if choice[i] >= 0 and raw[i] >= 0]
            estimates[lens][f"{split}/{arm}"] = {
                "roots": len(twins), "worlds": len(indices),
                "mean_restricted_kl": float(np.maximum(kl[indices], 0).mean()),
                "restricted_final_top1_agreement": sum(choice[i] >= 0 and choice[i] == target[i] for i in indices),
                "restricted_fidelity_unknown": sum(choice[i] < 0 or target[i] < 0 for i in indices),
                "restricted_oracle_correct": sum(choice[i] == gold[i] for i in indices),
                "restricted_both_world_roots": sum(all(choice[i] == gold[i] for i in twin) for twin in twins.values()),
                "restricted_oracle_unknown": sum(choice[i] < 0 for i in indices),
                "oracle_repairs_vs_raw": sum(choice[i] == gold[i] and raw[i] != gold[i] for i in comparable),
                "oracle_regressions_vs_raw": sum(choice[i] != gold[i] and raw[i] == gold[i] for i in comparable),
                "oracle_paired_unknown": len(indices) - len(comparable),
            }
            if arm == "focused_erased":
                estimates[lens][f"{split}/{arm}"]["exact_erased_twin_score_matches"] = sum(
                    np.array_equal(scores[i], scores[j]) for i, j in twins.values()
                )
    return estimates, private


def run_backend(backend, cfg, source_cfg, roster, support, tokens, head, output, check, np):
    stages = [(p, layer) for p in cfg["passes"] for layer in cfg["layers_zero_based"]]
    rows, residual, closure = read_cases(backend, source_cfg, roster, stages, tokens, head, check, np)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["split"], row["arm"]].append(i)
    if {key: len(idx) for key, idx in groups.items()} != {
            (split, arm): 2 * count for split, count in source_cfg["roots"].items()
            for arm in source_cfg["arms"]}:
        raise ValueError("Incorrect source-unit coverage")
    train = [i for i, row in enumerate(rows)
             if row["split"] == "calibration" and row["arm"] in cfg["training_arms"]]
    cal_roots = {rows[i]["root"] for i in train}
    dev_roots = {row["root"] for row in rows if row["split"] == "development"}
    if (cal_roots & dev_roots or len(cal_roots) != source_cfg["roots"]["calibration"]
            or len(dev_roots) != source_cfg["roots"]["development"]
            or len(train) != 2 * len(cal_roots) * len(cfg["training_arms"])):
        raise ValueError("Invalid CAL and DEV root partition")
    final = center_rows(head.logits(residual[FINAL], tokens).astype(np.float64))
    metrics, private = {}, {}
    for stage in stages:
        check()
        phase, layer = stage
        raw = center_rows(head.logits(residual[stage], tokens).astype(np.float64))
        predictions = {"raw": raw} if stage == FINAL else lenses(residual[stage], raw, final, train, cfg, np)
        name = f"pass{phase}_layer{layer}"
        metrics[name], private[name] = summarize(rows, predictions, final, support, groups, np)
    write_new(output / f"{backend}_private.json", {"rows": rows, "site_scores": private})
    return {"native_closure_max_abs_logit": closure, "cal_fit_worlds_per_site": len(train),
            "sites": metrics, "native_unrestricted": {
                f"pass{phase}": {
                    f"{split}/{arm}": replay_native_group([rows[i] for i in indices], phase)
                    for (split, arm), indices in groups.items()
                }
                for phase in cfg["passes"]
            }}


def main():
    if not __debug__:
        raise RuntimeError("Run with verification assertions enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze:
        write_new(MANIFEST, seal())
        print(digest(MANIFEST))
        return
    if not (args.snapshot and args.output):
        parser.error("--snapshot and --output are required for a run")
    if json.loads(MANIFEST.read_bytes()) != seal():
        raise ValueError("Lens source manifest mismatch")
    cfg = json.loads((HERE / "registration.json").read_bytes())
    source_cfg = json.loads((READPATH / "registration.json").read_bytes())
    if args.snapshot.name != source_cfg["revision"]:
        raise ValueError("Wrong snapshot revision")
    with lease(cfg) as (check, samples):
        if version("numpy") != "2.5.5" or version("tokenizers") != "0.20.3":
            raise ValueError("Unpinned numerical or tokenizer runtime")
        import numpy as np
        from tokenizers import Tokenizer

        bound(args.snapshot / "tokenizer.json", source_cfg["snapshot_pins"]["tokenizer.json"])
        tokenizer = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
        support = cfg["label_support"]
        encoded = [tokenizer.encode(label, add_special_tokens=False).ids for label in support]
        if any(len(ids) != 1 for ids in encoded) or len({ids[0] for ids in encoded}) != len(support):
            raise ValueError("Bare-label support is not uniquely tokenized")
        tokens = [ids[0] for ids in encoded]
        roster, _ = joined(source_cfg)
        args.output.mkdir(parents=True, exist_ok=False)
        results = {}
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, source_cfg, np, check)
            for backend in source_cfg["backends"]:
                results[backend] = run_backend(backend, cfg, source_cfg, roster, support, tokens,
                                               head, args.output, check, np)
                print(f"{backend}: verified {len(roster)} prefix records", flush=True)
        if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
            raise RuntimeError("Archive audit imported Torch")
        check()
    write_new(args.output / "aggregate.json", {
        "experiment": cfg["experiment"], "status": "complete", "source_manifest_sha256": digest(MANIFEST),
        "roots": source_cfg["roots"], "world_arm_rows_per_backend": len(roster),
        "physical_sites_per_row": len(cfg["layers_zero_based"]) * len(cfg["passes"]),
        "new_pretrained_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False,
        "per_backend": results,
        "resource": {"max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
                     "max_sampled_working_set_bytes": max(v["working_set_bytes"] for v in samples),
                     "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples),
                     "sample_count": len(samples)},
    })


if __name__ == "__main__":
    main()
