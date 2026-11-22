"""Commission the full immutable WINDOW66 journal and both-world reducer."""
import argparse
import json
from pathlib import Path

from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime import TARGETS
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_route58.custody import SOURCE_CHECK, bound_source, settings
from experiments.nan_route58.host import native_gate
from experiments.nan_route58.verify import record

from .capture import FOCUS
from .host import audit, audit_route
from .lane import EXPERIMENT, file_hashes, write_worlds
from .telemetry import KEYS, LINEAR, TERMS

BUDGET = {"minimum_start_available_RAM_bytes": 5 * 2**30, "minimum_available_RAM_bytes": 4 * 2**30,
          "maximum_private_bytes": 128 * 2**20, "maximum_seconds": 60}
TAG = "0" * 64


def fixture_record(own, donor, multiplier):
    result = record(own, donor)
    base = result["baseline"]["bare_margin"]
    for item in result["outcomes"].values():
        item["bare_margin"] = base + (item["bare_margin"] - base) * multiplier
    profile = {}
    for key in KEYS:
        native = result["outcomes"][key]["bare_margin"] - base
        layers = {}
        for layer in FOCUS:
            final = layer == FOCUS[-1]
            swing = native if final else native * (layer + 1) / 32
            terms = dict.fromkeys(TERMS, 0.) | {"attention_input": swing * .5, "attention_write": swing * .5}
            cell = {"native_swing": swing, "finite": terms,
                    "continuous": dict.fromkeys(LINEAR, 0.) | terms,
                    "attention_rounding_delta_l2": 0., "mlp_rounding_delta_l2": 0.}
            if final:
                cell |= {"actual_selected_head_swing": native, "max_selected_logit_error": 0.}
            layers[str(layer)] = cell
        profile[key] = layers
    return result | {"delayed_window": profile}


def fail(call, message):
    try:
        call()
    except ValueError as error:
        assert message in str(error), error
    else:
        raise AssertionError(f"Expected refusal: {message}")


def fixture(source, work, guard):
    cfg, _, round_cfg, dose = settings()
    gold, tokens = (bound_source(source, key) for key in ("gold", "tokens"))
    labels = tokens["bare_label_token_ids"]
    native = [
        {"id": r["id"], "text": r["answer"] if r["arm"] != "focused_erased" else "A",
         "token_id": labels[r["answer"] if r["arm"] != "focused_erased" else "A"]} for r in gold
    ]
    gate = native_gate(source, native, cfg)
    assert gate["passed"] and len(gate["qualified_focused_roots"]) == cfg["source_roots"] == 12
    focused = {(r["root"], r["world"]): r for r in gold if r["arm"] == "focused"}
    ids = {r["id"]: key for key, r in focused.items()}
    data = {(r["root"], r["arm"], r["world"]): (r["id"], tokens["row_bindings"][r["id"]]["ids"]) for r in gold}
    custody = {key: (item["answer"], f"fixture-{i}") for i, (key, item) in enumerate(sorted(focused.items()))}
    first_four = list(sorted(focused))[:4]
    scale = dict(zip(first_four, (1, 2, 9, None), strict=True))
    called = set()

    def trial(tag, prompt, own, other, fingerprint):
        guard()
        root, world = ids[tag]
        assert (tag, prompt) == data[root, "focused", world] and custody[root, world] == (own, fingerprint)
        assert tag not in called
        called.add(tag)
        factor = scale.get((root, world))
        return fixture_record(labels[own], labels[other], factor) if factor else {
            "status": "head_unqualified", "edited_pass2_runs": 0}

    compared = {"greedy_tokens_equal": True, "vocabulary_rows_compared": 166144}
    parity = {"prefill": [compared, compared], "increment": [compared, compared], "pruned_increment": compared}
    depth = {"tested_max_position": json.loads(SOURCE_CHECK.read_bytes())["max_prompt_length"] + 2,
             "KV_bytes_at_max_position": 1, "checks": {"actual_CAL": parity, "padded_CAL": parity}}
    preflight = {"synthetic": True, "depths": {"1": depth, "2": depth},
        "window": {"layers": list(FOCUS), "same_shape_native_bytes_equal": True,
                   "checked_residual_additions": len(FOCUS)*2*2*2}}
    for name, value in (("preflight.json", preflight), ("native.json", native), ("qualification.json", gate)):
        write_new(work / name, value)
    written = write_worlds(work, data, custody, gate, trial)
    assert len(called) == 24
    inactive = work / "refused_gate"
    inactive.mkdir()
    stopped = write_worlds(inactive, data, custody, gate | {"passed": False}, None)
    assert stopped["records"] == 24 and all(
        json.loads(file.read_bytes())["status"] == "native_gate_failed" for file in inactive.glob("episode_*.json"))

    target = TARGETS[cfg["backend"]]
    terminal_path = work / "terminal.json"
    terminal = written | {"experiment": EXPERIMENT, "mode": "fixture", "window_layers": list(FOCUS),
        "runtime_sha256": TAG, "backend": cfg["backend"], "target_before": target, "target_after": target,
        "quant_unchanged": True, "checkpoint_verified_after": True, "status": "completed", "vocabulary_size": 166144,
        "preflight_sha256": sha(work / "preflight.json"), "native_sha256": sha(work / "native.json"),
        "qualification_sha256": sha(work / "qualification.json")}
    write_new(terminal_path, terminal)

    def host_call():
        return audit(source, work, TAG, cfg, round_cfg, dose, fixture=True)

    result = host_call()
    assert result["roots_with_both_worlds_scored"] == 1 and result["worlds_scored"] == 3
    assert result["refusals"] == {"head_unqualified": 21}
    assert result["root_mean"]["signed_pass2_margin_contrast_minus_sham"] == .28125
    assert result["window_root_macro_mean"][str(FOCUS[-1])]["restricted_head_effect"] == .28125
    assert result["window_root_macro_mean"][str(FOCUS[0])]["restricted_head_effect"] == .10546875
    fail(lambda: audit(source, work, TAG, cfg, round_cfg, dose), "registered WINDOW66 terminal")

    ep = work / "episode_000.json"
    journal = work / "closed.jsonl"
    original = {path: path.read_bytes() for path in (ep, journal, terminal_path)}
    edited = json.loads(original[ep])
    edited["delayed_window"][KEYS[0]][str(FOCUS[-1])]["actual_selected_head_swing"] += 1.
    try:
        ep.write_text(json.dumps(edited) + "\n", encoding="utf-8", newline="\n")
        fail(host_call, "differs from its journal")
        lines = original[journal].decode("utf-8").splitlines()
        lines[0] = json.dumps({"file": ep.name, "sha256": sha(ep)})
        journal.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        terminal_path.write_text(json.dumps(terminal | {"closed_journal_sha256": sha(journal)}) + "\n",
                                 encoding="utf-8", newline="\n")
        baseline = audit_route(source, work, TAG, cfg, round_cfg, dose)
        assert baseline["root_mean"] == result["root_mean"]
        fail(host_call, "does not bind the native route score")
    finally:
        for path, data in original.items():
            path.write_bytes(data)
    assert host_call() == result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        parser.error("Enable Python assertions")
    initial = file_hashes()
    with lease(BUDGET) as (guard, samples):
        args.work.mkdir(parents=True, exist_ok=False)
        report = fixture(args.source_dir, args.work, guard)
    if initial != file_hashes():
        raise ValueError("A runtime dependency changed during verification")
    result = {"status": "passed", "experiment": EXPERIMENT, "checked_world_slots": 24,
              "blocked_gate_world_slots": 24, "checked_complete_roots": report["roots_with_both_worlds_scored"],
              "checked_closed_worlds": report["worlds_scored"], "incomplete_closed_worlds_excluded": 1,
              "malformed_receipts_refused": 3, "fixture_final_effect": report["root_mean"]["signed_pass2_margin_contrast_minus_sham"],
              "fixture_first_window_effect": report["window_root_macro_mean"][str(FOCUS[0])]["restricted_head_effect"],
              "fixture_refusals": report["refusals"], "files_sha256": initial,
              "max_sampled_private_bytes": max(x["private_bytes"] for x in samples),
              "min_sampled_available_RAM_bytes": min(x["available_physical_bytes"] for x in samples)}
    write_new(args.work / "verification.json", result)
    print({k: v for k, v in result.items() if k != "files_sha256"}, flush=True)


if __name__ == "__main__":
    main()
