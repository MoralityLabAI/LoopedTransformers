"""Phase 2, controls 7 (planted positive) and 8 (absent negative) through the association pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research_gym.interp.association import answer_codes, hash_bit, select_then_confirm, token_parity
from research_gym.interp.dictionaries import load_dictionary
from research_gym.interp.phase2 import MODELS, core_examples, load_model, select_config, selection_half, subset

PLANTED, PARENT = "LOOP-T s1 planted", "LOOP-T s1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["LOOP-T s1", "LOOP-T s2", "LOOP-T s3", "FF-U s3", PLANTED])
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--monitor-bundle", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/dictionaries.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/controls_7_8.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    calibration = core_examples(args.monitor_bundle, "monitor_calibration")
    half = selection_half(calibration["groups"])
    select, report = subset(calibration, half), subset(calibration, ~half)
    labels = {
        "hash_bit": (hash_bit(select["ids"]), hash_bit(report["ids"])),
        "token_parity": (token_parity(select["rows"]), token_parity(report["rows"])),
    }
    direction = None
    planted_result = args.runs / MODELS[PLANTED][0] / "result.json"
    if planted_result.exists():
        direction = torch.tensor(json.loads(planted_result.read_text(encoding="utf-8"))["plant"]["direction"])

    results: dict = {}
    for name in args.models:
        if name not in metrics:
            print("no dictionaries yet, skipped:", name, flush=True)
            continue
        chosen, met = select_config(metrics[name])
        dictionary, _ = load_dictionary(Path(metrics[name][chosen]["path"]), device=device)
        model = load_model(name, args.runs, args.artifact_dir, device)
        codes = (answer_codes(model, dictionary, select["tokens"]), answer_codes(model, dictionary, report["tokens"]))
        entry = {"dictionary": chosen, "selection_bar_met": met}
        entry["control8_hash_bit"] = select_then_confirm(codes[0], labels["hash_bit"][0], codes[1], labels["hash_bit"][1],
                                                         device=device)
        if name in (PLANTED, PARENT) and direction is not None:
            parity = select_then_confirm(codes[0], labels["token_parity"][0], codes[1], labels["token_parity"][1],
                                         device=device)
            cosines = dictionary.decoder.detach().cpu() @ direction
            parity["decoder_cosine_to_direction"] = float(cosines[parity["feature"]])
            parity["max_abs_decoder_cosine_to_direction"] = float(cosines.abs().max())
            entry["control7_token_parity"] = parity
        results[name] = entry
        print(name, json.dumps(entry, sort_keys=True), flush=True)

    summary = {"control8_pass": all(not entry["control8_hash_bit"]["found"] for entry in results.values())}
    if PLANTED in results and PARENT in results and "control7_token_parity" in results[PLANTED]:
        planted, parent = results[PLANTED]["control7_token_parity"], results[PARENT]["control7_token_parity"]
        summary["control7_pass"] = bool(
            planted["found"] and abs(planted["decoder_cosine_to_direction"]) >= 0.5
            and abs(parent["decoder_cosine_to_direction"]) < 0.5
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"models": results, "summary": summary}, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
    print("summary", summary, flush=True)


if __name__ == "__main__":
    main()
