"""Exercise native scoped BF16 capture and independent host scoring."""

import argparse
from copy import deepcopy
from importlib.metadata import version
import json
from math import exp, log
import os
from pathlib import Path
import sys

from experiments.nan_comp14.parser import label
from experiments.nan_phase35.run import lease
from experiments.nan_premlp64.verify import BUDGET, CONFIGURATION, PACKAGES, hook_count
from experiments.nan_radial76.native import WriterReaderCapture, WRITER
from experiments.nan_readpath37.run import bound, write_new
from experiments.nan_readout34.tokenizer_check import sha, tokenizer
from experiments.nan_writer77.pipeline import MODES, batch_baseline, root_trial
from experiments.nan_writer77.source import CONFIG, load
from experiments.nan_writer77.verify import SEED

from .capture import observed_baseline, observed_trial
from .heads import INVALID, LABELS, Reducer, partition, partition_sha
from .host import score


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WRITER_MANIFEST = REPO / "experiments/nan_writer77/RUNTIME_MANIFEST.json"
BASE = REPO / "experiments/nan_readpath37/registration.json"
MANIFEST = HERE / "SOURCE_MANIFEST.json"


def seal():
    local = [HERE / n for n in (".gitattributes", "__init__.py", "PROTOCOL.md",
                                "capture.py", "heads.py", "host.py", "verify.py")]
    dependency = [BASE, WRITER_MANIFEST, REPO / "experiments/nan_writer77/verify.py",
                  REPO / "experiments/nan_vocab81/SOURCE_MANIFEST.json",
                  REPO / "experiments/nan_premlp64/verify.py",
                  REPO / "experiments/nan_phase35/run.py"]
    return {"experiment": "NAN-VOCAB-82", "stage": "observer_and_host_cpu_commission",
            "files_sha256": {p.relative_to(REPO).as_posix(): sha(p) for p in local + dependency},
            "resource_lease": BUDGET}


def replay_fixture(snapshot, work, check, np, torch):
    from experiments.nan_fit07.pointwise import masks
    from experiments.nan_recomp09.common import target_digest
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import build_nanbeige_model_and_loop
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    torch.manual_seed(SEED)
    model, loop = build_nanbeige_model_and_loop(
        snapshot, exec_device=torch.device("cpu"), dtype=torch.bfloat16, n_gpu_layers=2,
        random_init=True, config_overrides={"vocab_size": 97, "num_hidden_layers": 3,
        "hidden_size": 768, "intermediate_size": 320, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "kv_channels": 16})
    model.requires_grad_(False)
    loop.stage_layer.requires_grad_(False)
    saved, hooks = target_digest(loop), hook_count(loop)
    rng = np.random.default_rng(SEED)
    classes = bytes(i % (INVALID + 1) for i in range(97))
    reduce = Reducer(classes)
    # A complete tied vocabulary proves the class-level tie rule independently.
    flat = reduce(torch.ones(2, 2, len(classes)))
    assert all(list(map(tuple, phase)) == [(j, 1.) for j in range(INVALID + 1)]
               for row in flat for phase in row)
    head = StreamedVocabularyHead(model.lm_head.weight, loop.device, 13)
    capture = WriterReaderCapture(loop, head, chunk=3)
    try:
        bare = dict(zip(LABELS, range(INVALID), strict=True))
        prompts = [[1, 4, 32, 5, 29, 6, 42], [8, 39, 3, 4, 65, 77, 91]]
        payload = {"public": [{"root": "fixture", "arm": "focused", "world": world, "id": world}
                              for world in ("A", "B")],
                   "tokens": {"row_bindings": {w: {"ids": ids, "sha256": w} for w, ids in
                                               zip(("A", "B"), prompts, strict=True)},
                              "bare_label_token_ids": bare}}
        prior = head.calls
        baseline, focus = observed_baseline(capture, payload, ["focused"], check, reduce, classes)
        observed_calls = head.calls - prior
        prior = head.calls
        ordinary, _ = batch_baseline(capture, payload, ["focused"], check)
        assert observed_calls == head.calls - prior == 2
        assert [{k: v for k,v in row.items() if k != "class_maxima"} for row in baseline] == ordinary
        clean, tokens = focus["fixture"]
        dense = clean["logits"].numpy()
        for i, row in enumerate(baseline):
            for j, maxima in enumerate(row["class_maxima"]):
                for k, (token, value) in enumerate(maxima):
                    ids = [t for t, class_id in enumerate(classes) if class_id == k]
                    expected = max(ids, key=lambda t: (dense[j, i, t], -t))
                    assert (token, value) == (expected, dense[j, i, expected])
        dim_in, dim_out = (clean[key][WRITER].numel() for key in ("writer_input", "writer_raw"))
        native = np.stack([clean["writer_raw"][1, row].float().numpy() for row in (0, 1)])
        bank = {"V": rng.normal(size=(dim_in,8)) *.05, "U": rng.normal(size=(8,dim_out)) *.05,
                "mean_comp_acts": np.zeros((2,8)),
                "mean_write_rms": np.full(2, float(np.sqrt(np.mean(native*native))) *1.05),
                "ci__proj_in.weight": rng.normal(size=(32,dim_in)) *.02,
                "ci__proj_in.bias": np.zeros(32),
                "ci__out_proj.weight": rng.normal(size=(8,32)) *.015,
                "ci__out_proj.bias": np.full(8,.5)}
        fit = bank, {"variant": "state_only", "n_blocks": 0}, np.full((2,8), .5), None
        previous = head.calls
        augmented = observed_trial(capture, "fixture", clean, tokens, bare, fit, masks,
                                   np, check, reduce, classes)
        observed_calls = head.calls - previous
        previous = head.calls
        reference = root_trial(capture, "fixture", clean, tokens, bare, fit, masks, np, check)
        assert augmented["status"] == reference["status"] == "closed"
        assert observed_calls == head.calls - previous
        old = deepcopy(augmented)
        for world in old["worlds"]:
            del world["class_maxima"]
        assert old == reference and target_digest(loop) == saved
        assert not torch.cuda.is_initialized()
        write_new(work / "native_private.json", augmented)
        return {"clean_head_calls": 2, "ordinary_trial_head_calls": observed_calls,
                "observed_trial_head_calls": observed_calls, "edited_worlds": len(augmented["worlds"]),
                "edited_observed_heads": len(augmented["worlds"]) * len(MODES) * 2,
                "legacy_outputs_equal": True, "all_vocab_tie_fixture": True,
                "gold_loaded_by_worker": False}
    finally:
        capture.close()
        head.close()
        assert hooks == hook_count(loop)


def host_fixture(snapshot, roster, native, check):
    cfg = json.loads(CONFIG.read_bytes())
    source, proof = load(roster, ("public", "gold", "tokens"))
    tok, pins = tokenizer(snapshot)
    if pins != proof["tokenizer_pins"]:
        raise ValueError("Host tokenizer differs from the audited roster")
    tok = tok.tok
    vocab = json.loads((snapshot / "config.json").read_bytes())["vocab_size"]
    classes = partition(tok.backend_tokenizer, vocab, check)
    counts = {c: classes.count(i) for i, c in enumerate(LABELS + "?")}
    # Verify that the underlying fast tokenizer partition agrees with the exact
    # wrapper call used by the frozen host, including tokens with empty text.
    for start in range(0, vocab, 4096):
        check()
        ids = [[i] for i in range(start, min(start + 4096, vocab))]
        decoded = tok.batch_decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        expected = bytes(INVALID if k is None else LABELS.index(k) for k in map(label, decoded))
        assert classes[start:start+len(ids)] == expected
    class_first = [classes.index(k) for k in range(INVALID + 1)]
    bare = source["tokens"]["bare_label_token_ids"]

    def make(first_label, second_label, other_label, *, edited=False):
        changed = {bare[first_label]: 3., bare[second_label]: .5 if edited else 1.,
                   bare[other_label]: 2.}
        if edited:
            changed[class_first[-1]] = 5.
        raw = [[i, 0.] for i in class_first]
        for key, value in changed.items():
            raw[classes[key]] = [key, value]
        winner, peak = min(changed.items(), key=lambda kv: (-kv[1], kv[0]))
        second = sorted(changed.values(), reverse=True)[1]
        legacy = {"winner": winner, "peak": peak, "second_logit": second,
                  "log_normalizer": log(vocab + sum(exp(v)-1 for v in changed.values())),
                  "labels": {name: changed.get(i, 0.) for name, i in bare.items()}}
        return legacy, raw

    gold = {p["id"]: p["answer"] for p in source["gold"]}
    roots = {}
    for p in source["public"]:
        if p["arm"] == "focused":
            roots.setdefault(p["root"], {})[p["world"]] = p["id"]
    enriched = []
    for p in source["public"]:
        key = p["id"]
        target = gold[key] if p["arm"] != "focused_erased" else "A"
        if p["arm"] == "focused":
            other = gold[roots[p["root"]]["A" if p["world"] == "B" else "B"]]
        else:
            other = next(k for k in LABELS if k != target)
        third = next(k for k in LABELS if k not in (target, other))
        first = make(other, target, third) if p["arm"] == "focused" else make(target, other, third)
        second = make(target, other, third)
        enriched.append({"id": key, "token_sha256": source["tokens"]["row_bindings"][key]["sha256"],
                         "phases": [first[0], second[0]], "class_maxima": [first[1], second[1]]})
    journal = []
    for i, (root, ids) in enumerate(sorted(roots.items())):
        twins = [ids[w] for w in ("A", "B")]
        result = {"root":root, "source_ids":twins, "status":"geometry_unqualified"}
        if i == 0:
            result |= {"status":"closed", "plans":native["plans"], "worlds":[]}
            for world, key in enumerate(twins):
                target, other = gold[key], gold[twins[1-world]]
                third = next(k for k in LABELS if k not in (target, other))
                first = make(other, target, third)
                unedited = make(target, other, third)
                active = make(target, other, third, edited=True)
                result["worlds"].append(native["worlds"][world] | {
                    "arms": {m:[first[0], (active if m=="active" else unedited)[0]] for m in MODES},
                    "class_maxima": {m:[first[1], (active if m=="active" else unedited)[1]] for m in MODES}})
        journal.append(result)
    outcomes, private = score(source, enriched, journal, classes, tok, cfg)
    assert len(private) == 1 and outcomes["legacy"]["unresolved_roots"] == 11
    primary = outcomes["contrasts"]["active_minus_matched_seeded"]["pass2"]
    assert primary["root_macro_delta"]["full_margin"] == -3.
    assert primary["root_macro_delta"]["paired_margin"] == .5
    assert primary["root_macro_delta"]["gold_logit"] == 0. and primary["root_macro_delta"]["foil_logit"] == 3.
    assert primary["regressions"] == primary["paired_and_full_opposite_sign_worlds"] == 2
    assert primary["competitor_class_switches"] == 2 and primary["unknown_worlds"] == 22
    corrupted = deepcopy(journal)
    corrupted[0]["worlds"][0]["class_maxima"]["active"][1][INVALID] = [bare["A"], 5.]
    try:
        score(source, enriched, corrupted, classes, tok, cfg)
    except ValueError as exc:
        assert "invalid token" in str(exc)
    else:
        raise AssertionError("An incorrect grammar class leader was accepted")
    return {"registered_worlds": len(enriched), "partition_sha256": partition_sha(classes),
            "class_counts": counts, "primary_fixture": primary, "tamper_refused": True,
            "wrapper_partition_match": True, "full_legacy_agreement": True}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--roster", type=Path)
    cli.add_argument("--work", type=Path)
    cli.add_argument("--native-result", type=Path)
    args = cli.parse_args()
    now = seal()
    if args.freeze:
        write_new(MANIFEST, now)
        return
    if not args.snapshot or not args.work or "torch" in sys.modules or not __debug__:
        cli.error("Use assertions, the admitted snapshot and a new work directory")
    if json.loads(MANIFEST.read_bytes()) != now:
        raise ValueError("Source manifest changed")
    writer = json.loads(WRITER_MANIFEST.read_bytes())
    for path, expected in writer["files_sha256"].items():
        if sha(REPO / path) != expected:
            raise ValueError(f"Frozen WRITER77 file differs: {path}")
    base = json.loads(BASE.read_bytes())
    if args.snapshot.name != base["revision"]:
        raise ValueError("Snapshot revision differs")
    with lease(BUDGET) as (check, samples):
        args.work.mkdir(parents=True, exist_ok=False)
        if args.native_result:
            if not args.roster:
                cli.error("Host verification requires the frozen WRITER77 source")
            old = json.loads((args.native_result / "verification.json").read_bytes())
            source_path = args.native_result / "native_private.json"
            assert old["native_private_sha256"] == sha(source_path) and old["source_manifest_sha256"] == sha(MANIFEST)
            result = host_fixture(args.snapshot, args.roster, json.loads(source_path.read_bytes()), check)
            result |= {"status":"host_fixture_passed", "native_fixture": old,
                       "native_receipt_sha256":sha(args.native_result / "verification.json")}
            assert "torch" not in sys.modules
        else:
            os.environ.update(USE_TORCH="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                              HF_HOME=str((args.work/"hf_home").resolve()),
                              HF_MODULES_CACHE=str((args.work/"hf_modules").resolve()))
            os.environ.pop("TRANSFORMERS_CACHE", None)
            for name in ("config.json", "modeling_nanbeige.py"):
                bound(args.snapshot/name, base["snapshot_pins"][name])
            bound(args.snapshot/"configuration_nanbeige.py", CONFIGURATION)
            assert {k: version(k) for k in PACKAGES} == PACKAGES
            import numpy as np
            import torch

            torch.set_num_threads(1)
            result = replay_fixture(args.snapshot, args.work, check, np, torch)
            result |= {"status":"native_cpu_fixture_passed",
                       "native_private_sha256":sha(args.work/"native_private.json")}
    assert now == seal()
    result |= {"source_manifest_sha256": sha(MANIFEST), "pretrained_forwards": 0,
               "sealed_test_accessed": False, "formal_comparisons": 0,
               "max_sampled_private_bytes": max(x["private_bytes"] for x in samples),
               "max_sampled_working_set_bytes": max(x["working_set_bytes"] for x in samples),
               "min_sampled_available_RAM_bytes": min(x["available_physical_bytes"] for x in samples)}
    write_new(args.work/"verification.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
