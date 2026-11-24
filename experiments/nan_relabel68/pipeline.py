"""Read both physical heads from one native paired prefill and bind batch custody."""
import json
from math import exp, isfinite, log, log1p
from time import monotonic

from experiments.nan_gated46.source import inputs
from experiments.nan_readout34.tokenizer_check import sha, tokenizer
from experiments.nan_stream41.pipeline import commit, filename

from .plan import DEPTHS, pair_roster
from .source import NODES, digest
from .verify import HERE, MANIFEST as SOURCE_MANIFEST, REVISION, seal


VERIFY = HERE / "VERIFICATION.json"
MAX_POSITION = 307


def source_bindings(folder, fields):
    lock = json.loads(VERIFY.read_bytes())
    if (lock["manifest_sha256"] != sha(SOURCE_MANIFEST) or json.loads(SOURCE_MANIFEST.read_bytes()) != seal()
            or lock["status"] != "source_tokenizer_and_host_verified"):
        raise ValueError("RELABEL68 frozen source or tokenizer receipt changed")
    prepared = {}
    for name in fields:
        value = json.loads((folder / f"{name}.json").read_bytes())
        if digest(value) != lock["payload_sha256"][name]:
            raise ValueError(f"RELABEL68 {name} does not bind the registered source")
        prepared[name] = value
    return prepared, lock


def prepare(folder, snapshot, check):
    if snapshot.name != REVISION:
        raise ValueError("Wrong pinned Nanbeige revision")
    sources, lock = source_bindings(folder, ("public", "token_bindings", "b2_roster"))
    tok, pins = tokenizer(snapshot)
    provided = sources["token_bindings"]
    if pins != lock["tokenizer_pins"] or inputs(tok, sources["public"], snapshot, check) != provided:
        raise ValueError("Actual tokenizer does not reproduce the frozen public inputs")
    bindings = provided["row_bindings"]
    jobs = pair_roster(sources["public"], bindings)
    if jobs != sources["b2_roster"] or len(jobs) != 288 or max(j["length"] for j in jobs) + 2 != MAX_POSITION:
        raise ValueError("Packed runtime job roster or preflight capacity changed")
    return tok, jobs, bindings, provided["bare_label_token_ids"]


def compress(logits, labels):
    import torch

    if logits.ndim != 2 or logits.shape[0] != 2 or not logits.isfinite().all():
        raise ValueError("Expected complete, finite two-row vocabulary logits")
    if set(labels) != set(NODES) or len(set(labels.values())) != 16:
        raise ValueError("Expected all 16 distinct bare-label token IDs")
    values = logits.detach().to(device="cpu", dtype=torch.float64)
    winner = values.argmax(dim=-1)
    top = values.topk(2, dim=-1).values
    logz = values.logsumexp(dim=-1)
    return [{"winner": int(winner[i]), "peak": float(top[i, 0]), "second_logit": float(top[i, 1]),
             "log_normalizer": float(logz[i]), "labels": {k: float(values[i, token]) for k, token in labels.items()}}
            for i in range(2)]


def paired_heads(engine, traces, guard):
    """A single B2 trace owns both physical heads. The prompt is never decoded onward."""
    began = monotonic()
    if len(traces) != 2 or len(traces[0]) != len(traces[1]):
        raise ValueError("Each experimental prefill needs both equally shaped A/B rows")
    guard(began, 2, len(traces[0]))
    engine.reset(2)
    try:
        packet = engine.cache.prefill(traces, capture=False)
        if (len(packet["boundaries"]) != 2 or packet["states"] or len(engine.cache.caches) != 2
                or engine.cache.semantic_lengths != list(map(len, traces))
                or engine.cache.physical_length != len(traces[0])):
            raise RuntimeError("Paired capture omitted a physical phase or changed the canvas")
        result = []
        for boundary in packet["boundaries"]:
            guard(began, 2, len(traces[0]))
            view = engine.head(boundary)
            if view.shape != (2, engine.vocabulary_size):
                raise ValueError("A physical pass did not read the full vocabulary")
            result.append(view.detach().cpu())
        return result
    finally:
        engine.reset(2)


def observe(engine, traces, labels, guard):
    return [compress(view, labels) for view in paired_heads(engine, traces, guard)]


def capture_preflight(engine, traces, labels, guard):
    import torch

    began = monotonic()
    engine.reset(1)
    try:
        guard(began, 2, len(traces[0]))
        shallow = engine.prefill_logits(traces).detach().cpu()
    finally:
        engine.reset(1)
    paired = paired_heads(engine, traces, guard)
    if not torch.equal(shallow, paired[0]):
        raise RuntimeError("Paired runtime first pass differs from standalone B2 first-pass readout")
    compress(shallow, labels)
    return {"same_shape_shallow_equals_paired": True, "rows": 2, "labels_checked_per_row": 16,
            "full_vocabulary_rows_checked": shallow.shape[1]}


def close_batch(work, job, readings, bindings, tape):
    if len(readings) != 2 or any(len(rows) != 2 for rows in readings):
        raise ValueError("Batch lacks a phase or physical row")
    packets = [{"source_id": key, "tokens": bindings[key]["tokens"], "token_sha256": bindings[key]["sha256"],
                "phases": [reading[row] for reading in readings]}
               for row, key in enumerate(job["source_ids"])]
    commit(work, {"index": job["ordinal"], "length": job["length"], "episodes": packets,
                  "state_tape": tape}, job["ordinal"])


def valid_num(value):
    return type(value) in (float, int) and isfinite(value)


def audited_score(reading, labels, vocab):
    if set(reading) != {"winner", "peak", "second_logit", "log_normalizer", "labels"}:
        raise ValueError("Unexpected full-head confidence record")
    winner = reading["winner"]
    peak, second, logz = (reading[key] for key in ("peak", "second_logit", "log_normalizer"))
    vals = reading["labels"]
    if (type(winner) is not int or not 0 <= winner < vocab or set(vals) != set(labels)
            or not all(valid_num(value) for value in (peak, second, logz, *vals.values()))):
        raise ValueError("Invalid vocabulary or logit values")
    if (second > peak or logz < peak or logz > peak + log(vocab) + 1e-10
            or logz + 1e-10 < peak + log1p(exp(second - peak))):
        raise ValueError("Full-vocabulary peak, runner-up and normalization are inconsistent")
    for key, token in labels.items():
        if (token == winner and vals[key] != peak) or (token != winner and vals[key] > second):
            raise ValueError("Winner or runner-up does not bind bare-label logits")
    return winner


def audited_batch(record, job, bindings, labels, vocab):
    if ((record["index"], record["length"]) != (job["ordinal"], job["length"])
            or len(record["episodes"]) != 2):
        raise ValueError("Immutable result has the wrong batch index, shape or occupancy")
    result = {}
    for row, key in zip(record["episodes"], job["source_ids"], strict=True):
        binding = bindings[key]
        if (row["source_id"], row["tokens"], row["token_sha256"]) != (key, binding["tokens"], binding["sha256"]):
            raise ValueError("Immutable result attaches to the wrong prompt or physical row")
        if len(row["phases"]) != 2:
            raise ValueError("One physical phase is missing")
        for phase, reading in zip(DEPTHS, row["phases"], strict=True):
            result[key, phase] = audited_score(reading, labels, vocab)
    return result
