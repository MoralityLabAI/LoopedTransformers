"""Prospective matched schedule and gold-independent greedy completion."""
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import time

from . import parser, source
from .tokenizer_check import id_hash, sha, verify_source


HERE = Path(__file__).resolve().parent
CAP = 96
CHAR_CAP = 512
DEPTH_PROBE = ("final", "packet")


@dataclass(frozen=True)
class Slot:
    ordinal: int
    passes: int
    source: dict

    @property
    def file(self):
        return f"episode_{self.ordinal:04d}.json"


def schedule(rows):
    slots = []
    for row in rows:
        if row["schema"]["readout"] in DEPTH_PROBE:
            slots.append(Slot(len(slots), 1, row))
        slots.append(Slot(len(slots), 2, row))
    if len(rows) != 432 or len({row["id"] for row in rows}) != 432:
        raise ValueError("The entire 432-row source roster is required")
    if Counter(slot.passes for slot in slots) != {1: 240, 2: 432}:
        raise ValueError("The predeclared depth schedule changed")
    return slots


def context_bindings(receipt_path):
    seal = verify_source()
    receipt = json.loads(receipt_path.read_bytes())
    public = json.loads((HERE / "TOKENIZER_VERIFICATION.json").read_bytes())
    if sha(receipt_path) != public["private_receipt_sha256"]:
        raise ValueError("Private tokenizer receipt differs from the verified bytes")
    for name in ("tokenizer_check_sha256", "prompt_ids_source_sha256", "source_manifest_sha256"):
        if receipt[name] != public[name]:
            raise ValueError(f"Tokenizer binding changed: {name}")
    contexts = receipt["contexts"]
    if (source.digest(contexts) != public["context_bindings_sha256"]
            or set(contexts) != {row["id"] for row in source.build()}):
        raise ValueError("Fixed prompt token bindings changed")
    if (receipt["generated_token_cap"] != CAP or source.CONFIG["raw_reply_character_cap"] != CHAR_CAP
            or seal["audit"]["rows_per_backend"] != dict(nf4=432, bf16=432)):
        raise ValueError("Frozen source budgets changed")
    return receipt


def stop_reason(text, schema, eos, count):
    if eos:
        return "eos"
    if parser.parse(text, schema) is not None:
        return "complete_packet"
    if len(text) > CHAR_CAP:
        return "character_cap"
    if count == CAP:
        return "token_cap"
    return None


def generate(engine, slot, ids, binding, *, decode, eos_token, budget):
    row = slot.source
    if set(binding) != {"tokens", "ids_sha256"} or (
            len(ids) != binding["tokens"] or id_hash(ids) != binding["ids_sha256"]):
        raise ValueError("Prompt input differs from the pinned tokenizer receipt")
    started = time.monotonic()
    generated = []
    text = ""
    reason = None
    engine.reset(slot.passes)
    try:
        for count in range(1, CAP + 1):
            budget(started)
            token = engine.first(ids) if count == 1 else engine.next(generated[-1])
            if type(token) is not int or not 0 <= token < engine.vocabulary_size:
                raise ValueError("Greedy token is outside the full vocabulary")
            generated.append(token)
            is_eos = token == eos_token
            text = decode(generated[:-1] if is_eos else generated)
            reason = stop_reason(text, row["schema"], is_eos, count)
            if reason:
                break
        budget(started)
        if not reason:
            raise RuntimeError("Finite generation did not record a stop")
        return {
            "ordinal": slot.ordinal,
            "source_id": row["id"],
            "passes": slot.passes,
            "prompt_tokens": len(ids),
            "prompt_ids_sha256": binding["ids_sha256"],
            "generated_ids": generated,
            "text": text,
            "stop_reason": reason,
            "phase_token_visits": slot.passes * (len(ids) + len(generated) - 1),
            "wall_seconds": time.monotonic() - started,
        }
    finally:
        engine.reset(slot.passes)


def write_new(path, value):
    """The writer's exclusive file is closed before it is eligible for the journal."""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=True)
        stream.write("\n")


def execute(engine, slots, bindings, *, tokens, decode, eos_token, budget, record):
    for slot in slots:
        row = slot.source
        result = generate(engine, slot, tokens(row), bindings[row["id"]],
                          decode=decode, eos_token=eos_token, budget=budget)
        record(slot, result)
