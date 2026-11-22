"""Exercise matched budget replay and missing outcomes on the complete source roster."""
from copy import deepcopy

from experiments.nan_readout34 import oracle, source, worlds
from experiments.nan_readout34.runtime_audit import audit_episode
from experiments.nan_readout34.runtime_plan import schedule

from .run import prefixes, summarize


def close(slot, text, length):
    def decode(ids):
        return text if ids and ids[-1] == 1 else " " * len(ids)

    row = slot.source
    binding = {"tokens": 100, "ids_sha256": "synthetic"}
    record = {
        "ordinal": slot.ordinal, "source_id": row["id"], "passes": slot.passes,
        "prompt_tokens": binding["tokens"], "prompt_ids_sha256": binding["ids_sha256"],
        "generated_ids": [0] * (length - 1) + [1], "stop_reason": "complete_packet",
        "text": text, "wall_seconds": 1.0, "phase_token_visits": slot.passes * (100 + length - 1),
    }
    score = audit_episode(record, slot, binding, decode, eos_token=2, vocab=3)
    receipt = (score, record)
    for cap in range(1, length + 2):
        direct = oracle.grade(decode(record["generated_ids"][:cap]), row["schema"], row["label"])
        replayed = prefixes(receipt, cap, slot.passes)
        assert replayed["exact"] == direct["joint_exact"]
        assert replayed["format_valid"] == direct["format_valid"]
        assert replayed["final_room"] == direct["fields_correct"]["final_room"]
        assert replayed["decode_visits"] == slot.passes * (min(cap, length) - 1)
    return receipt


def main():
    rows = source.build()
    slots = schedule(rows)
    example_root = rows[0]["root"]
    examples = {
        (row["world"], row["schema"]["readout"], slot.passes): slot for slot in slots
        if (row := slot.source)["root"] == example_root and row["event_count"] == 8
        and row["arm"] == "compute" and row["schema"]["readout"] in ("final", "packet")
    }
    verified = {}
    for world, readout, passes, length in [
        ("A", "final", 1, 9), ("A", "final", 2, 7),
        ("A", "packet", 1, 9), ("A", "packet", 2, 13), ("B", "final", 1, 6),
    ]:
        slot = examples[world, readout, passes]
        label = deepcopy(slot.source["label"])
        if world == "B":
            label["final_room"] = next(room for room in ("R0", "R1") if room != label["final_room"])
        if readout == "packet" and passes == 1:
            label["path"][-1] = next(room for room in ("R0", "R1") if room != label["path"][-1])
        verified[slot.ordinal] = close(slot, worlds.canonical(label, readout), length)

    lengths = {row["id"]: 7 if row["schema"]["readout"] == "final" else 19 for row in rows}
    result = summarize(slots, {"nf4": verified, "bf16": {}}, lengths)
    assert all(len(item["depth"]) == 600 and len(item["answer_and_packet"]) == 288
               for item in result.values())
    nf4, bf16 = result["nf4"], result["bf16"]
    stratum = ["CAL", "actual", 8, "compute"]
    specific = [row for row in nf4["depth"] if row["stratum"] == [*stratum, "final"]]
    short = next(row for row in specific if row["budget_kind"] == "output_tokens" and row["budget"] == 8)
    assert short["exact_one_to_two"] == {
        "registered_pairs": 2, "repair": 1, "regression": 0, "both_correct": 0,
        "both_incorrect": 0, "unknown": 1,
    }
    assert short["both_worlds_by_depth"][0]["incorrect"] == 1
    assert short["both_worlds_by_depth"][1]["unknown"] == 1
    assert short["complete_pair_cost"] == {
        "pairs": 1, "tokens": [8, 7], "decode_visits": [7, 12], "unpadded_visits": [107, 212],
    }
    work = next(row for row in specific if row["budget_kind"] == "decode_phase_visits" and row["budget"] == 16)
    assert work["token_caps_by_depth"] == [17, 9]
    assert work["exact_one_to_two"]["both_correct"] == 1
    wrong_path = next(row for row in nf4["answer_and_packet"]
                      if row["stratum"] == [*stratum, 1] and row["output_token_cap"] == 96)
    assert wrong_path["final_only_to_packet_final"]["both_correct"] == 1
    assert wrong_path["packet_final_to_verified_packet"]["regression"] == 1
    assert wrong_path["path_and_checkpoints_verifiable"]["unknown"] == 1
    assert all(row["exact_one_to_two"]["unknown"] == row["exact_one_to_two"]["registered_pairs"]
               and row["complete_pair_cost"]["tokens"] is None for row in bf16["depth"])
    assert sum(row["exact_one_to_two"]["registered_pairs"] for row in nf4["depth"]) == 10 * 240
    print("Passed first-stop parity, depth and cost accounting, packet-path failure, and missing-row checks.")


if __name__ == "__main__":
    main()
