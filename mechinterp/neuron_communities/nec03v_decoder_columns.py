"""NEC-03V: frozen-source SAE DECODER COLUMN extraction, no neural forward.

Reads only immutable, independently audited source-only .pt weight archives.
Reconstructs (128 features x 32 native MLP output dimensions) decoder columns
by explicit linear algebra at the original registered recurrent visits.

This CANNOT recover the 1280 source-fit code activations or the 256 per-case
positive TopK coefficients. It does not load any task data, access a world
generator, score logits or train models. Never pass protected-world archives.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .nec03v_source_attest import ARMS, attest, canonical, sha256_file, _existing_name


def _load_cpu(path: Path):
    import torch
    if torch.version.cuda is not None or getattr(torch.version, "hip", None) is not None:
        raise RuntimeError("NEC03V geometry extraction requires CPU-only PyTorch")
    return torch.load(path, map_location="cpu", weights_only=True)


def original_visits(roster: dict, candidate):
    if candidate is None:
        return {}
    data = roster.get("roster", roster.get("source_roster"))
    if not isinstance(data, dict):
        raise ValueError("historical source roster is missing")
    if data.get("source_candidate") != candidate:
        raise ValueError("source-site roster differs from original registered candidate")
    hubs = data["source_graph"]["hubs"]
    if not isinstance(hubs, list):
        raise ValueError("historical source hubs must be a list")
    result, seen = {}, set()
    for node in candidate["nodes"]:
        if not isinstance(node, str) or node.count(":") != 1:
            raise ValueError("invalid original visit:neuron source site")
        v, n = node.split(":")
        if not (v.isdecimal() and n.isdecimal() and 0 <= int(v) < 12 and 0 <= int(n) < 64):
            raise ValueError("original source-site index outside recurrence")
        if node in seen:
            raise ValueError("duplicate frozen original source-site node")
        seen.add(node)
        result.setdefault(int(v), set()).add(int(n))
    for node in hubs:
        if not isinstance(node, str) or node.count(":") != 1:
            raise ValueError("malformed source hub")
        v, n = node.split(":")
        if not (v.isdecimal() and n.isdecimal() and 0 <= int(v) < 12 and 0 <= int(n) < 64):
            raise ValueError("source hub outside original 12 visits")
        if int(v) in result:
            result[int(v)].add(-int(n)-1)
    if not 1 <= len(result) <= 12:
        raise ValueError("original candidate must select original recurrent visits")
    # Nonnegative entries are native source nodes. Signed negative entries are
    # the local hub indices encoded as -neuron-1, distinguished from sources.
    return result


def extract(root: int, source_dir: Path, t_dir: Path,
            reset_dir: Path | None, output: Path):
    if output.exists():
        raise ValueError("decoder column export destination must be fresh")
    source_attestation = attest(root, source_dir, t_dir, reset_dir)
    source_file = ("nec03o_source_refit_receipt.json" if root < 4
                   else "nec03ob_source_receipt.json")
    meta = json.loads((source_dir / source_file).read_text(encoding="utf-8"))
    report = {
        "schema": "NEC-03V/source-frozen-projected-decoder-columns/v1",
        "world_index": root,
        "world_id": source_attestation["world_id"],
        "original_source_attestation": source_attestation,
        "original_nec03t_lock_sha256":
            source_attestation["original_nec03t_source_lock_sha256"],
        "arms": {},
        "native_forward_calls": 0, "SAE_forward_calls": 0,
        "model_or_SAE_optimizer_steps": 0,
        "labels_accessed": 0, "logits_accessed": 0,
        "new_recruited_root_reads": 0,
        "embargoed_confirmation_reads": 0,
        "source_fit_activations_recovered": False,
        "per_case_active_code_coefficients_recovered": False,
        "decoder_columns_computed_from_authenticated_weights": True,
        "scientific_gate": "REDESIGN_OR_PILOT_ONLY",
    }
    for arm in ARMS:
        a = meta["arms"][arm]
        receipt = source_attestation["source_checkpoint_detail"][arm]
        roster = json.loads((source_dir / f"{arm}.source_roster.json").read_text(encoding="utf-8"))
        sites = original_visits(roster, a["source_candidate"])
        if not sites:
            if a["SAE_source"] is not None or root not in (3, 5):
                raise ValueError("nonexistent registered source site has an SAE or invalid root")
            report["arms"][arm] = {"selected_visits": [], "decoder32": {},
                                    "source_sites_sha256":receipt["source_roster_sha256"],
                                    "original_SAE_absent": True}
            continue

        native_file = (reset_dir if root < 4 else source_dir) / _existing_name(
            receipt["native_checkpoint_file"])
        sae_file = source_dir / _existing_name(a["SAE_source"]["source_dictionary_file"])
        native = _load_cpu(native_file)
        sae = _load_cpu(sae_file)
        state = native["model"]
        sae_state = sae["state"]
        down = state["block.mlp_down.weight"].detach().float().cpu()
        decoder = sae_state["decoder.weight"].detach().float().cpu()
        scale = sae["scale"].detach().float().cpu()
        if (tuple(down.shape) != (32,64) or tuple(decoder.shape) != (64,128)
                or tuple(scale.shape) != (64,) or
                not bool(down.isfinite().all()) or
                not bool(decoder.isfinite().all()) or
                not bool(scale.isfinite().all()) or
                bool((scale <= 0).any())):
            raise ValueError("frozen source native/SAE tensor geometry is invalid")
        if (sae.get("model_sha256") != receipt["source_model_digest"]
                or sae.get("world_sha256") != source_attestation["world_sha256"]
                or sae.get("arm") != arm
                or sae.get("source_fit_context_groups") != 40
                or sae.get("source_rank_context_groups") != 40):
            raise ValueError("SAE payload belongs to a different original source checkpoint")
        selected = {}
        for visit, encoded in sorted(sites.items()):
            hubs = {-x - 1 for x in encoded if x < 0}
            site_cols = decoder * scale[:, None]
            mask = down.new_ones((64,))
            for hub in hubs:
                mask[hub] = 0.0
            projected = (down @ (site_cols * mask[:,None])).T.contiguous()
            if projected.shape != (128, 32) or not bool(projected.isfinite().all()):
                raise ValueError("invalid original projected 32D decoder columns")
            selected[str(visit)] = projected.tolist()
        report["arms"][arm] = {
            "selected_visits": sorted(sites),
            "decoder32": selected,
            "source_sites_sha256": receipt["source_roster_sha256"],
            "native_checkpoint_sha256": receipt["native_checkpoint_sha256"],
            "SAE_checkpoint_sha256": receipt["sae_checkpoint_sha256"],
            "original_SAE_absent": False,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical(report) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-index", type=int, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--nec03t-root", type=Path, required=True)
    p.add_argument("--reset-root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    result = extract(args.world_index,args.source_root,args.nec03t_root,
                     args.reset_root,args.out)
    print(canonical({"world_id":result["world_id"],
                     "projected_visits":{arm:len(data["selected_visits"])
                                          for arm,data in result["arms"].items()},
                     "native_forward_calls":0,
                     "per_case_active_code_coefficients_recovered":False}))
    return result


if __name__ == "__main__":
    main()
