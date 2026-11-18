"""Tokenization-only NAN-COMP-04 preflight before any Nanbeige 3B weight load.

Loads only the pinned tokenizer and source-only cohort. Requires exact prompt
length pairing, N/R current-observation identity, and unique root prompt
fingerprints. Never opens host labels, EvalMath or ARG-03 heldout data.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from . import run
from .cohort import HERE
from .decode import sequence_hash

PINNED_TOKENIZER_CONFIG_SHA256="82316b70213c65e5f820c03d5a11561ccda4055bda984cb770ed0d51fe497c56"


def scan(rows:list[dict],tokenizer,snapshot:Path,*,max_prompt:int)->dict:
    if max_prompt < 1:
        raise ValueError("invalid registered prompt bound")
    names=set()
    cards=[]
    counts=Counter()
    for split in ("calibration","development"):
        for root,items in run.group_rows(rows,split,0):
            tokens=run.tokenize_root(tokenizer,snapshot,items,max_prompt)
            unique={sequence_hash(ids) for ids,kind in tokens.values()}
            if len(tokens)!=11 or len(unique)!=9:
                raise AssertionError("expected eleven rows and nine distinct prompts per root")
            # Even an identical tokenizer-prompt from ANOTHER root would spoil
            # root-held-out evaluation, irrespective of different case IDs.
            if names.intersection(unique):
                raise AssertionError("duplicate rendered prompt across distinct roots")
            names.update(unique)
            lengths={row["case_id"]:len(tokens[row["case_id"]][0]) for row in items}
            for row in items:
                counts[split+":"+row["kind"]+":"+row["tier"]]+=1
            cards.append({
                "root_group":root,"split":split,
                "n_tasks":len(items),"distinct_prompt_hashes":len(unique),
                "min_tokens":min(lengths.values()),
                "max_tokens":max(lengths.values()),
                "token_identity_digest":sha256(json.dumps(sorted(unique)).encode()).hexdigest(),
            })
    if len(cards)!=20 or len(names)!=180:
        raise AssertionError("incomplete tokenized CAL/DEV root cohort")
    return {
        "schema":"nan_comp04.tokenizer_only_preflight.v1",
        "status":"PASS","pretrained_model_loaded":False,
        "heldout_168_accessed":False,"reserved_confirmation_used":False,
        "n_roots":20,"n_model_visible_tasks":220,
        "unique_rendered_prompts":len(names),
        "max_prompt_tokens":max(x["max_tokens"] for x in cards),
        "registered_prompt_cap":max_prompt,
        "per_kind_tier":dict(sorted(counts.items())),
        "cards":cards,
    }


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir",required=True,type=Path)
    p.add_argument("--compactification-root",required=True,type=Path)
    p.add_argument("--snapshot",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args(argv)
    run.forbid_test_paths(a.source_dir,a.out)
    if a.out.exists():
        raise FileExistsError("refuse to overwrite tokenizer preflight receipt")
    rows,src_receipt=run.read_public(a.source_dir)
    lock=run.lock_source()
    cfg=json.loads((HERE/"config.json").read_text(encoding="utf-8"))
    entry_lock=cfg["checkpoint_revision"]
    if lock["checkpoint_revision"]!=entry_lock:
        raise ValueError("registered snapshot revision mismatch")
    config=a.snapshot/"tokenizer_config.json"
    if not config.is_file() or run.sha(config)!=PINNED_TOKENIZER_CONFIG_SHA256:
        raise RuntimeError("missing or changed pinned Nanbeige tokenizer configuration")
    # Only after all source/lock checks: import native tokenizer source.
    import sys
    sys.path.insert(0,str(a.compactification_root.resolve()))
    from research.algernon.algebraic_reasoning_gym.arg02.nanbeige import load_tokenizer
    from research.algernon.algebraic_reasoning_gym.arg02.settings import load_lock
    model_entry=load_lock()["nanbeige"]
    if model_entry["revision"]!=entry_lock:
        raise RuntimeError("ARG-02 checkpoint lock points at a different model revision")
    tok=load_tokenizer(a.snapshot,model_entry["bos_token_id"])
    report=scan(rows,tok,a.snapshot,
                max_prompt=cfg["generation"]["model_prompt_cap_tokens"])
    report["pinned_checkpoint_revision"]=entry_lock
    report["tokenizer_config_sha256"]=run.sha(config)
    report["public_source_sha256"]=src_receipt["files_sha256"]["public.jsonl"]
    report["local_tokenizer_source_sha256"]=run.sha(Path(__file__))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in
                      ("status","n_roots","n_model_visible_tasks",
                       "unique_rendered_prompts","max_prompt_tokens")},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
