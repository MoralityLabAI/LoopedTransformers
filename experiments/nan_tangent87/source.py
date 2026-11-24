"""Draw and independently verify the fresh NF4 MLP tangent roster."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from experiments.nan_delay39.source import digest, independently_verify, source_cases
from experiments.nan_gated46.source import inputs
from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha, tokenizer, verify_source
from experiments.nan_suffix80.prepare import load as load_suffix, prior_prompts
from experiments.nan_writer77.source import exposed


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
EXPERIMENT="NAN-TANGENT-87"
SOURCE=HERE/"SOURCE_MANIFEST.json"
CHECK=HERE/"SOURCE_CHECK.json"
CONFIG=HERE/"registration.json"
BUDGET={"minimum_start_available_RAM_bytes":5*2**30,"minimum_available_RAM_bytes":4*2**30,
        "maximum_private_bytes":128*2**20,"maximum_seconds":120}
NODES="ABCDEFGHIJKLMNOP"


def config():
    return json.loads(CONFIG.read_bytes())


def seal():
    paths=[HERE/n for n in ("__init__.py",".gitattributes","registration.json","PROTOCOL.md","source.py")]
    paths += [REPO/n for n in ("experiments/nan_comp14/sources.py","experiments/nan_comp14/fixtures/public.jsonl",
        "experiments/nan_gym05/fixtures/public.jsonl","experiments/nan_delay39/source.py",
        "experiments/nan_delay39/SOURCE_CHECK.json","experiments/nan_route58/SOURCE_CHECK.json",
        "experiments/nan_gated46/source.py","experiments/nan_gated46/SOURCE_CHECK.json",
        "experiments/nan_readout34/tokenizer_check.py","experiments/nan_writer77/source.py",
        "experiments/nan_writer77/SOURCE_CHECK.json","experiments/nan_suffix80/prepare.py",
        "experiments/nan_suffix80/SOURCE_CHECK.json","experiments/nan_relabel68/VERIFICATION.json")]
    return {"experiment":EXPERIMENT,"files_sha256":{p.relative_to(REPO).as_posix():sha(p) for p in paths}}


def graph_signature(prompt):
    lines=prompt.splitlines()
    edge_lines=[l for l in reversed(lines) if l.startswith("Edges: ")]
    if not edge_lines:
        return None
    matches=re.findall(r"\b([A-P])->([A-P])\b",edge_lines[0])
    if len(matches)!=16 or set(a for a,_ in matches)!=set(NODES) or set(b for _,b in matches)!=set(NODES):
        return None
    return tuple(b for _,b in sorted(matches))


def prepare(snapshot, out):
    settings=config()
    frozen=seal()
    if json.loads(SOURCE.read_bytes())!=frozen or snapshot.name!=settings["checkpoint_revision"]:
        raise ValueError("Wrong freeze or tokenizer checkpoint")
    with lease(BUDGET) as (check,samples):
        verify_source()
        prior=prior_prompts()|exposed()
        suffix,_=load_suffix(REPO/"runs/nan_suffix80/source01",("public",))
        prior.update(p["prompt"] for p in suffix["public"])
        seen={signature for prompt in prior if (signature:=graph_signature(prompt)) is not None}
        public,gold=source_cases(settings)
        for row in public+gold:
            for k in ("id","root"):
                row[k]=row[k].replace("delay39:","tangent87:",1)
            row["split"]="fresh_development"
        count=independently_verify(public,gold,settings["source_roots"],prior)
        signatures=[graph_signature(row["prompt"]) for row in public if row["arm"]=="focused"]
        if None in signatures or seen.intersection(signatures) or len(set(signatures))!=count:
            raise ValueError("A focused graph is incomplete or repeats a named source graph")
        balance=Counter(r["answer"] for r in gold if r["arm"]=="focused")
        if dict(balance)!=dict.fromkeys(NODES,2):
            raise ValueError("The fresh focused answer roster is not balanced")
        tok,pins=tokenizer(snapshot)
        tokens=inputs(tok,public,snapshot,check)
        pairs={}
        for row in public:
            key=row["root"],row["arm"]
            pairs.setdefault(key,{})[row["world"]]=tokens["row_bindings"][row["id"]]
        for (_,arm),twins in pairs.items():
            a,b=(twins[w] for w in ("A","B"))
            if a["tokens"]!=b["tokens"] or (arm=="focused_erased" and a!=b):
                raise ValueError("Unmatched B2 length or changed erased bytes")
    assert seal()==frozen
    out.mkdir(parents=True,exist_ok=False)
    fields={"public":public,"tokens":tokens,"gold":gold}
    for key,value in fields.items():
        write_new(out/f"{key}.json",value)
    proof={"experiment":EXPERIMENT,"status":"source_tokenizer_verified","source_manifest_sha256":sha(SOURCE),
           "payload_sha256":{k:digest(v) for k,v in fields.items()},"tokenizer_pins":pins,
           "verified_focused":count,"roots":settings["source_roots"],"prompts":len(public),
           "b2_batches":len(pairs),"max_position":max(r["tokens"] for r in tokens["row_bindings"].values())+2,
           "prior_exposed_prompts":len(prior),"prior_complete_labelled_graphs":len(seen),
           "pretrained_forwards":0,"sealed_test_accessed":False,
           "max_sampled_private_bytes":max(s["private_bytes"] for s in samples)}
    write_new(out/"SOURCE_CHECK.json",proof)
    print(proof,flush=True)


def load(folder,fields):
    proof=json.loads(CHECK.read_bytes())
    if proof["source_manifest_sha256"]!=sha(SOURCE) or json.loads(SOURCE.read_bytes())!=seal():
        raise ValueError("The frozen source receipt changed")
    rows={name:json.loads((folder/f"{name}.json").read_bytes()) for name in fields}
    if any(digest(value)!=proof["payload_sha256"][key] for key,value in rows.items()):
        raise ValueError("A frozen source payload changed")
    return rows,proof


def worker_inputs(folder,snapshot,check):
    payload,proof=load(folder,("public","tokens"))
    tok,pins=tokenizer(snapshot)
    if pins!=proof["tokenizer_pins"] or inputs(tok,payload["public"],snapshot,check)!=payload["tokens"]:
        raise ValueError("Worker tokenizer did not reproduce the public bindings")
    return payload


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze",action="store_true")
    cli.add_argument("--snapshot",type=Path)
    cli.add_argument("--work",type=Path)
    args=cli.parse_args()
    if not __debug__:
        cli.error("Require source verification assertions")
    if args.freeze:
        write_new(SOURCE,seal())
    elif args.snapshot and args.work:
        prepare(args.snapshot,args.work)
    else:
        cli.error("Require the pinned snapshot and a fresh source directory")


if __name__=="__main__":
    main()
