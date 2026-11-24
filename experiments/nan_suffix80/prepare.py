"""Bind the new source, independent oracle and exact A/B token schedule."""
import argparse
import json
from pathlib import Path

from experiments.nan_delay39.source import digest
from experiments.nan_gated46.source import inputs
from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha,tokenizer,verify_source
from experiments.nan_relabel68.source import build as relabel
from experiments.nan_writer77.source import load as load_writer

from .audit import check as check_source
from .source import build


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
SOURCE=HERE/"SOURCE_MANIFEST.json"
RECEIPT=HERE/"SOURCE_CHECK.json"
REVISION="451ed48c3273ecef7ea8faaa43c31ce529763bb1"
BUDGET={"minimum_start_available_RAM_bytes":5*2**30,"minimum_available_RAM_bytes":4*2**30,
        "maximum_private_bytes":768*2**20,"maximum_seconds":180}


def seal():
    paths=[HERE/n for n in (".gitattributes","__init__.py","PROTOCOL.md","source.py","audit.py","prepare.py")]
    paths += [REPO/n for n in ("experiments/nan_comp14/sources.py","experiments/nan_comp14/fixtures/public.jsonl",
                                "experiments/nan_gym05/fixtures/public.jsonl","experiments/nan_delay39/source.py",
                                "experiments/nan_relabel68/source.py","experiments/nan_relabel68/VERIFICATION.json",
                                "experiments/nan_writer77/SOURCE_CHECK.json","experiments/nan_writer77/source.py",
                                "experiments/nan_gated46/source.py","experiments/nan_readout34/tokenizer_check.py")]
    return {"experiment":"NAN-SUFFIX-80","files_sha256":{p.relative_to(REPO).as_posix():sha(p) for p in paths}}


def prior_prompts():
    old=set()
    for name in ("nan_comp14","nan_gym05"):
        file=REPO/"experiments"/name/"fixtures/public.jsonl"
        old.update(json.loads(line)["prompt"] for line in file.read_bytes().splitlines())
    earlier,_=relabel()
    verified=json.loads((REPO/"experiments/nan_relabel68/VERIFICATION.json").read_bytes())
    if digest(earlier)!=verified["payload_sha256"]["public"]:
        raise ValueError("The exposed RELABEL68 source changed")
    old.update(r["prompt"] for r in earlier)
    bound,_=load_writer(REPO/"runs/nan_writer77/source01",("public",))
    old.update(row["prompt"] for row in bound["public"])
    return old


def schedule(public,bindings):
    groups={}
    for p in public:
        key=p["split"],p["root"],p["arm"],p["remaining"]
        worlds=groups.setdefault(key,{})
        if p["world"] in worlds:
            raise ValueError("Duplicate world in a B2 pair")
        worlds[p["world"]]=p["id"]
    jobs=[]
    across={}
    for i,((split,root,arm,n),sides) in enumerate(sorted(groups.items())):
        if set(sides)!={"A","B"}:
            raise ValueError("A scheduled prefill needs both A/B rows")
        keys=[sides[w] for w in ("A","B")]
        a,b=(bindings[k] for k in keys)
        if a["tokens"]!=b["tokens"] or (arm=="erased" and a!=b):
            raise ValueError("Paired token lengths or erased bytes differ")
        key=root,arm
        if across.setdefault(key,a["tokens"])!=a["tokens"]:
            raise ValueError("Token counts vary across the matched suffix family")
        jobs.append({"ordinal":i,"root":root,"split":split,"arm":arm,"remaining":n,
                     "length":a["tokens"],"source_ids":keys})
    if len(jobs)!=224:
        raise ValueError("The B2 roster changed")
    return jobs


def prepare(snapshot,work):
    frozen=seal()
    if json.loads(SOURCE.read_bytes())!=frozen or snapshot.name!=REVISION:
        raise ValueError("The frozen source or tokenizer snapshot changed")
    with lease(BUDGET) as (check,samples):
        verify_source()
        public,gold=build()
        audit=check_source(public,gold,prior_prompts())
        tok,pins=tokenizer(snapshot)
        encoded=inputs(tok,public,snapshot,check)
        jobs=schedule(public,encoded["row_bindings"])
    assert frozen==seal()
    work.mkdir(parents=True,exist_ok=False)
    values={"public":public,"gold":gold,"tokens":encoded,"jobs":jobs}
    for name,value in values.items():
        write_new(work/f"{name}.json",value)
    receipt={"status":"source_tokenizer_verified","source_manifest_sha256":sha(SOURCE),
             "payload_sha256":{k:digest(v) for k,v in values.items()},"tokenizer_pins":pins,
             "oracle":audit,"prompts":len(public),"b2_batches":len(jobs),
             "max_position":2+max(j["length"] for j in jobs),"pretrained_forwards":0,
             "sealed_test_accessed":False,"max_sampled_private_bytes":max(s["private_bytes"] for s in samples)}
    write_new(work/"SOURCE_CHECK.json",receipt)
    print(receipt,flush=True)


def load(folder,fields):
    receipt=json.loads(RECEIPT.read_bytes())
    if receipt["source_manifest_sha256"]!=sha(SOURCE) or json.loads(SOURCE.read_bytes())!=seal():
        raise ValueError("The frozen source receipt changed")
    values={}
    for name in fields:
        value=json.loads((folder/f"{name}.json").read_bytes())
        if digest(value)!=receipt["payload_sha256"][name]:
            raise ValueError(f"Source payload changed: {name}")
        values[name]=value
    return values,receipt


def worker_inputs(folder,snapshot,check):
    values,receipt=load(folder,("public","tokens","jobs"))
    tok,pins=tokenizer(snapshot)
    tokens=values["tokens"]
    if (pins!=receipt["tokenizer_pins"] or inputs(tok,values["public"],snapshot,check)!=tokens
            or schedule(values["public"],tokens["row_bindings"])!=values["jobs"]):
        raise ValueError("The runtime tokenizer or native schedule changed")
    return values


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze",action="store_true")
    cli.add_argument("--snapshot",type=Path)
    cli.add_argument("--work",type=Path)
    args=cli.parse_args()
    if not __debug__:
        cli.error("Enable assertions")
    if args.freeze:
        write_new(SOURCE,seal())
        return
    if not args.snapshot or not args.work:
        cli.error("Supply the pinned snapshot and fresh output")
    prepare(args.snapshot,args.work)


if __name__=="__main__":
    main()
