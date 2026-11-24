"""Generate fresh source roots and verify isolated host and worker custody."""
import argparse
import json
from pathlib import Path

from experiments.nan_delay39.source import digest, independently_verify, source_cases
from experiments.nan_gated46.source import inputs
from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha, tokenizer, verify_source
from experiments.nan_relabel68.source import build as previous_labels


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = HERE/"registration.json"
SOURCE = HERE/"SOURCE_MANIFEST.json"
CHECK = HERE/"SOURCE_CHECK.json"
BUDGET = {"minimum_start_available_RAM_bytes": 5*2**30, "minimum_available_RAM_bytes": 4*2**30,
          "maximum_private_bytes": 768*2**20, "maximum_seconds": 120}


def seal():
    paths = [HERE/n for n in (".gitattributes","__init__.py","registration.json","PROTOCOL.md","source.py")]
    paths += [REPO/n for n in ("experiments/nan_comp14/fixtures/public.jsonl","experiments/nan_comp14/sources.py",
                              "experiments/nan_delay39/source.py","experiments/nan_delay39/SOURCE_CHECK.json",
                              "experiments/nan_gated46/source.py","experiments/nan_gated46/SOURCE_CHECK.json",
                              "experiments/nan_route58/SOURCE_CHECK.json","experiments/nan_relabel68/source.py",
                              "experiments/nan_relabel68/VERIFICATION.json","experiments/nan_readout34/tokenizer_check.py")]
    return {"experiment":"NAN-WRITER-77","files_sha256":{p.relative_to(REPO).as_posix():sha(p) for p in paths}}


def exposed():
    data = [json.loads(line) for line in (REPO/"experiments/nan_comp14/fixtures/public.jsonl").read_bytes().splitlines()]
    for name, key in (("nan_delay39","source_digest"),("nan_gated46","public_sha256"),("nan_route58","public_sha256")):
        receipt = json.loads((REPO/"experiments"/name/"SOURCE_CHECK.json").read_bytes())
        rows = json.loads((REPO/"runs"/name/"source/public.json").read_bytes())
        if digest(rows)!=receipt[key]:
            raise ValueError("A prior exposed roster differs from its receipt")
        data.extend(rows)
    labeled,_ = previous_labels()
    prior = json.loads((REPO/"experiments/nan_relabel68/VERIFICATION.json").read_bytes())
    if digest(labeled)!=prior["payload_sha256"]["public"]:
        raise ValueError("The prior label-disjoint prompts differ from their receipt")
    data.extend(labeled)
    return {row["prompt"] for row in data if row["arm"]=="focused"}


def prepare(snapshot,out):
    cfg = json.loads(CONFIG.read_bytes())
    frozen = seal()
    if json.loads(SOURCE.read_bytes())!=frozen or snapshot.name!=cfg["checkpoint_revision"]:
        raise ValueError("Wrong source freeze or tokenizer checkpoint")
    with lease(BUDGET) as (check,samples):
        verify_source()
        prior = exposed()
        public,gold = source_cases(cfg)
        for row in public+gold:
            for key in ("id","root"):
                row[key] = row[key].replace("delay39:","writer77:",1)
            row["split"] = "fresh_development"
        count = independently_verify(public,gold,cfg["source_roots"],prior)
        tok,pins = tokenizer(snapshot)
        tokens = inputs(tok,public,snapshot,check)
        groups = {(r["root"],r["arm"],r["world"]):tokens["row_bindings"][r["id"]] for r in public}
        for root in {r["root"] for r in public}:
            for arm in cfg["arms"]:
                left,right = (groups[root,arm,w] for w in ("A","B"))
                if left["tokens"]!=right["tokens"] or (arm=="focused_erased" and left!=right):
                    raise RuntimeError("The B2 twins did not preserve the registered shape")
    assert seal()==frozen
    out.mkdir(parents=True,exist_ok=False)
    payloads = {"public":public,"gold":gold,"tokens":tokens}
    for name,value in payloads.items():
        write_new(out/f"{name}.json",value)
    proof = {"status":"source_and_tokenizer_verified","source_manifest_sha256":sha(SOURCE),
             "payload_sha256":{k:digest(v) for k,v in payloads.items()},"tokenizer_pins":pins,
             "roots":cfg["source_roots"],"prompts":len(public),"verified_focused":count,
             "prior_focused_prompt_count":len(prior),"max_position":2+max(x["tokens"] for x in groups.values()),
             "pretrained_forwards":0,"sealed_test_accessed":False,
             "max_sampled_private_bytes":max(s["private_bytes"] for s in samples)}
    write_new(out/"SOURCE_CHECK.json",proof)
    print(proof,flush=True)


def load(folder,fields):
    proof = json.loads(CHECK.read_bytes())
    if proof["source_manifest_sha256"]!=sha(SOURCE) or json.loads(SOURCE.read_bytes())!=seal():
        raise ValueError("Frozen source registration changed")
    values = {}
    for field in fields:
        value = json.loads((folder/f"{field}.json").read_bytes())
        if digest(value)!=proof["payload_sha256"][field]:
            raise ValueError(f"Incorrect frozen source payload {field}")
        values[field] = value
    return values,proof


def worker_inputs(folder,snapshot,check):
    payload,proof = load(folder,("public","tokens"))
    tok,pins = tokenizer(snapshot)
    if pins!=proof["tokenizer_pins"] or inputs(tok,payload["public"],snapshot,check)!=payload["tokens"]:
        raise ValueError("Runtime inputs changed after tokenization")
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze",action="store_true")
    p.add_argument("--snapshot",type=Path)
    p.add_argument("--work",type=Path)
    args = p.parse_args()
    if not __debug__:
        p.error("Enable Python assertions")
    if args.freeze:
        write_new(SOURCE,seal())
        return
    if not args.snapshot or not args.work:
        p.error("Require the pinned snapshot and fresh output")
    prepare(args.snapshot,args.work)


if __name__=="__main__":
    main()
