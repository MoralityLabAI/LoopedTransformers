"""Development-only Nanbeige NAN-COMP-04 native-two-loop inference runner.

Isolated model OS process: reads public cohort ONLY. Host labels, EvalMath
oracle and 168-case sealed ARG-03 test remain entirely outside this process.
Pretrained inference requires explicit --execute-development and a source lock.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from copy import deepcopy
from hashlib import sha1,sha256
import json
from pathlib import Path
import time

from .cohort import HERE,PUBLIC_KEYS,fingerprint
from .prompt import body,encode


def sha(path:Path)->str:
    return sha256(path.read_bytes()).hexdigest()


def forbid_test_paths(*paths:Path):
    for path in paths:
        if any(part.lower() in {"test","heldout","test_full168"} for part in path.parts):
            raise ValueError("original sealed ARG-03 test location is forbidden")


def read_public(source_dir:Path)->tuple[list[dict],dict]:
    forbid_test_paths(source_dir)
    receipt=json.loads((source_dir/"source_receipt.json").read_text(encoding="utf-8"))
    if (receipt.get("schema")!="nan_comp04.sourceonly.v1"
        or receipt.get("model_inference") is not False
        or receipt.get("original_ARG03_168_test_opened") is not False
        or receipt.get("reserved_confirmation_sources_generated") is not False):
        raise ValueError("development-only source receipt invalid")
    if receipt["config_sha256"]!=sha(HERE/"config.json"):
        raise ValueError("registered source configuration mismatch")
    if receipt["files_sha256"]["public.jsonl"]!=sha(source_dir/"public.jsonl"):
        raise ValueError("public-source file hash changed")
    rows=[json.loads(line) for line in (source_dir/"public.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(rows)!=220 or len({r["case_id"] for r in rows})!=220:
        raise ValueError("expected complete 220 CAL/DEV public source rows")
    for r in rows:
        if set(r)!=PUBLIC_KEYS or r["split"] not in ("calibration","development"):
            raise ValueError("nonpublic or unregistered source row")
        if r["public_sha256"]!=fingerprint({k:v for k,v in r.items()
                                               if k not in ("case_id","public_sha256")}):
            raise ValueError("source public identity mismatch")
        body(r)  # validates visible source-only fields, with no oracle.
    return rows,receipt


def group_rows(rows:list[dict],split:str,limit_roots:int):
    if split not in ("calibration","development") or limit_roots<0:
        raise ValueError("invalid development scope")
    by=OrderedDict()
    for r in rows:
        if r["split"]==split:
            by.setdefault(r["root_group"],[]).append(r)
    count=4 if split=="calibration" else 16
    if len(by)!=count or any(len(v)!=11 for v in by.values()):
        raise ValueError("expected complete registered root grouping")
    selected=list(by.items())[:limit_roots or None]
    return selected


def lock_source():
    manifest_path=HERE/"SOURCE_LOCK.json"
    if not manifest_path.is_file():
        raise RuntimeError("no source lock: commit SOURCE_LOCK.json before model inference")
    lock=json.loads(manifest_path.read_text())
    if lock.get("schema")!="nan_comp04.preinference_git_blob_lock.v1":
        raise ValueError("source lock schema")
    for name,digest in lock["source_git_blobs"].items():
        p=HERE/name
        data=p.read_bytes()
        actual=sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
        if actual!=digest:
            raise RuntimeError("locked source moved: "+name)
    if lock.get("sealed_168_test_accessed") is not False:
        raise RuntimeError("unsafe registration")
    return lock


def tokenize_root(tokenizer,snapshot,items:list[dict],max_prompt:int):
    tokens={}
    tiers={}
    for r in items:
        ids=encode(tokenizer,snapshot,r)
        if not ids or len(ids)>max_prompt:
            raise ValueError("prompt exceeds registered token budget")
        tokens[r["case_id"]]=(ids,r["kind"])
        tiers.setdefault(r["tier"],[]).append((r,ids))
    for tier in ("early","erased"):
        pairs=tiers[tier]
        if len(pairs)!=2 or pairs[0][1]!=pairs[1][1]:
            raise AssertionError("N/R hidden-target prompt parity failed: "+tier)
    for tier in ("first","cross"):
        pairs=tiers[tier]
        if len({len(ids) for _,ids in pairs})!=1:
            raise AssertionError("operator targets expose a prompt-length side channel: "+tier)
    return tokens


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir",type=Path,required=True)
    p.add_argument("--runs",type=Path,required=True)
    p.add_argument("--compactification-root",type=Path,required=True)
    p.add_argument("--snapshot",type=Path,required=True)
    p.add_argument("--backend",choices=("bf16","nf4"),required=True)
    p.add_argument("--split",choices=("calibration","development"),default="development")
    p.add_argument("--limit-roots",type=int,default=2)
    p.add_argument("--execute-development",action="store_true")
    p.add_argument("--resume",action="store_true")
    a=p.parse_args(argv)
    forbid_test_paths(a.source_dir,a.runs)
    rows,receipt=read_public(a.source_dir)
    selected=group_rows(rows,a.split,a.limit_roots)
    cfg=json.loads((HERE/"config.json").read_text(encoding="utf-8"))
    planned={"status":"source-only-dry-run" if not a.execute_development else "running",
             "schema":"nan_comp04.native2.modelonly.v1",
             "backend":a.backend,"split":a.split,"roots_selected":[k for k,_ in selected],
             "source_public_sha256":receipt["files_sha256"]["public.jsonl"],
             "source_receipt_sha256":sha(a.source_dir/"source_receipt.json"),
             "checkpoint_revision":cfg["checkpoint_revision"],"BPD_transfer_attempted":False,
             "original_ARG03_168_test_accessed":False,"root_limit":a.limit_roots,
             "torch_inference_executed":bool(a.execute_development)}
    if not a.execute_development:
        print(json.dumps(planned,indent=2,sort_keys=True));return 0
    lock=lock_source()
    # Torch is imported only after the source and source-lock gates pass.
    import torch
    from experiments.nan_bpd01.backend import load_backend
    from experiments.nan_comp02.head import OperatorHead
    from experiments.nan_comp03.prompt import aliases
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import assert_oracle_isolated
    from .decode import generate
    if not torch.cuda.is_available():
        raise RuntimeError("4GiB CUDA experiment requires a working CUDA device")
    if not (a.snapshot/"tokenizer_config.json").is_file():
        raise FileNotFoundError("pinned Nanbeige tokenizer_config.json absent")
    torch.cuda.set_per_process_memory_fraction(cfg["compute"]["cuda_memory_fraction"])
    assert_oracle_isolated()
    start=time.perf_counter()
    # Backend source explicitly verifies all checkpoint weights and keeps
    # embeddings/full output head resident in host memory.
    loop,head0,tok,entry=load_backend(
        a.compactification_root,a.snapshot,a.backend,
        chunk_rows=cfg["compute"]["vocabulary_chunk_rows"])
    if entry["revision"]!=cfg["checkpoint_revision"]:
        raise ValueError("checkpoint revision drift")
    head=OperatorHead(head0.weight,aliases(tok),chunk_rows=cfg["compute"]["vocabulary_chunk_rows"])
    saved={
        **planned,"status":"running","runtime_torch":torch.__version__,
        "source_lock_git_blobs":lock["source_git_blobs"],
        "output_head_on_cpu":bool(head.weight.device.type=="cpu"),
        "existing_completed_roots":[],
        "candidate_chunk_size":4,
    }
    output=a.runs/a.backend/a.split
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/"manifest.json"
    if manifest.exists():
        if not a.resume:
            raise FileExistsError("run output already exists, refuse silent overwrite")
        prior=json.loads(manifest.read_text(encoding="utf-8"))
        for key in ("schema","split","backend","roots_selected",
                    "source_public_sha256","source_receipt_sha256"):
            if prior.get(key)!=saved[key]:
                raise ValueError("resume source identity mismatch: "+key)
    else:
        manifest.write_text(json.dumps(saved,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    for root,items in selected:
        target=output/("root_"+root+".json")
        if target.exists():
            if not a.resume:
                raise FileExistsError("existing model root receipt")
            prev=json.loads(target.read_text(encoding="utf-8"))
            if prev.get("source_public_sha256")!=saved["source_public_sha256"]:
                raise ValueError("root resume identity mismatch")
            saved["existing_completed_roots"].append(root)
            continue
        ids=tokenize_root(tok,a.snapshot,items,cfg["generation"]["model_prompt_cap_tokens"])
        unique=OrderedDict()
        owners={}
        for cid,(token_ids,kind) in ids.items():
            key=(kind,tuple(token_ids))
            if key not in unique:
                unique[key]=(cid,token_ids,kind)
            owners[cid]=unique[key][0]
        outcomes={}
        compact=list(unique.values())
        for start_idx in range(0,len(compact),4):
            sub=compact[start_idx:start_idx+4]
            block={cid:(seq,kind) for cid,seq,kind in sub}
            outcomes.update(generate(loop,head,tok.tok,block,
                                     max_class=cfg["generation"]["max_class_tokens"],
                                     max_numeric=cfg["generation"]["max_coefficient_tokens"]))
        cases=[]
        for item in items:
            cid=item["case_id"]
            results=deepcopy(outcomes[owners[cid]])
            cases.append({"case_id":cid,"public_sha256":item["public_sha256"],
                          "kind":item["kind"],"tier":item["tier"],
                          "reused_from_case_id":owners[cid] if owners[cid]!=cid else None,
                          "result":results})
        target.write_text(json.dumps({
            "schema":"nan_comp04.modelonly.root.v1","root_group":root,
            "split":a.split,"backend":a.backend,"source_public_sha256":saved["source_public_sha256"],
            "cases":cases,"BPD_transfer_attempted":False,
            "original_ARG03_168_test_accessed":False,
            },sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")
        print(json.dumps({"backend":a.backend,"root_group":root,"completed":len(cases),
                          "unique_prompt_runs":len(unique)}),flush=True)
    saved["status"]="completed"
    saved["elapsed_seconds"]=round(time.perf_counter()-start,3)
    saved["peak_cuda_allocated_bytes"]=torch.cuda.max_memory_allocated()
    saved["peak_cuda_reserved_bytes"]=torch.cuda.max_memory_reserved()
    saved["output_head_on_cpu"]=bool(head.weight.device.type=="cpu")
    manifest.write_text(json.dumps(saved,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    assert_oracle_isolated()
    print(json.dumps({"status":saved["status"],"elapsed_seconds":saved["elapsed_seconds"],
                      "peak_cuda_allocated_bytes":saved["peak_cuda_allocated_bytes"],
                      "peak_cuda_reserved_bytes":saved["peak_cuda_reserved_bytes"]}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
