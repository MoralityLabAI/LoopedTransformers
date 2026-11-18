"""Model-only unforced full-vocabulary greedy decoding of native two-loop Nanbeige.

No host labels, EvalMath oracle, candidate search, forced operator or logit mask.
Reuses the existing grouped two-pass checkpoint adapter and CPU vocabulary head.
"""
from __future__ import annotations

import hashlib
import json
import torch

from experiments.nan_comp03.decode import grouped_states
from .parser import parse


def sequence_hash(ids:list[int])->str:
    return hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()


@torch.inference_mode()
def generate(loop,head,tok,items:dict[str,tuple[list[int],str]],
             *,max_class:int=8,max_numeric:int=16)->dict:
    """Items map opaque case ids to (model-visible token ids, task kind)."""
    if not items:
        raise ValueError("empty model batch")
    rec={}
    for cid,(ids,kind) in items.items():
        if kind not in ("classification","coefficient") or not ids:
            raise ValueError("unregistered model input")
        rec[cid]={"initial_prompt_ids":list(ids),
                  "initial_prompt_sha256":sequence_hash(ids),
                  "kind":kind,"generated_ids":[],"steps":[],
                  "parser":{"status":"pending","reason":"grammar_prefix"}}
    active=list(rec)
    for step in range(max(max_class,max_numeric)):
        if not active:
            break
        prefixes=[items[cid][0]+rec[cid]["generated_ids"] for cid in active]
        states=grouped_states(loop,prefixes)
        for cid,prefix,h in zip(active,prefixes,states,strict=True):
            item=rec[cid]
            score=head.score(h)
            token=int(score["free_top_token_id"])
            item["generated_ids"].append(token)
            text=tok.decode(item["generated_ids"],skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
            cap=max_class if item["kind"]=="classification" else max_numeric
            at_eos=(token==tok.eos_token_id)
            parsed=parse(text,item["kind"],final=(at_eos or step+1>=cap))
            item["steps"].append({"step":step+1,"prefix_sha256":sequence_hash(prefix),
                                  "decoded_prefix":text,"score":score})
            item["parser"]=parsed
            if parsed["status"]!="pending":
                item["stop_reason"]="eos" if at_eos else parsed["reason"]
        active=[cid for cid in active if rec[cid]["parser"]["status"]=="pending"]
        del states
    for item in rec.values():
        item["generated_text"]=item["steps"][-1]["decoded_prefix"]
        item["tokens_generated"]=len(item["generated_ids"])
        if item["parser"]["status"]=="pending":
            raise AssertionError("maximum answer tokens must terminate every item")
    return rec
