"""RAM-15 registered shared-source grammar.

Exactly the original 20 RAM-13 families plus bounded RAM-14 structural
actions generated from current source-visible theorem statement/prior headers.
Returns 20-family proxy labels to every frozen policy. All policy controllers
see the same concrete pool and cannot access the original reference proof.
"""
from __future__ import annotations
import hashlib, importlib.util, json, os, sys
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
MATHCAN=Path(os.environ.get("RAM15_MATHCAN",str(ROOT/"_mathcan"))).resolve()

def imported(name:str,path:Path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise RuntimeError(f"missing frozen import: {path}")
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

ORIGINAL=imported("ram15_frozen_ram13_grammar",ROOT/"experiments/lean_ram13/grammar.py")
PUBLIC=imported("ram15_pinned_public_structural",MATHCAN/"research/ram14/grammar.py")
PROXY=imported("ram15_frozen_twenty_family_proxy",ROOT/"experiments/lean_ram14/proxy_adapter.py")
FAMILIES=ORIGINAL.FAMILIES
signature=ORIGINAL.signature

_CONTEXT=None
_ABLATION=False
_GENERATED=[]
_ALLOWED=set()
_COUNTER=Counter()

def set_context(source:str,theorem:str,*,ablation:bool=False):
    """Call once per independent theorem/policy search, prior to any probe."""
    global _CONTEXT,_ABLATION,_GENERATED,_ALLOWED,_COUNTER
    if not isinstance(source,str) or not source or not theorem:
        raise ValueError("source and theorem required")
    _CONTEXT=(hashlib.sha256(source.encode("utf8")).hexdigest(),theorem)
    _ABLATION=ablation
    _GENERATED=[] if ablation else [x.as_dict() for x in PUBLIC.generate(source,theorem)]
    _ALLOWED=set()
    _COUNTER=Counter()


def valid_script(script:str)->bool:
    if ORIGINAL.valid_script(script):return True
    return script in _ALLOWED


def candidates(state:dict):
    if _CONTEXT is None:raise AssertionError("source-only grammar context missing")
    old=ORIGINAL.candidates(state)
    _COUNTER["states_visited"]+=1
    _COUNTER["original_generated"]+=len(old)
    if _ABLATION or not _GENERATED:
        return old
    all_rows,receipts=PROXY.extend(old,_GENERATED,extra_cap=32)
    new=all_rows[len(old):]
    for item in receipts:
        if not PROXY.safe_extended_script(item.original_family,item.script):
            raise AssertionError("invalid expanded registered script")
        _ALLOWED.add(item.script)
    # Interleave one new action after every four original actions to give
    # every policy including fixed native order the same reachable action pool.
    merged=[]
    j=0
    for i,item in enumerate(old):
        merged.append(item)
        if (i+1)%4==0 and j<len(new):
            merged.append(new[j]);j+=1
    merged.extend(new[j:])
    if len(merged)!=len(set(merged)) or Counter(merged)!=Counter(all_rows):
        raise AssertionError("registered source-only grammar changed multiset")
    _COUNTER["structural_generated"]+=len(receipts)
    _COUNTER["structural_unique_scripts"]+=len({v.script for v in receipts})
    return merged


def audit_summary()->dict:
    if _CONTEXT is None:raise AssertionError("grammar not initialized")
    return dict(source_sha256=_CONTEXT[0],theorem=_CONTEXT[1],
                ablation_original_twenty_families=_ABLATION,
                origin="RAM14 source-visible statement/prior-declaration grammar",
                grammar_calls=dict(_COUNTER),
                statically_generated_extra=len(_GENERATED),
                allowed_extra_scripts=len(_ALLOWED),
                generated_actions=_GENERATED)
