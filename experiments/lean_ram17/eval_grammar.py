"""RAM-17 shared concrete original Lean tactic pool for all eight policies.

Original 20-family RAM13 grammar is unchanged, with the preregistered
RAM17 source-only structural family extension added as explicit 23-row
family labels. The original frozen RAM08 proxy policy maps those three
new family labels to its old 20 IDs only at ranking time.
"""
from __future__ import annotations
from collections import Counter
import hashlib,importlib.util,json
from pathlib import Path
import grammar as structure

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def frozen_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ImportError(str(path))
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

OLD=frozen_module("ram16_exact_original_ram13_grammar",
                  ROOT/"experiments/lean_ram13/grammar.py")
FAMILIES=OLD.FAMILIES+tuple(structure.FAMILIES)
signature=OLD.signature
MAX_EXTRA=32
_CONTEXT=None
_ALLOWS=set()
_EXTRAS=[]
_ABLATION=False
_STATS=Counter()

def family_proxy(family:str,script:str)->str:
    if family in OLD.FAMILIES:return family
    if family=="nat_induction":return "induction_list"
    if family=="simp_prior":return "simp_all_only_visible"
    if family=="rewrite_prior":
        if script.startswith("rw ["):return "rw_local"
        if script.startswith("simpa ["):return "simpa_local"
        if script.startswith("simp ["):return "simp_all_only_visible"
    raise ValueError("unknown extended family or script")

def set_context(original_source:str,theorem:str,*,ablation=False):
    global _CONTEXT,_ALLOWS,_EXTRAS,_ABLATION,_STATS
    if not original_source or not theorem:raise ValueError("original source/theorem missing")
    _CONTEXT=dict(source_sha256=hashlib.sha256(original_source.encode()).hexdigest(),
                  theorem=theorem)
    _ABLATION=ablation
    _EXTRAS=[] if ablation else [x.as_dict() for x in structure.generate(original_source,theorem)]
    _ALLOWS={x["script"] for x in _EXTRAS if structure._safe(x["family"],x["script"])}
    if len(_ALLOWS)!=len({x["script"] for x in _EXTRAS}):
        raise AssertionError("unregistered or duplicated structural source candidate")
    _STATS=Counter()

def valid_script(script:str)->bool:
    if OLD.valid_script(script):return True
    return script in _ALLOWS

def candidates(state:dict)->list[tuple[str,str]]:
    if _CONTEXT is None:raise AssertionError("source-only grammar uninitialized")
    old=OLD.candidates(state)
    _STATS["states"]+=1
    _STATS["original_tactics"]+=len(old)
    if _ABLATION:return old
    seen={script for family,script in old}
    new=[]
    for candidate in _EXTRAS:
        if len(new)>=MAX_EXTRA:break
        family,script=candidate["family"],candidate["script"]
        if family not in structure.FAMILIES or not structure._safe(family,script):
            raise AssertionError("candidate outside structural source whitelist")
        if script in seen:continue
        seen.add(script)
        new.append((family,script))
    _STATS["new_tactics"]+=len(new)
    merged=[];j=0
    for index,item in enumerate(old):
        merged.append(item)
        if (index+1)%4==0 and j<len(new):
            merged.append(new[j]);j+=1
    merged.extend(new[j:])
    if Counter(merged)!=Counter(old+new):
        raise AssertionError("every policy must receive same registered candidate multiset")
    if len(merged)!=len(set(merged)):
        raise AssertionError("source candidate pool repeated an actual script")
    return merged

def audit_summary()->dict:
    if _CONTEXT is None:raise AssertionError("no current goal/source")
    return dict(**_CONTEXT,ablation_original_20=_ABLATION,
       structural_candidates_source_generated=len(_EXTRAS),
       source_visible_structural_candidates=_EXTRAS,
       stats=dict(_STATS),new_family_rows=3,
       original_ram13_grammar_untouched=True)
