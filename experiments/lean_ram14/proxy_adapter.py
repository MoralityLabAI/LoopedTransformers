"""RAM-14-to-frozen-RAM08 inference compatibility adapter.

This is a source-only construction layer, not a tested model or new
prospective theorem score. The frozen RAM08 policies have an exactly
20-family embedding; map new structurally generated actions onto the
nearest pre-existing embedding type without changing model parameters.

The public RAM14 grammar supplies the actual source-visible candidate list.
All policies must receive the SAME concrete expanded pool.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import re

RAM08_FAMILIES=(
 "omega","rfl","decide","simp_all","all_simp","assumption","exact_local",
 "exact_typed","simpa_local","intro","intro_named","constructor","rw_local",
 "apply_local","rcases_local","cases_local","induction_list",
 "simp_only_visible","simp_all_only_visible","dsimp_only_visible")
EXTRA_FAMILIES=("nat_induction","simp_prior","rewrite_prior")
MAX_EXTRA=32
TYPED_WORD=r"[A-Za-z_][A-Za-z0-9_']*"
TERM=r"[A-Za-z_][A-Za-z0-9_'.]*"


def family_proxy(family:str,script:str)->str:
    if family=="nat_induction":
        return "induction_list"
    if family=="simp_prior":
        return "simp_all_only_visible"
    if family=="rewrite_prior":
        if script.startswith("rw ["):return "rw_local"
        if script.startswith("simpa ["):return "simpa_local"
        if script.startswith("simp ["):return "simp_all_only_visible"
    raise ValueError("unknown registered structural tactic family")


def safe_extended_script(family:str,script:str)->bool:
    if family not in EXTRA_FAMILIES or not isinstance(script,str):
        return False
    if any(tok in script for tok in ("\n","\r","sorry","admit","axiom","run_tac","set_option","--","/-")):
        return False
    # Multipass one-shot macros are one registered atomic candidate:
    # the original policy sees their script tokens as a single action.
    if family=="nat_induction":
        pattern=rf"induction {TYPED_WORD} generalizing {TYPED_WORD}(?: <;> (?:simp|simp_all)(?: \[(?:{TERM})(?:, {TERM}){{0,7}}\])?)?"
        return re.fullmatch(pattern,script) is not None
    if family in ("rewrite_prior","simp_prior"):
        pat=rf"(?:rw|simp|simpa|simp_all) \[(?:{TERM})(?:, {TERM}){{0,7}}\]"
        return re.fullmatch(pat,script) is not None
    return False


@dataclass(frozen=True)
class CandidateReceipt:
    original_family:str
    embedding_family:str
    script:str
    source_visible_witnesses:tuple[str,...]
    declared_posthoc_origin:bool

    def as_dict(self):
        return dict(original_family=self.original_family,
                    embedding_family=self.embedding_family,
                    script=self.script,
                    source_visible_witnesses=list(self.source_visible_witnesses),
                    declared_posthoc_origin=self.declared_posthoc_origin)


def extend(original:list[tuple[str,str]], generated:list[dict],
           *, extra_cap:int=MAX_EXTRA) -> tuple[list[tuple[str,str]], list[CandidateReceipt]]:
    """Extend without editing original candidates or any frozen model tensors.

    generated must be the public RAM14 source-derived grammar's complete
    ordered Candidate.as_dict output at this theorem/source. In an
    executable prospective scorer, recompute from pinned source immediately
    before calling this function and include the source SHA in its receipt.
    """
    if len(original)!=len(set(original)):
        raise ValueError("duplicate originally registered RAM08 candidates")
    if any(f not in RAM08_FAMILIES for f,_ in original):
        raise ValueError("unknown original family")
    if not 0<=extra_cap<=MAX_EXTRA:
        raise ValueError("extra budget outside registered compatibility cap")
    result=list(original)
    provenance=[]
    seen_scripts={script for _,script in original}
    for record in generated:
        if len(provenance)>=extra_cap:break
        fam=record["family"];script=record["script"]
        witnesses=tuple(record["witnesses"])
        if not witnesses or not all(isinstance(w,str) and
                                    w.startswith(("binder:","generalizing:",
                                        "prior_or_field:","earlier_theorem:","field:"))
                                    for w in witnesses):
            raise ValueError("missing typed source provenance")
        if not safe_extended_script(fam,script):
            raise ValueError("source-derived script outside frozen expanded grammar")
        if script in seen_scripts:continue
        proxy=family_proxy(fam,script)
        result.append((proxy,script))
        provenance.append(CandidateReceipt(fam,proxy,script,witnesses,True))
        seen_scripts.add(script)
    if result[:len(original)]!=original:
        raise AssertionError("original 20-family candidate order changed")
    if any(x.embedding_family not in RAM08_FAMILIES for x in provenance):
        raise AssertionError("model would receive invalid family embedding ID")
    return result,provenance


def fingerprint(original:list[tuple[str,str]],generated:list[dict],provenance:list[CandidateReceipt]):
    """Tamper-evident source-less adapter receipt for any future run."""
    raw=json.dumps(dict(original=original,generated=generated,
                        extension=[x.as_dict() for x in provenance]),
                   ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
    return hashlib.sha256(raw).hexdigest()
