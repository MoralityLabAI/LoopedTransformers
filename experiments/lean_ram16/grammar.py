"""RAM-16 general source-header-only structural macro grammar.

Registered amendment A precedes every RAM16 native-training observation.
Vendor-pinned RAM14 candidates remain an identical ordered prefix; generic
source-derived templates follow within the original three family caps.
Never read the target theorem's reference proof as an input feature.
"""
from __future__ import annotations
import hashlib
import json
import re
from dataclasses import dataclass
import ram14_frozen as old

FAMILIES=("nat_induction","simp_prior","rewrite_prior")
MAX_PER_FAMILY=16
MAX_TOTAL=48
IDENT=r"[A-Za-z_][A-Za-z0-9_']*"
TERM=r"[A-Za-z_][A-Za-z0-9_'.]*"
STOP=frozenset(("theorem","lemma","def","abbrev","Nat","Int","Bool",
    "Type","Prop","List","True","False","fun","forall","∀","by","where",
    "and","or","exists","Eq","And","Or","Exists","Nonempty","not",
    "zero","one","some","none","match","with","if","then","else",
    "h","n","m","x","y","z","a","b","c","f","g","p","q","r","s",
    "hf","ha","hx","h₀","h₁","ih","α","β"))
Candidate=old.Candidate
git_blob=old.git_blob
mask_comments=old.mask_comments


def target_header(source:str,theorem:str)->tuple[str,str]:
    prefix,header=old.statement_scope(source,theorem)
    if not header.startswith(("theorem ","lemma ")):
        raise ValueError("source parsed theorem without a valid original declaration")
    if ":= by" in header:
        raise ValueError("reference proof leaked into theorem header")
    return prefix,header


def params(header:str)->list[tuple[str,str]]:
    """Only explicitly parenthesized binder declarations in target header."""
    return [(n,ty.strip()) for n,ty in re.findall(
        rf"\(\s*({IDENT})\s*:\s*([^()]+)\)",header)
        if re.fullmatch(IDENT,n)]


def meaningful_tokens(text:str,names:set[str])->set[str]:
    return {w for w in old.word_tokens(text)
            if len(w)>2 and w not in STOP and w not in names}


def source_fields(prefix:str,binders:list[tuple[str,str]])->list[str]:
    """Read structure-field identifiers strictly from a prior structure body."""
    names={n for n,ty in binders}
    out=[]
    structures=list(re.finditer(rf"^structure\s+({IDENT})\b",prefix,re.M))
    for match in structures:
        typename=match.group(1)
        first=prefix.find("\n",match.start())+1
        if not first:continue
        ending=re.search(r"^(?:theorem|lemma|def|abbrev|structure|end|namespace|section|#)\b",
                         prefix[first:],re.M)
        end=first+ending.start() if ending else len(prefix)
        body=prefix[first:end]
        fields=re.findall(rf"^\s+({IDENT})\s*:",body,re.M)
        for param,ty in binders:
            if ty.startswith(typename+" ") or ty==typename:
                for field in fields[:3]:
                    action=f"{param}.{field}"
                    if re.fullmatch(rf"{IDENT}\.{IDENT}",action) and action not in out:
                        out.append(action)
    return out[:3]


def _safe(family:str,script:str)->bool:
    if family not in FAMILIES or not script or any(s in script for s in (
            "\n","\r","sorry","admit","run_tac","axiom","unsafe","--","/-",
            "set_option")):return False
    if family=="nat_induction":
        return re.fullmatch(rf"induction {IDENT}(?: generalizing {IDENT})?"
              rf"(?: <;> (?:simp|simp_all)(?: \[{TERM}(?:, {TERM}){{0,7}}\])?)?",
              script) is not None
    if family in ("simp_prior","rewrite_prior"):
        return re.fullmatch(rf"(?:simp|simp_all|simpa|rw) \[{TERM}(?:, {TERM}){{0,7}}\]",
                           script) is not None
    return False


def generate(source:str,theorem:str)->list[Candidate]:
    previous,header=target_header(source,theorem)
    result=list(old.generate(source,theorem))
    if any(not _safe(x.family,x.script) for x in result):
        raise AssertionError("original RAM14 source-generated action is invalid")
    counts={family:sum(x.family==family for x in result) for family in FAMILIES}
    seen={x.script for x in result}
    binders=params(header)
    bindernames={n for n,_ in binders}
    target_tokens=meaningful_tokens(header,bindernames)
    defined=[m.group(1) for m in old.DEFINE.finditer(previous)
             if m.group(1) in old.word_tokens(header)]
    defined=list(dict.fromkeys(defined))[:2]
    earlier=[]
    for name,prior in old.earlier_headers(previous):
        if name==theorem or not old.NAME.fullmatch(name):continue
        # Previous lemma source headers may mention local f and x binders.
        priors=set(re.findall(rf"\(\s*({IDENT})\s*:",prior))
        overlap=target_tokens & meaningful_tokens(prior,priors)
        if overlap and name not in earlier:
            earlier.append(name)
    earlier=earlier[:3]
    fields=source_fields(previous,binders)
    shared=(defined+fields+earlier[:1])[:5]
    def add(family,script,witnesses):
        if not _safe(family,script):
            raise ValueError("new structural candidate outside preregistered syntax")
        if script in seen or len(result)>=MAX_TOTAL or counts[family]>=MAX_PER_FAMILY:
            return
        if not witnesses or any(not isinstance(w,str) for w in witnesses):
            raise ValueError("missing source-visible action provenance")
        seen.add(script)
        counts[family]+=1
        result.append(Candidate(family,script,tuple(witnesses)))

    nat=[n for n,ty in binders if ty=="Nat"][:2]
    gen=[n for n,ty in binders if ty!="Nat" and n not in nat][:2]
    for n in nat:
        # First place richer type-safe induction forms, then plain root
        # induction. Each is one atomic candidate under the 48-probe budget.
        for x in gen:
            for simp in ("simp_all","simp"):
                for lemmas in (shared[:4],defined,fields,[]):
                    args=list(dict.fromkeys(lemmas))
                    bracket=f" [{', '.join(args)}]" if args else ""
                    script=f"induction {n} generalizing {x} <;> {simp}{bracket}"
                    add("nat_induction",script,[
                        f"binder:{n}:Nat",f"generalizing:{x}",
                        *[f"prior_or_field:{v}" for v in args]])
            add("nat_induction",f"induction {n} generalizing {x}",
                [f"binder:{n}:Nat",f"generalizing:{x}"])
        for simp in ("simp_all","simp"):
            for lemmas in (shared[:4],defined,fields,[]):
                args=list(dict.fromkeys(lemmas))
                suffix=f" [{', '.join(args)}]" if args else ""
                add("nat_induction",f"induction {n} <;> {simp}{suffix}",
                    [f"binder:{n}:Nat",*[f"prior_or_field:{v}" for v in args]])
        add("nat_induction",f"induction {n}",[f"binder:{n}:Nat"])

    for names in (defined+fields,defined,fields,earlier[:2]):
        cleaned=list(dict.fromkeys(names))
        if not cleaned:continue
        for simp in ("simp","simp_all"):
            add("simp_prior",f"{simp} [{', '.join(cleaned)}]",
                [f"prior_or_field:{v}" for v in cleaned])

    for name in earlier:
        for tact in ("rw","simp","simpa"):
            add("rewrite_prior",f"{tact} [{name}]",
                [f"earlier_theorem:{name}"])
        for field in fields[:1]:
            for simp in ("simp","simpa"):
                add("rewrite_prior",f"{simp} [{name}, {field}]",
                    [f"earlier_theorem:{name}",f"field:{field}"])
    if len(result)>MAX_TOTAL or any(counts[n]>MAX_PER_FAMILY for n in FAMILIES):
        raise AssertionError("registered 48/16 structural caps exceeded")
    return result


def describe(source:str,theorem:str)->dict:
    previous,header=target_header(source,theorem)
    old_actions=old.generate(source,theorem)
    actions=generate(source,theorem)
    if actions[:len(old_actions)]!=old_actions:
        raise AssertionError("original historical RAM14 candidate order changed")
    return dict(theorem=theorem,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        statement_sha256=hashlib.sha256(header.encode()).hexdigest(),
        prior_header_prefix_sha256=hashlib.sha256(previous.encode()).hexdigest(),
        original_ram14_prefix_length=len(old_actions),candidates=[c.as_dict() for c in actions],
        counts={f:sum(c.family==f for c in actions) for f in FAMILIES})
