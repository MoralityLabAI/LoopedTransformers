"""LRF-01: conservative terminal reward for trusted Horn task contracts.

The only model-owned inputs are a bounded declarative claim and action
indices. Never pass arbitrary Lean text, tactics, imports or identifiers
from an untrusted model to this renderer.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

ALLOWED=frozenset(("propext","Classical.choice","Quot.sound"))
DENY=frozenset(("sorryAx","Lean.ofReduceBool","Lean.ofReduceNat","ofReduceBool","ofReduceNat"))
ROOT="LRFSubmission.submitted"
CONTRACT="LRFSubmission.contract"
APPLIED="LRFSubmission.applied"
MAX_ATOMS=128
MAX_RULES=512
MAX_TRACE=512
TIMEOUT=12

def sha(x):
    return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()

def canonical(task):
    """Validate shape; metadata and historical reference proofs stay outside."""
    n=task["n"]
    if type(n) is not int or not 1<=n<=MAX_ATOMS:raise ValueError("atom count")
    def atom(x):
        if type(x) is not int or not 0<=x<n:raise ValueError("atom")
        return x
    facts=task["facts"];rules=task["rules"]
    if type(facts) is not list or len(set(facts))!=len(facts):raise ValueError("facts")
    if type(rules) is not list or len(rules)>MAX_RULES:raise ValueError("rules")
    result=[]
    for r in rules:
        if type(r) is not list or len(r)!=2 or type(r[0]) is not list or len(r[0])>MAX_ATOMS:
            raise ValueError("rule")
        ps=[atom(x) for x in r[0]]
        if not ps or len(ps)!=len(set(ps)):raise ValueError("rule premises")
        result.append([ps,atom(r[1])])
    return {"n":n,"facts":sorted(atom(x) for x in facts),"rules":result,"goal":atom(task["goal"])}

def proposition(t):
    bind=" ".join(f"P{i}" for i in range(t["n"]))
    facts=[f"P{x}" for x in t["facts"]]
    rules=["("+" → ".join(f"P{x}" for x in [*ps,h])+")" for ps,h in t["rules"]]
    return f"∀ ({bind} : Prop), "+" → ".join(facts+rules+[f"P{t['goal']}"])

def checked_trace(t,trace):
    if type(trace) is not list or len(trace)>MAX_TRACE or any(type(j) is not int for j in trace):
        raise ValueError("malformed trace")
    known={p:f"h{p}" for p in t["facts"]};lines=[]
    for k,j in enumerate(trace):
        if not 0<=j<len(t["rules"]):raise ValueError("unknown rule")
        ps,h=t["rules"][j]
        if h in known or not all(p in known for p in ps):raise ValueError("illegal rule")
        lines.append(f"  have s{k} : P{h} := r{j} "+" ".join(known[p] for p in ps))
        known[h]=f"s{k}"
    if t["goal"] not in known:return None
    return lines+["  exact "+known[t["goal"]]]

def render(task,candidate):
    expected=canonical(task)
    claim=canonical(candidate.get("claim",task))
    # A different atom universe is not an admissible candidate envelope.
    if claim["n"]!=expected["n"]:raise ValueError("atom universe changed")
    mode=candidate.get("mode","trace")
    unused=candidate.get("unused_axiom",False)
    if type(unused) is not bool:raise ValueError("unused flag")
    if mode not in ("trace","fixture_axiom","fixture_alias"):raise ValueError("unknown proof mode")
    lines=["import Init","set_option linter.unusedVariables false",
           "namespace LRFSubmission",
           "def intended : Prop := "+proposition(expected),
           "def advertised : Prop := "+proposition(claim)]
    if unused or mode!="trace":
        lines.append("axiom forbiddenPremise : advertised")
    if mode=="fixture_alias":
        lines.append("theorem hiddenAlias : advertised := forbiddenPremise")
    if mode=="trace":
        proof=checked_trace(claim,candidate.get("trace"))
        if proof is None:return None
        args=[f"P{i}" for i in range(claim["n"])]+[f"h{p}" for p in claim["facts"]]+[f"r{i}" for i in range(len(claim["rules"]))]
        lines.extend(["theorem submitted : advertised := by","  intro "+" ".join(args),*proof])
    else:
        lines.append("theorem submitted : advertised := "+("hiddenAlias" if mode=="fixture_alias" else "forbiddenPremise"))
    lines.extend(["end LRFSubmission","#print axioms "+ROOT])
    return "\n".join(lines)+"\n"

def reports(s):
    found={}
    for m in re.finditer(r"'([^']+)' (does not depend on any axioms|depends on axioms: \[([^\]]*)\])",s):
        if m[1] in found:raise ValueError("duplicate axiom report")
        found[m[1]]=sorted(set(x.strip() for x in (m[3] or "").split(",") if x.strip()))
    return found

def compile_source(source,path,lean,timeout=TIMEOUT):
    path.write_text(source,encoding="utf-8")
    log=path.with_suffix(".log")
    start=time.monotonic()
    try:
        p=subprocess.run([lean,str(path.resolve())],cwd=path.parent,capture_output=True,text=True,
                         encoding="utf-8",errors="replace",timeout=timeout)
        text=p.stdout+"\n"+p.stderr
        log.write_text(text,encoding="utf-8")
        return {"status":"compiled" if p.returncode==0 and not re.search(r"(^|\n).*error:",text) else "rejected",
                "returncode":p.returncode,"source_sha256":sha(source),"log_sha256":sha(log.read_bytes()),
                "seconds":time.monotonic()-start,"reports":reports(text),
                "error_excerpt":text[-1800:] if p.returncode else ""}
    except subprocess.TimeoutExpired:
        log.write_text("native compile timed out\n",encoding="utf-8")
        return {"status":"timeout_unknown","source_sha256":sha(source),
                "log_sha256":sha(log.read_bytes()),"seconds":time.monotonic()-start}
    except (OSError,ValueError) as e:
        if not log.exists():log.write_text(str(e),encoding="utf-8")
        return {"status":"infrastructure_unknown","source_sha256":sha(source),
                "log_sha256":sha(log.read_bytes()),"seconds":time.monotonic()-start,
                "error_excerpt":str(e)}

def evaluate_candidate(task,candidate,out:Path,lean="lean",allowed=ALLOWED):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    allowed=frozenset(allowed)
    if DENY&allowed:raise ValueError("forbidden kernel axiom allowlist")
    result={"schema":"LRF-01/v1","kernel":None,"foundation":None,"contract":None,
            "reward":None,"classification":"unknown","compiler_calls":0,
            "task_sha256":sha(json.dumps(canonical(task),sort_keys=True)),
            "candidate_sha256":sha(json.dumps(candidate,sort_keys=True)),
            "allowed_axioms":sorted(allowed)}
    def done(reason):
        result["classification"]=reason
        (out/"receipt.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        return result
    try:
        source=render(task,candidate)
    except (KeyError,ValueError,TypeError) as e:
        result["diagnostic"]=str(e)
        return done("malformed_or_illegal_trace_censored")
    if source is None:return done("incomplete_trace_censored")
    first=compile_source(source,out/"candidate.lean",lean);result["first"]=first
    result["compiler_calls"]+=1
    if first["status"]!="compiled":
        result["kernel"]=False if first["status"]=="rejected" else None
        return done("native_nonclosure_censored" if result["kernel"] is False else "native_unknown_censored")
    if set(first["reports"])!={ROOT}:return done("missing_axiom_receipt_censored")
    actual=set(first["reports"][ROOT])
    result["kernel"]=True
    result["axioms"]=sorted(actual)
    result["excluded_axioms"]=sorted((actual-allowed)|(actual&DENY))
    result["foundation"]=not result["excluded_axioms"]
    replay=source+(
        "\nnamespace LRFSubmission\n"
        "theorem contract : advertised = intended := rfl\n"
        "theorem applied : intended := submitted\n"
        "end LRFSubmission\n"
        "#print axioms "+CONTRACT+"\n#print axioms "+APPLIED+"\n")
    second=compile_source(replay,out/"contract.lean",lean)
    result["second"]=second;result["compiler_calls"]+=1
    if second["status"]=="compiled":
        rs=second["reports"]
        if set(rs)!={ROOT,CONTRACT,APPLIED} or set(rs[ROOT])!=actual or set(rs[APPLIED])!=actual:
            return done("replay_disagreement_censored")
        if set(rs[CONTRACT])-allowed or set(rs[CONTRACT])&DENY:
            return done("contract_axiom_disagreement_censored")
        result["contract"]=True
    elif second["status"]=="rejected":
        # A failed equality/application gate is an unestablished contract,
        # not a proof of semantic inequivalence or falsity.
        result["contract"]=False
    else:return done("contract_unknown_censored")
    if not result["foundation"]:
        result["reward"]=0
        return done("excluded_axiom")
    if not result["contract"]:
        result["reward"]=0
        return done("statement_contract_unestablished")
    result["reward"]=1
    return done("verified_target_proof")
