"""LRF-03: replay bounded native tactics against byte-pinned development roots.

Only evaluate_actions is intended for model calls. The checkout, output folder,
root dictionary, compiler and control-arm selector are verifier-owned.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

COMMIT="f0873083c5a6b0e424d0872db058b8edb8f56f5d"
SOURCE="MathCan/ConstructiveObservationAudit.lean"
BLOB="965687067ddb4ed4f696118ad5fea68286d56665"
NS="MathCan.ConstructiveObservationAudit"
SUB=NS+".lrf03Submitted"
CON=NS+".lrf03Contract"
ALLOW={"propext","Quot.sound"}
MAX_ACTIONS=8
DEADLINE=20.
COMPILE_LIMIT=10.
ROOTS={
    "factor_of_section":{
        "type":"∀ {X : Type u} {Y : Type v} (obs : X → Y) (f : X → X) (sectionOf : Y → X) (rightInv : ∀ y, obs (sectionOf y) = y) (compatible : FiberCompatible obs f), ∀ x, obs (f (sectionOf (obs x))) = obs (f x)",
        "actions":{"apply_section":"exact fun x => compatible (sectionOf (obs x)) x (rightInv (obs x))","skip":"skip"},
        "reference":["apply_section"]},
    "authorization_of_section":{
        "type":"∀ {S : Type u} {O : Type v} {A : Type w} (obs : S → O) (authorized : S → A → Bool) (sectionOf : O → S) (rightInv : ∀ o, obs (sectionOf o) = o) (compatible : AuthorizationFiberCompatible obs authorized), ∀ s a, authorized (sectionOf (obs s)) a = authorized s a",
        "actions":{"apply_section":"exact fun s a => compatible (sectionOf (obs s)) s (rightInv (obs s)) a","skip":"skip"},
        "reference":["apply_section"]},
    "demo_monitor_correct":{
        "type":"∀ (s : Bool × Bool) (a : Bool), demoMonitor (demoObserve s) a = demoAuthorize s a",
        "actions":{"unpack":"rcases s with ⟨visible, hidden⟩","enumerate":"cases visible <;> cases hidden <;> cases a <;> rfl","skip":"skip"},
        "reference":["unpack","enumerate"]},
    "demo_monitor_unknown_denied":{
        "type":"∀ (a : Bool), demoMonitor 2 a = false",
        "actions":{"enumerate":"cases a <;> rfl","skip":"skip"},
        "reference":["enumerate"]},
}
def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data,str) else data).hexdigest()

def blob(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()

def checkout(path):
    git=lambda *args:subprocess.check_output(["git","-C",str(path),*args],text=True).strip()
    if git("rev-parse","HEAD")!=COMMIT:raise ValueError("MathCan HEAD is not pinned")
    if subprocess.run(["git","-C",str(path),"diff","--quiet","HEAD","--"]).returncode!=0:
        raise ValueError("tracked source differs")
    if (path/"lean-toolchain").read_text().strip()!="leanprover/lean4:v4.35.0":
        raise ValueError("Lean toolchain mismatch")
    raw=(path/SOURCE).read_bytes()
    if blob(raw)!=BLOB:raise ValueError("source blob changed")
    return raw.decode("utf-8")

def prefix_header(source,root):
    if root not in ROOTS:raise ValueError("unknown root")
    matches=list(re.finditer(r"(?m)^theorem "+re.escape(root)+r"\b",source))
    if len(matches)!=1:raise ValueError("ambiguous theorem")
    start=matches[0].start()
    pos=source.find(":=",start)
    if pos<0:raise ValueError("missing declaration body")
    header=source[start:pos]
    if re.search(r"(?m)^(?:theorem|def|end)\b",header[len("theorem "):]):
        raise ValueError("unsupported header")
    return source[:start],header

def decode_actions(root,actions):
    if root not in ROOTS:raise ValueError("unknown root")
    if type(actions) is not list or not 1<=len(actions)<=MAX_ACTIONS:
        raise ValueError("invalid action list")
    grammar=ROOTS[root]["actions"]
    if any(type(a) is not str or a not in grammar for a in actions):
        raise ValueError("unregistered tactic ID")
    return [grammar[a] for a in actions]

def render(source,root,actions,fixture=None):
    prefix,header=prefix_header(source,root)
    expected=ROOTS[root]["type"]
    tactics=decode_actions(root,actions)
    # Correct-claim submissions preserve the original binder/type bytes.
    target=header.replace("theorem "+root,"theorem lrf03Submitted",1)
    original=target+":= by\n"+"\n".join("  "+x for x in tactics)+"\n"
    if fixture=="true_claim":
        body="theorem lrf03Submitted : True := True.intro\n"
    elif fixture=="extra_hypothesis":
        body=f"theorem lrf03Submitted : ({expected}) → ({expected}) := fun h => h\n"
    elif fixture=="forbidden_alias":
        body=(f"axiom lrf03Forbidden : {expected}\n"
              f"theorem lrf03Alias : {expected} := @lrf03Forbidden\n"
              f"theorem lrf03Submitted : {expected} := @lrf03Alias\n")
    elif fixture=="unused_axiom":
        body=f"axiom lrf03Forbidden : {expected}\n"+original
    elif fixture is None:body=original
    else:raise ValueError("invalid trusted fixture")
    base=prefix+body
    closing=f"\nend {NS}\n#print axioms {SUB}\n"
    alias=f"\ntheorem lrf03Contract : {expected} := @lrf03Submitted\n"
    # Independent file keeps exactly the same source prefix and candidate.
    return base+closing,base+alias+closing+f"#print axioms {CON}\n",sha(header),len(base.splitlines())+2

def inventories(output):
    reports={}
    for m in re.finditer(r"'([^']+)' (does not depend on any axioms|depends on axioms: \[([^\]]*)\])",output):
        if m[1] in reports:raise ValueError("duplicate axiom report")
        reports[m[1]]=sorted(set(x.strip() for x in (m[3] or "").split(",") if x.strip()))
    return reports

def native(mathcan,source,file,timeout):
    file.write_text(source,encoding="utf-8")
    log=file.with_suffix(".log");t=time.monotonic()
    try:
        p=subprocess.run(["lake","env","lean",str(file.resolve())],cwd=mathcan,
                         capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=timeout)
        raw=p.stdout+"\n"+p.stderr
        log.write_text(raw,encoding="utf-8")
        errors=bool(re.search(r"(?im)^.*error:",raw))
        try:reports=inventories(raw)
        except ValueError:reports=None
        return {"status":"compiled" if p.returncode==0 and not errors else "rejected",
                "returncode":p.returncode,"reports":reports,"error_excerpt":raw[-4000:] if errors or p.returncode else "",
                "source_sha256":sha(source),"log_sha256":sha(log.read_bytes()),"seconds":time.monotonic()-t}
    except subprocess.TimeoutExpired as e:
        log.write_text(str(e),encoding="utf-8")
        return {"status":"timeout_unknown","source_sha256":sha(source),"log_sha256":sha(log.read_bytes()),
                "seconds":time.monotonic()-t}
    except OSError as e:
        log.write_text(str(e),encoding="utf-8")
        return {"status":"infrastructure_unknown","source_sha256":sha(source),"log_sha256":sha(log.read_bytes()),
                "seconds":time.monotonic()-t}

def _evaluate(mathcan,source,root,actions,out,fixture=None):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    result={"schema":"LRF03/v1","source_commit":COMMIT,"source_blob":BLOB,"root":root,
            "actions":actions,"fixture":fixture,"reward":None,"kernel":None,"foundation":None,"contract":None,
            "compiler_calls":0,"deadline_ms":DEADLINE*1000,"allowlist":sorted(ALLOW)}
    def done(reason):
        result["elapsed_ms"]=1000*(time.monotonic()-start)
        if result["elapsed_ms"]>result["deadline_ms"]:
            result["reward"]=None;reason="deadline_unknown"
        result["classification"]=reason
        (out/"receipt.json").write_text(json.dumps(result,indent=2)+"\n")
        return result
    try:one,two,header,alias_line=render(source,root,actions,fixture)
    except (ValueError,TypeError):
        return done("malformed_censored")
    result["header_sha256"]=header
    def run(text,file):
        remaining=DEADLINE-(time.monotonic()-start)
        if remaining<=0:return {"status":"deadline_unknown"}
        result["compiler_calls"]+=1
        return native(mathcan,text,out/file,min(COMPILE_LIMIT,remaining))
    first=run(one,"first.lean");result["first"]=first
    if first["status"]!="compiled":
        if first["status"]=="rejected" and "unsolved goals" in first.get("error_excerpt","").lower():
            result["kernel"]=False
            return done("native_nonclosure_censored")
        return done("native_unknown_censored")
    if first.get("reports") is None or set(first["reports"])!={SUB}:
        return done("missing_axiom_inventory_censored")
    result["kernel"]=True
    ax=set(first["reports"][SUB]);result["axioms"]=sorted(ax)
    result["foundation"]=ax<=ALLOW
    second=run(two,"second.lean");result["second"]=second
    if second["status"]=="compiled":
        rs=second.get("reports")
        if rs is None or set(rs)!={SUB,CON} or set(rs[SUB])!=ax or set(rs[CON])!=ax:
            return done("independent_replay_disagreement_censored")
        result["contract"]=True
    elif second["status"]=="rejected":
        diagnostic=second.get("error_excerpt","")
        if not re.search(rf":{alias_line}:\d+:\s*error:\s*(?:Type mismatch|type mismatch)",diagnostic):
            return done("contract_unknown_censored")
        result["contract"]=False
    else:return done("contract_unknown_censored")
    if not result["foundation"]:
        result["reward"]=0;return done("excluded_axiom")
    if not result["contract"]:
        result["reward"]=0;return done("statement_contract_unestablished")
    result["reward"]=1
    return done("verified_target_proof")

def evaluate_actions(mathcan,source,root,actions,out):
    return _evaluate(mathcan,source,root,actions,out)

def pilot(mathcan,out):
    source=checkout(mathcan)
    version=subprocess.check_output(["lake","env","lean","--version"],cwd=mathcan,text=True).strip()
    if "4.35.0" not in version:raise ValueError("wrong compiler version")
    out.mkdir(parents=True,exist_ok=False)
    rows=[];failures=[]
    for root,spec in ROOTS.items():
        correct=spec["reference"]
        cases=[
            ("correct",correct,None,(True,True,True,1)),
            ("incomplete",["skip"],None,(False,None,None,None)),
            ("malformed",["exact arbitrary Lean"],None,(None,None,None,None)),
            ("altered",correct,"true_claim",(True,True,False,0)),
            ("added_hypothesis",correct,"extra_hypothesis",(True,True,False,0)),
            ("hidden_axiom",correct,"forbidden_alias",(True,False,True,0)),
            ("unused_axiom",correct,"unused_axiom",(True,True,True,1)),
        ]
        for arm,actions,fixture,expected in cases:
            result=_evaluate(mathcan,source,root,actions,out/root/arm,fixture)
            actual=tuple(result[k] for k in ("kernel","foundation","contract","reward"))
            row={"root":root,"arm":arm,"expected":expected,"actual":actual,"receipt":result}
            rows.append(row)
            if actual!=expected:failures.append({"root":root,"arm":arm,"expected":expected,"actual":actual})
            if arm=="malformed" and result["compiler_calls"]!=0:raise AssertionError("malformed request executed")
            if result["compiler_calls"]>2:raise AssertionError("compiler budget breached")
            print("CASE",root,arm,result["classification"],actual,flush=True)
    (out/"cases.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows))
    report={"version":version,"source_commit":COMMIT,"source_blob":BLOB,"roots":len(ROOTS),"cases":len(rows),
            "positive":sum(x["actual"][-1]==1 for x in rows),
            "negative":sum(x["actual"][-1]==0 for x in rows),
            "censored":sum(x["actual"][-1] is None for x in rows),
            "false_positive":sum(x["actual"][-1]==1 and x["expected"][-1]!=1 for x in rows),
            "compiler_calls":sum(x["receipt"]["compiler_calls"] for x in rows),
            "failures":failures,"passed":not failures}
    (out/"gate.json").write_text(json.dumps(report,indent=2)+"\n")
    print("LRF03_GATE",json.dumps(report),flush=True)
    if failures:raise SystemExit(1)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mathcan",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    pilot(a.mathcan.resolve(),a.out.resolve())
