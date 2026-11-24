"""LRF-04: a trusted in-process finite-policy and native proof gate.

The verifier owns mode, finite state inventory, source checkout, artifact path
and required theorem. No external receipt or untrusted Lean source is accepted.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
from itertools import product

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"lrf03"))
import adapter as A

ROOT="demo_monitor_correct"
STATES=tuple(product((False,True),repeat=2))
MODES=("compatible","collision")

def policy(mode,obs,action):
    if mode not in MODES or type(obs) is not int or type(action) is not bool:
        return False,"malformed_request"
    fiber=tuple(s for s in STATES if int(s[0])==obs)
    if not fiber:return False,"unknown_observation"
    permissions=tuple(s[0] and action and (mode=="compatible" or s[1]) for s in fiber)
    if all(permissions):return True,"unanimous_permission"
    if any(permissions):return False,"authorization_collision"
    return False,"policy_denied"

def digest_file(path):
    return A.sha(path.read_bytes())

def _trusted_receipt_valid(receipt,source,actions,out):
    """Validate freshly generated trusted process evidence, not remote input."""
    try:
        if not isinstance(receipt,dict):return False
        _,header=A.prefix_header(source,ROOT)
        if (receipt["schema"]!="LRF03/v1" or receipt["source_commit"]!=A.COMMIT
            or receipt["source_blob"]!=A.BLOB or receipt["root"]!=ROOT
            or receipt["actions"]!=actions or receipt["fixture"] is not None
            or receipt["classification"]!="verified_target_proof"
            or receipt["header_sha256"]!=A.sha(header)
            or type(receipt["reward"]) is not int or receipt["reward"]!=1
            or any(receipt[k] is not True for k in ("kernel","foundation","contract"))
            or type(receipt["compiler_calls"]) is not int or receipt["compiler_calls"]!=2
            or receipt["deadline_ms"]!=A.DEADLINE*1000
            or type(receipt["elapsed_ms"]) not in (int,float)
            or not math.isfinite(receipt["elapsed_ms"])
            or not 0<=receipt["elapsed_ms"]<=receipt["deadline_ms"]
            or receipt["allowlist"]!=sorted(A.ALLOW)):
            return False
        ax=receipt["axioms"]
        if type(ax) is not list or ax!=sorted(set(ax)) or not set(ax)<=A.ALLOW:return False
        one,two,_,_=A.render(source,ROOT,actions)
        for key,stem,expected,names in (
            ("first","first",one,{A.SUB:ax}),
            ("second","second",two,{A.SUB:ax,A.CON:ax}),
        ):
            record=receipt[key]
            if (record["status"]!="compiled" or record["returncode"]!=0 or record["reports"]!=names
                or record["source_sha256"]!=A.sha(expected)
                or digest_file(out/(stem+".lean"))!=record["source_sha256"]
                or digest_file(out/(stem+".log"))!=record["log_sha256"]):
                return False
            if A.inventories((out/(stem+".log")).read_text(encoding="utf-8"))!=names:return False
        return json.loads((out/"receipt.json").read_text())==receipt
    except (KeyError,ValueError,TypeError,OSError,AttributeError):
        return False

class Controller:
    """Caller submits only a request; construction is verifier-owned."""
    def __init__(self,mathcan:Path,mode:str):
        if mode not in MODES:raise ValueError("unknown policy mode")
        self.mathcan=mathcan.resolve()
        self.mode=mode
        self.source=A.checkout(self.mathcan)

    def review(self,request,out:Path):
        result={"schema":"LRF04/v1","allowed":False,"reason":"malformed_request",
                "policy":self.mode,"compiler_calls":0}
        if (type(request) is not dict or set(request)!={"observation","action","root","actions"}
            or type(request["observation"]) is not int or type(request["action"]) is not bool
            or type(request["root"]) is not str):
            return result
        allowed,reason=policy(self.mode,request["observation"],request["action"])
        result["reason"]=reason
        if not allowed:return result
        if request["root"]!=ROOT:
            result["reason"]="wrong_required_theorem";return result
        actions=request["actions"]
        try:A.decode_actions(ROOT,actions)
        except (ValueError,TypeError):
            result["reason"]="malformed_actions";return result
        try:
            # Recheck the trusted source at use time; the output must be new.
            if A.checkout(self.mathcan)!=self.source:
                result["reason"]="source_changed";return result
            receipt=A.evaluate_actions(self.mathcan,self.source,ROOT,list(actions),out)
        except Exception:
            result["reason"]="verifier_error";return result
        result["compiler_calls"]=receipt.get("compiler_calls") if isinstance(receipt,dict) else None
        result["native_receipt"]=receipt
        if not _trusted_receipt_valid(receipt,self.source,actions,out):
            result["reason"]="certificate_denied";return result
        result.update(allowed=True,reason="verified_permission")
        return result
