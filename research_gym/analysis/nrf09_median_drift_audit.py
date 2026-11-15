"""NRF-09: a posteriori exact-decimal eight-arm median drift certificates.

Input is the twelve full eleven-arm JSON receipts in the separate
NRF-09 reproduction bundle; compact Github receipts omit random-arm data.
"""
from __future__ import annotations
import argparse
import json
from fractions import Fraction as Q
from pathlib import Path

SEEDS=(131,137,139,149,151,157)

def objective(arm):
    E=Q(str(arm["heldout_mse"]))
    L=Q(str(arm["local_fiber_leakage"]))
    if E<0 or L<0:
        raise ValueError("invalid objective components")
    return E+L*L

def median8(arms):
    scores=sorted(objective(arms[f"random_{i}"]) for i in range(8))
    return (scores[3]+scores[4])/2

def margin(row):
    arms=row["arms"]
    return objective(arms["supervised"])-median8(arms)

def audit_pair(before, after):
    if (before["seed"],before["rounds"])!=(after["seed"],after["rounds"]):
        raise ValueError("mismatched seed or recurrent depth")
    if (before["optimizer_steps"],after["optimizer_steps"])!=(16,32):
        raise ValueError("only the registered step pair is permitted")
    old,new=before["arms"],after["arms"]
    delta_supervised=abs(objective(old["supervised"])-objective(new["supervised"]))
    delta_random=max(abs(objective(old[f"random_{i}"])-objective(new[f"random_{i}"]))
                     for i in range(8))
    bound=delta_supervised+delta_random
    old_margin,new_margin=margin(before),margin(after)
    if abs(new_margin-old_margin)>bound:
        raise AssertionError("median Lipschitz bound violated")
    certified=abs(old_margin)>bound
    if certified and not (old_margin*new_margin>0):
        raise AssertionError("wrong winner after certified drift")
    return {
        "rounds":before["rounds"],"seed":before["seed"],
        "margin_before":str(old_margin),"margin_after":str(new_margin),
        "drift_bound":str(bound),"delta_supervised":str(delta_supervised),
        "max_random_delta":str(delta_random),
        "sign_certified":certified,
    }

def audit_directory(receipts_dir:Path):
    out=[]
    for rounds in (16,32):
        for seed in SEEDS:
            raw=json.loads((receipts_dir/f"nrf09_R{rounds}_S{seed}.json").read_text())
            rows=raw["rows"]
            if len(rows)!=2:
                raise ValueError("two registered snapshots required")
            out.append(audit_pair(rows[0],rows[1]))
    return {
        "study_id":"NRF-09-retrospective-median-drift-v1",
        "note":"Uses arm scores from both training checkpoints; retrospective observed-drift certificate, not a prospective generalization claim.",
        "rows":out,
        "R16_winner_sign_certificates":sum(r["sign_certified"] for r in out if r["rounds"]==16),
        "R32_winner_sign_certificates":sum(r["sign_certified"] for r in out if r["rounds"]==32),
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--receipts-dir",required=True,type=Path)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    result=audit_directory(args.receipts_dir)
    payload=json.dumps(result,indent=2,sort_keys=True)+"\\n"
    if args.output:
        args.output.write_text(payload)
    print(payload)

if __name__=="__main__":
    main()
