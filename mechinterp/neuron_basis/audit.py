"""Independent receipt audit for NBR suite engineering results.

Checks hashes, labels, calibration-only feature selection, held-out AUROCs,
same-batch sham, verified deduction outcomes and graph arithmetic. Does not
re-execute the checkpoint, SAE optimizer, gradient graph or adapter edits.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def auc(y,s):
    p=s[y];n=s[~y]
    if not len(p) or not len(n):
        return None
    return float(((p[:,None]>n[None,:]).sum()+.5*(p[:,None]==n[None,:]).sum())/(len(p)*len(n)))

def schedule(p,t,blank):
    return [next((s+1 for s in range(16) if all(p[k,i]==t[i] for k in range(s,16))),17)
            for i in blank]

def audit(roster,out):
    report=json.loads((out/"report.json").read_text())
    assert report["engineering_only"] and report["roster_sha256"]==sha(roster)
    assert report["basis_sha256"]==sha(out/"basis.npz")
    rows={r["id"]:r for r in json.loads(roster.read_text())}
    saved={}
    for split,bs in [("cal",report["calibration"])]+[(s,report["families"][s]["boards"]) for s in ("eval","rep")]:
        saved[split]=[]
        for b in bs:
            r=rows[b["id"]]; assert r["split"]==split
            p=out/b["receipt"]; assert sha(p)==b["sha256"]
            with np.load(p,allow_pickle=False) as z:
                a={k:z[k] for k in z.files}
            t=np.array(list(r["answer"]),dtype=int)
            blanks=[i for i,v in enumerate(r["question"]) if v in ".0"]
            site=blanks[len(blanks)//2]; assert site==b["site"]
            clean=a["clean"]
            assert b["eligible"]==bool(clean[5,site]!=t[site])
            assert b["recovered"]==bool(np.all(clean[6:,site]==t[site]))
            assert b["failed"]==bool(np.any(clean[-1,blanks]!=t[blanks]))
            saved[split].append((b,a))
            if split=="cal":
                continue
            preds=a["pred"]
            assert np.array_equal(preds[0],preds[1])
            base=np.array(schedule(preds[0],t,blanks))
            vs=b["causal"]["variants"]; assert len(vs)==len(preds)
            for pred,v in zip(preds,vs):
                q=np.array(schedule(pred,t,blanks))
                assert v["vdd"]==int(np.abs(q-base).sum())
                assert v["signed_schedule_improvement"]==int((base-q).sum())
                assert v["final_blank_correct"]==int((q<17).sum())
                assert v["exact"]==bool(np.all(q<17))
                if "dose" in v:
                    assert abs(v["write_l2"]-v["dose"])<1e-6
                assert abs(v["linear_error"]-np.linalg.norm(np.array(v["local_logit_delta"])-v["linear_prediction"]))<1e-6
            graph=b["graph"]
            if graph:
                j=np.array(graph["jacobian"]);fd=np.array(graph["finite_difference"])
                assert abs(graph["relative_error"]-np.linalg.norm(j-fd)/max(np.linalg.norm(j),1e-12))<1e-5
    for endpoint in ("recovery","failure"):
        label="recovered" if endpoint=="recovery" else "failed"
        selected=report[endpoint+"_selected"]
        def eligible(b):return endpoint=="failure" or b["eligible"]
        cal=[(b,a) for b,a in saved["cal"] if eligible(b)]
        yc=np.array([b[label] for b,a in cal],dtype=bool)
        for arm,s in selected.items():
            if not yc.any() or yc.all():
                assert s is None;continue
            mat=np.stack([a["score_"+arm] for b,a in cal])
            av=np.array([auc(yc,mat[:,i]) for i in range(mat.shape[1])])
            strength=np.maximum(av,1-av)
            i=s["neuron"]
            # Rationally tied AUCs may differ by an ulp in the two formulas.
            assert abs(strength[i]-strength.max())<1e-12
            assert s["sign"]==(1 if av[i]>=.5 else -1)
            assert abs(s["cal_auc"]-strength[i])<1e-12
        for split in ("eval","rep"):
            rs=[(b,a) for b,a in saved[split] if eligible(b)]
            y=np.array([b[label] for b,a in rs],dtype=bool)
            fam=report["families"][split][endpoint]
            assert fam["n"]==len(rs) and fam["positive"]==int(y.sum())
            for arm,s in selected.items():
                expected=None if s is None or not rs else auc(y,np.array([a["score_"+arm][s["neuron"]]*s["sign"] for b,a in rs]))
                got=fam["arms"][arm]
                assert (got is None and expected is None) or (got is not None and expected is not None and abs(got-expected)<1e-12)
    assert sum(map(len,saved.values()))==len(rows)
    print("NBR receipt audit PASS",len(rows),"boards; arithmetic only")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roster",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();audit(a.roster,a.out)

if __name__=="__main__":main()
