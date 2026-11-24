"""Bound the immediate full head after NF4 MLP writer tangent interventions."""

import argparse
from collections import Counter,defaultdict
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
import sys

from experiments.nan_inverse65.lattice import add,lattice
from experiments.nan_inverse65.run import open_archive
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead,SHARD,bound,digest,joined,write_new
from experiments.nan_routematch86.geometry import ARMS,plan
from experiments.nan_route50.run import state_packet
from experiments.nan_tangent87.worker import prepare_fit
from experiments.nan_vocab81.run import LABELS,native_controls,token_classes

from .bounds import calculate,fixture,residual_box


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
BASE=REPO/"experiments/nan_readpath37/registration.json"
MANIFEST=HERE/"SOURCE_MANIFEST.json"
PREVIOUS=REPO/"runs/nan_routematch86/attempt04/nf4_private.json"
BUDGET={"minimum_start_available_RAM_bytes":5*2**30,"minimum_available_RAM_bytes":4*2**30,
        "maximum_private_bytes":768*2**20,"maximum_seconds":600}
LIMIT=.125
STAGES=[(p,l) for p in (1,2) for l in (5,11,17,21)]


def seal():
    paths=[HERE/n for n in ("__init__.py",".gitattributes","PROTOCOL.md","bounds.py","run.py")]
    paths+=[BASE,*[REPO/n for n in ("experiments/nan_inverse65/run.py","experiments/nan_inverse65/lattice.py",
        "experiments/nan_inverse65/SOURCE_MANIFEST.json","experiments/nan_inverse65/CPU_VERIFICATION.json",
        "experiments/nan_route50/run.py","experiments/nan_tangent87/worker.py",
        "experiments/nan_tangent87/RUNTIME_MANIFEST.json","experiments/nan_routematch86/geometry.py",
        "experiments/nan_routematch86/SOURCE_MANIFEST.json","reports/nanbeige_bpd_gym/nan_routematch86_results.json",
        "experiments/nan_vocab81/run.py","experiments/nan_readpath37/run.py",
        "experiments/nan_phase35/run.py","experiments/nan_ram28/resources.py","experiments/nan_ranksite13/resources.py")]]
    return {"experiment":"NAN-CARRYBOUND-88","resources":BUDGET,"current_logit_limit":LIMIT,
            "archived_NF4_geometry_sha256":digest(PREVIOUS),
            "files_sha256":{p.relative_to(REPO).as_posix():digest(p) for p in paths}}


def panel(base, selected, head, routes, geometry_sha, check, np):
    rows,x,y=state_packet("nf4",base,selected,check,np)
    ix=[i for i,r in enumerate(rows) if r["arm"]=="focused"]
    x,y=(values[ix,0] for values in (x,y))
    rows=[rows[i] for i in ix]
    if len(rows)!=48:
        raise ValueError("Expected all focused CAL and DEV worlds")
    focus={i:r for i,r in selected.items() if r["arm"]=="focused"}
    if list(focus.values())[0]["id"]!=rows[0]["id"]:
        raise ValueError("Native archive row order differs")
    states,replay_error=open_archive("nf4",base,focus,head,STAGES,check,np)
    saved={r["id"]:r["anchored_tangent"] for r in json.loads(bound(PREVIOUS,geometry_sha))}
    grid=lattice(np)
    primary=STAGES.index((1,21))
    learned,static=routes(x.astype(np.float64),y.astype(np.float64))
    clean, low, high, owners, labels, other, details=[],[],[],[],[],[],[]
    for i,row in enumerate(rows):
        check()
        key=row["split"],row["arm"],row["root"]
        world=row["world"]
        current=states[key][world]
        if list(focus.values())[i]["id"]!=row["id"]:
            raise ValueError("Out-of-order focused archive row")
        clean.append(current[2][primary,0])
        h,a,w=current[2][primary]
        if not np.array_equal(y[i],w):
            raise ValueError("Independent archives differ on the observed writer")
        foil=states[key]["B" if world=="A" else "A"][0]
        labels.append(LABELS.index(current[0]))
        other.append(LABELS.index(foil))
        plans,report=plan(w.astype(np.float64),learned[i],static[i],row["id"]+"|phase1",np)
        if report["anchored_tangent"]!=saved[row["id"]] or set(plans["anchored_tangent"])!=set(ARMS):
            raise ValueError("Rebuilt edits differ from the independent ROUTEMATCH86 geometry")
        clean_lo,clean_hi,c_lo,c_hi=residual_box(grid,h,a,w,w,np)
        if not np.array_equal(clean_lo,h) or not np.array_equal(clean_hi,h):
            raise ValueError("Compatible carry endpoints do not reproduce the archived state")
        for arm in ARMS:
            replacement=plans["anchored_tangent"][arm].astype(np.float32)
            left,right=add(c_lo,replacement,np),add(c_hi,replacement,np)
            if not np.isfinite([left,right]).all() or (left>right).any():
                raise ValueError("Edited residual intervals are incomplete")
            owners.append(i)
            low.append(left)
            high.append(right)
            details.append(row|{"arm":arm,"unique_carry_coordinates":int((c_lo==c_hi).sum()),
                                 "unique_edited_residual_coordinates":int((left==right).sum()),
                                 "residual_box_l2_width":float(np.linalg.norm(right.astype(np.float64)-left))})
    return focus,np.stack(clean),(np.stack(low),np.stack(high)),np.asarray(owners),np.asarray(labels),np.asarray(other),details,replay_error


def group(details, scores):
    rows=[]
    for i,row in enumerate(details):
        drift=tuple(map(float,scores["peak_drift"][i]))
        shift=tuple(map(float,scores["paired_margin_change"][i]))
        gold=tuple(map(float,scores["gold_margin"][i]))
        w,c=(tuple(map(float,scores[k][i])) for k in ("winner_value","competitor"))
        same=w[0]>c[1]
        moved=w[1]<c[0]
        pass_gate=same and drift[1]<=LIMIT and max(map(abs,shift))<=LIMIT
        fail_gate=moved or drift[0]>LIMIT or shift[0]>LIMIT or shift[1]<-LIMIT
        if pass_gate and fail_gate:
            raise ValueError("Current-head pass and fail certificates intersect")
        rows.append(row|{"winner":"preserved" if same else "changed" if moved else "unresolved",
                         "current_gate":"preserved" if pass_gate else "failed" if fail_gate else "unresolved",
                         "peak_drift_interval":drift,"paired_margin_change_interval":shift,"gold_margin_interval":gold,
                         "archive_winner_logit_interval":w,"strongest_other_logit_interval":c,
                         "normalized_unique_coordinates":int(scores["normalized_unique"][i])})
    cells={}
    for split in ("calibration","development"):
        use=[r for r in rows if r["split"]==split]
        cells[split]={}
        roots=defaultdict(list)
        for arm in ARMS:
            selected=[r for r in use if r["arm"]==arm]
            count=Counter(r["winner"] for r in selected)
            gate=Counter(r["current_gate"] for r in selected)
            cells[split][arm]={"worlds":len(selected),"winner_status":dict(count),"current_head_gate":dict(gate),
               "median_unique_edited_residual_coordinates":median(r["unique_edited_residual_coordinates"] for r in selected),
               "median_peak_drift_lower":median(r["peak_drift_interval"][0] for r in selected),
               "median_peak_drift_upper":median(r["peak_drift_interval"][1] for r in selected)}
            for row in selected:
                roots[row["root"]].append(row)
        if any(len(r)!=2*len(ARMS) for r in roots.values()):
            raise ValueError("A root or arm failed to close")
        cells[split]["both_world_roots"]={"registered":len(roots),
            "all_arms_guaranteed_winner_preserved":sum(all(r["winner"]=="preserved" for r in v) for v in roots.values()),
            "all_arms_guaranteed_gate_preserved":sum(all(r["current_gate"]=="preserved" for r in v) for v in roots.values()),
            "some_arm_guaranteed_gate_failed":sum(any(r["current_gate"]=="failed" for r in v) for v in roots.values())}
    return cells,rows


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze",action="store_true")
    cli.add_argument("--snapshot",type=Path)
    cli.add_argument("--work",type=Path)
    args=cli.parse_args()
    sealed=seal()
    if args.freeze:
        write_new(MANIFEST,sealed)
        return
    if not __debug__ or not args.snapshot or not args.work or json.loads(MANIFEST.read_bytes())!=sealed:
        cli.error("Require the frozen source, checkpoint and a fresh output directory")
    base=json.loads(BASE.read_bytes())
    if args.snapshot.name!=base["revision"]:
        cli.error("Checkpoint revision differs")
    with lease(BUDGET) as (check,samples):
        if (version("numpy"),version("tokenizers"))!=("2.5.5","0.20.3"):
            raise ValueError("Unpinned numerical environment")
        import numpy as np
        from tokenizers import Tokenizer
        from experiments.nan_fit07.pointwise import masks

        test=fixture(np)
        routes=prepare_fit(np,masks)
        tok_file=args.snapshot/"tokenizer.json"
        bound(tok_file,base["snapshot_pins"]["tokenizer.json"])
        tok=Tokenizer.from_file(str(tok_file))
        ids=[tok.encode(c,add_special_tokens=False).ids for c in LABELS]
        if any(len(c)!=1 for c in ids):
            raise ValueError("Canonical answer did not encode as one token")
        canonical=[v[0] for v in ids]
        selected,_=joined(base)
        args.work.mkdir(parents=True,exist_ok=False)
        with (args.snapshot/SHARD).open("rb") as stream:
            head=FixedHead(stream,args.snapshot,base,np,check)
            classes=token_classes(tok,head.vocab,np,check)
            if classes[canonical].tolist()!=list(range(len(LABELS))):
                raise ValueError("Canonical label class differs")
            focus,clean,box,owners,targets,foil,details,error=panel(base,selected,head,routes,
                                                    sealed["archived_NF4_geometry_sha256"],check,np)
            controls=native_controls("nf4",base,focus,tok,check,np)[:,0]
            archive_top=controls[:,0].astype(np.int32)
            scores=calculate(head,clean,box,owners,targets,foil,archive_top,classes,canonical,check,np)
            if (not np.array_equal(scores["winner"],archive_top) or not np.array_equal(scores["peak"],controls[:,1])
                    or np.max(abs(scores["logz"]-controls[:,2]))>base["native_logit_abs_tol"]):
                raise ValueError("Clean full vocabulary differs from the native phase-1 archive")
        cells,private=group(details,scores)
        check()
    assert sealed==seal() and "torch" not in sys.modules
    write_new(args.work/"private.json",private)
    write_new(args.work/"aggregate.json",{"experiment":sealed["experiment"],"status":"passed",
               "source_manifest_sha256":digest(MANIFEST),"evidence":"archived NF4 CPU enclosure",
               "complete_vocab_rows":head.vocab,"clean_native_heads_matched":len(clean),"bounded_edited_heads":len(private),
               "max_native_selected_logit_error":error,"fixture":test,"splits":cells,
               "pretrained_decoder_forwards":0,"formal_comparisons":0,"sealed_test_accessed":False,
               "resources":{"max_sampled_private_bytes":max(s["private_bytes"] for s in samples),
                  "max_sampled_working_set_bytes":max(s["working_set_bytes"] for s in samples),
                  "min_sampled_available_RAM_bytes":min(s["available_physical_bytes"] for s in samples)}})
    print(cells,flush=True)


if __name__=="__main__":
    main()
