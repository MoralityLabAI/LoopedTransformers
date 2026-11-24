"""Apply exactly the phase-1 tangent planner to native paired MLP writes."""

import json

from experiments.nan_radial76.native import WRITER
from experiments.nan_recipe85.run import ROUTE, load_controllers, phase_edits
from experiments.nan_refit49.run import load_fits
from experiments.nan_route50.run import fit_packet
from experiments.nan_routematch86.geometry import plan
from experiments.nan_routematch86.native import observed_trial
from experiments.nan_vocab82.heads import Reducer, partition_sha
from experiments.nan_vocab83.batches import record_baseline, record_roots
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_writer77.pipeline import matched_duplicate, source_index

from .source import EXPERIMENT, REPO, config


def prepare_fit(np, masks):
    backend="nf4"
    base=json.loads((REPO/"experiments/nan_readpath37/registration.json").read_bytes())
    banks=load_fits(base,np)
    controller,_=load_controllers(backend,base,masks,np)
    _,t2,_,eps,_,_=fit_packet(backend,base,np)
    initial,t1=banks[f"{backend}/T1"]
    peer,t2_factors=banks[f"{backend}/T2"]
    if not all(np.array_equal(a,b) for a,b in zip(initial,peer,strict=True)):
        raise ValueError("The IO18 T1 and T2 fits do not share their initializer")
    raw=json.loads(ROUTE.read_bytes())["per_backend"][backend]["passes"]["1"]["static_prefit"]["selected_anchor"]
    anchor=raw.removeprefix("binary:")
    if not raw.startswith("binary:") or len(anchor)!=base["rank"] or set(anchor)-{"0","1"}:
        raise ValueError("The phase-1 ROUTE50 anchor changed")
    anchor=np.asarray([int(x) for x in anchor])
    phenotype={"mean_comp_acts":t2["mean_comp_acts"],"mean_write_rms":t2["mean_write_rms"],"eps":eps}

    def routes(x,y):
        choices,mass,_=phase_edits(x,y,0,{0:initial,1:t1,2:t2_factors},controller,phenotype,anchor,masks,np)
        if mass>1e-12:
            raise ValueError("Learned and static gate masses disagree")
        return choices["T2_T2_sphere"],choices["T2_static_sphere"]

    return routes


def choose(capture, anchor, prompts, ids, routes, labels, reduce, classes, check, np):
    inputs, outputs = (np.stack([anchor[name][1,i].float().numpy() for i in (0,1)]).astype(np.float64)
                       for name in ("writer_input","writer_raw"))
    learned, static=routes(inputs,outputs)
    planned=[]
    for i,key in enumerate(ids):
        check()
        writes, proof=plan(outputs[i],learned[i],static[i],key+"|phase1",np)
        geometry=proof.get("anchored_tangent",{"status":proof["status"]})
        planned.append((writes.get("anchored_tangent",{}),geometry))
    receipts=[r for _,r in planned]
    if any(r["status"]!="matched" or r["same_learned_and_static_bytes"] for r in receipts):
        return {"status":"geometry_unqualified","plans":receipts}
    duplicates=[]
    for i,prompt in enumerate(prompts):
        check()
        clean=capture.capture([prompt,prompt])
        if not matched_duplicate(anchor,clean,i):
            return {"status":"mixed_duplicate_unqualified","plans":receipts}
        duplicates.append(clean)
    worlds=[observed_trial(capture,prompt,clean,writes,labels,reduce,classes,check)
            for prompt,clean,(writes,_) in zip(prompts,duplicates,planned,strict=True)]
    return {"status":"closed","worlds":worlds,"plans":receipts}


def run_pairs(work,capture,payload,classes,routes,np,check,host_gate):
    reduce=Reducer(classes)
    focus, baseline_digest, batches=record_baseline(work,capture,payload,config(),check,reduce,classes)
    host_gate()
    received=json.loads((work/"qualification.json").read_bytes())
    gate=received["gate"]
    if (received["experiment"]!=EXPERIMENT or received["baseline_sha256"]!=sha(work/"baseline.json")
            or received["partition_sha256"]!=partition_sha(classes) or received["vocab"]!=len(classes)
            or gate["baseline_digest"]!=baseline_digest or gate["registered_roots"]!=len(focus)
            or type(gate["passed"]) is not bool or not set(gate["qualified_focused_roots"])<=set(focus)):
        raise ValueError("Host qualification did not bind this B2 baseline")
    lookup,_,labels=source_index(payload)

    def trial(root, entry):
        anchor,prompts=entry
        ids=[lookup[root,"focused",w]["id"] for w in ("A","B")]
        return choose(capture,anchor,prompts,ids,routes,labels,reduce,classes,check,np)

    return batches,record_roots(work,payload,focus,gate,trial)
