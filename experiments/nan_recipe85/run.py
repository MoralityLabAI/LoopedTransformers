"""Replay crossed IO18 factors, controllers and write rules on common states."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from statistics import fmean
import sys

from experiments.nan_delay39.instrument import prepare
from experiments.nan_gauge42.run import read_archive
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bf16, bound, digest, joined, write_new
from experiments.nan_refit49.run import FIT_ARCHIVE, FIT_REPORT, load_fits
from experiments.nan_route50.masks import mass_matched
from experiments.nan_route50.run import fit_packet, npz_fields, state_packet
from experiments.nan_sieve84.run import paired_tape, readout
from experiments.nan_vocab81.run import LABELS, native_controls, token_classes


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO / "experiments/nan_readpath37/registration.json"
DOSE = REPO / "experiments/nan_delay39/registration.json"
ROUTE = REPO / "reports/nanbeige_bpd_gym/nan_route50_results.json"
ALIGNMENT = REPO / "reports/nanbeige_bpd_gym/nan_refit49_results.json"
MANIFEST = HERE / "SOURCE_MANIFEST.json"
BUDGET = {"minimum_start_available_RAM_bytes":5*2**30,"minimum_available_RAM_bytes":4*2**30,
          "maximum_private_bytes":768*2**20,"maximum_seconds":600}
CORE = ("T1_T1_zero","T1_T2_zero","T2_T1_zero","T2_T2_zero")


def seal():
    paths = [HERE / n for n in (".gitattributes","__init__.py","PROTOCOL.md","run.py")]
    paths += [BASE, DOSE, ROUTE, ALIGNMENT, FIT_REPORT, *[REPO / n for n in (
        "experiments/nan_fit16_io18/SOURCE_MANIFEST.json","experiments/nan_refit49/run.py",
        "experiments/nan_route50/run.py","experiments/nan_route50/masks.py",
        "experiments/nan_fit07/pointwise.py","mechinterp/nanbeige_bpd_gym/native_components.py",
        "experiments/nan_gauge42/run.py","experiments/nan_sieve84/run.py","experiments/nan_sieve84/SOURCE_MANIFEST.json",
        "experiments/nan_vocab81/run.py","experiments/nan_readpath37/run.py",
        "experiments/nan_delay39/instrument.py","experiments/nan_dose38/run.py",
        "experiments/nan_phase35/run.py","experiments/nan_ram28/resources.py","experiments/nan_ranksite13/resources.py",
    )]]
    return {"experiment":"NAN-RECIPE-85","resources":BUDGET,
            "files_sha256":{p.relative_to(REPO).as_posix():digest(p) for p in paths}}


def load_controllers(backend, base, masks, np):
    published = json.loads(FIT_REPORT.read_bytes())
    audit = json.loads(bound(FIT_ARCHIVE/"audits"/f"{backend}.json",
                             published["backends"][backend]["audit_sha256"]))
    names = ("ci__proj_in.weight","ci__proj_in.bias","ci__out_proj.weight","ci__out_proj.bias")
    fitted, errors = {}, {}
    for item in audit["cells"]:
        depth = item["passes"]
        parent = FIT_ARCHIVE/backend/f"t{depth}_layer21_down_proj_rank8"
        receipt = json.loads(bound(parent/"receipt.json",item["receipt_sha256"]))
        final = receipt["final_checkpoint"]
        bank = npz_fields(parent/final["file"],final["sha256"],names,np)
        native = npz_fields(parent/"CI_replay_local.npz",receipt["CI_replay_sha256"],("X","masks"),np)
        ci = final["CI_config"]
        error = float(np.max(abs(masks(native["X"],bank,ci)-native["masks"])))
        rule = receipt["training_contract"]["ablation"]
        if ((rule["mode"],rule["sphere_rescale"]) != (
                ("zero",False) if depth==1 else ("pass_mean",True))
                or error > 2e-5 or ci["max_passes"] != depth
                or receipt["target_parameter_sha256_unchanged"] != base["backends"][backend]["target"]):
            raise ValueError("Learned controller or ablation recipe differs from archive")
        fitted[depth] = bank,ci
        errors[f"T{depth}"] = error
    if set(fitted) != {1,2}:
        raise ValueError("Missing one fitted IO18 controller")
    return fitted,errors


def crossed(values):
    """Bank and controller terms close to diagonal change for vectors or scores."""
    aa, ab, ba, bb = (values[k] for k in CORE)
    return {"bank":((ba-aa)+(bb-ab))/2, "controller":((ab-aa)+(bb-ba))/2,
            "interaction":(bb-ba)-(ab-aa),"diagonal":bb-aa}


def phase_edits(x, y, time, factors, controller, phenotype, anchor, masks, np):
    raw = [masks(x,*controller[d]) for d in (1,2)]
    mass = (raw[0].sum(axis=-1)+raw[1].sum(axis=-1))/2
    gate1,gate2,static = [mass_matched(m,mass,np) for m in (*raw,anchor)]
    gate = {1:gate1,2:gate2}
    reads = {d:x @ bank[0] for d,bank in factors.items()}

    def zero(d, m):
        return ((m-1)*reads[d]) @ factors[d][1]

    def mean_backup(m):
        return ((1-m)*(phenotype["mean_comp_acts"][time]-reads[2])) @ factors[2][1]

    def sphere(delta, m):
        candidate = y+delta
        square = (candidate*candidate).mean(axis=1,keepdims=True)
        eps = phenotype["eps"]
        rms = np.sqrt(square+eps)
        frac = (1-m).mean(axis=1,keepdims=True)
        target = max(float(phenotype["mean_write_rms"][time]),eps)
        delta_radial = candidate*np.where(square>eps,frac*(target/rms-1),0.)
        return delta+delta_radial

    z = {f"T{d}_T{c}_zero":zero(d,gate[c]) for d in (1,2) for c in (1,2)}
    mean = mean_backup(gate2)
    static_mean = mean_backup(static)
    vectors = z | {"T2_T2_mean":mean,"T2_T2_sphere":sphere(mean,gate2),
                   "T2_static_zero":zero(2,static),"T2_static_sphere":sphere(static_mean,static),
                   "SVD_static_zero":zero(0,static)}
    maximum = max(float(np.max(abs(m.sum(axis=-1)-mass))) for m in (gate1,gate2,static))
    return vectors,maximum,raw


def ratios(vectors, raw, ix, np):
    bank = crossed(vectors)
    np.testing.assert_allclose(bank["bank"]+bank["controller"],bank["diagonal"],rtol=0,atol=1e-10)
    z2, mean, sphere = (vectors[k] for k in ("T2_T2_zero","T2_T2_mean","T2_T2_sphere"))
    metrics = bank | {"pass_mean_increment":mean-z2, "sphere_increment":sphere-mean,
                      "whole_recipe_increment":sphere-z2,
                      "learned_vs_static_zero":z2-vectors["T2_static_zero"],
                      "learned_vs_static_sphere":sphere-vectors["T2_static_sphere"]}
    return {"median_physical_l2":{name:float(np.median(np.linalg.norm(val[ix],axis=1)))
                                  for name,val in metrics.items()},
            "median_T2_zero_l2":float(np.median(np.linalg.norm(z2[ix],axis=1))),
            "mean_raw_T1_mass":float(np.mean(raw[0][ix].sum(axis=1))),
            "mean_raw_T2_mass":float(np.mean(raw[1][ix].sum(axis=1)))}


def qualified_edits(rows, states, y, directions, dose, np):
    bins = {}
    for i,r in enumerate(rows):
        if r["split"] == "development":
            bins.setdefault(r["root"],[]).append(i)
    records, eligibility = [], {}
    for root, twins in bins.items():
        if len(twins) != 2:
            raise ValueError("Expected an A/B test pair")
        patches, diagnostics = [], []
        for i in twins:
            arms = {}
            for name, direction in directions.items():
                # The modeled write changes first land on the BF16 write grid.
                delivered = bf16(y[i]+direction[i],np).astype(np.float64)-y[i]
                arms[name] = [prepare(states[i],s*delivered,dose["phase2_readout_dose_fraction"],
                                      dose["dose_relative_tolerance"],dose["minimum_direction_cosine"],np)
                              for s in (1.,-1.)]
            all_patches = [v for pair in arms.values() for v in pair]
            targeted = all_patches[0].requested_l2
            if targeted<=0 or any(p.requested_l2 != targeted for p in all_patches):
                raise ValueError("Matched residual dose target changed")
            spread = (max(p.realized_l2 for p in all_patches)-min(p.realized_l2 for p in all_patches))/targeted
            ok = all(p.eligible for p in all_patches) and spread<=dose["max_contrast_dose_gap_fraction"]
            diagnostics.append({"index":i,"qualified":ok,"spread":spread,
                "arms":{name:[{"requested_l2":p.requested_l2,"realized_l2":p.realized_l2,
                               "cosine":p.cosine,"eligible":p.eligible} for p in group]
                        for name,group in arms.items()}})
            patches.extend((root,i,name,sign,p.after) for name,group in arms.items()
                           for sign,p in zip((1,-1),group,strict=True))
        eligibility[root] = diagnostics
        if all(v["qualified"] for v in diagnostics):
            records.extend(patches)
    return records,eligibility


def endpoints(data, eligibility, clean, methods):
    registered = len(eligibility)
    roots = {r:[row["index"] for row in group] for r,group in eligibility.items()
             if all(row["qualified"] for row in group)}
    results = {}
    for a in methods:
        counts = {"repairs":0,"regressions":0,"direction_sign_disagreements":0}
        shifts = []
        for root,idx in roots.items():
            per_pair = []
            for i in idx:
                plus,minus = (data[root,i,a,s] for s in (1,-1))
                full = (plus["full_margin"]-minus["full_margin"])/2
                paired = (plus["paired_margin"]-minus["paired_margin"])/2
                per_pair.append((full,paired,plus["full_margin"]-clean[i]["full_margin"]))
                counts["repairs"] += plus["correct"] and not clean[i]["correct"]
                counts["regressions"] += clean[i]["correct"] and not plus["correct"]
                counts["direction_sign_disagreements"] += full*paired<0
            shifts.append(tuple(fmean(x) for x in zip(*per_pair,strict=True)))
        scores = [fmean(x) for x in zip(*shifts,strict=True)] if shifts else [None]*3
        results[a] = counts | dict(zip(("full_half_difference","paired_half_difference","active_minus_clean"),
                                      scores,strict=True))
    contrast = None
    if roots:
        v = {a:results[a]["full_half_difference"] for a in methods}
        contrast = crossed(v) | {"pass_mean":v["T2_T2_mean"]-v["T2_T2_zero"],
            "sphere":v["T2_T2_sphere"]-v["T2_T2_mean"],
            "learned_minus_static_sphere":v["T2_T2_sphere"]-v["T2_static_sphere"]}
    return {"closed_roots":len(roots),"unknown_worlds":2*(registered-len(roots)),
            "readout_by_arm":results,"factorial_and_policy_contrasts":contrast}


def evaluate(backend, base, dose, all_selected, head, tok, classes, canonical, banks,
             check, masks, work, np):
    reference = json.loads(ALIGNMENT.read_bytes())["comparisons"][f"{backend}/T1_{backend}/T2"]["geometry"]
    if any(reference[field]!=list(range(base["rank"])) for field in ("final_assignment","initial_assignment")):
        raise ValueError("Common SVD component slots do not match across fits")
    initial,T1 = banks[f"{backend}/T1"]
    shared,T2 = banks[f"{backend}/T2"]
    if not all(np.array_equal(a,b) for a,b in zip(initial,shared,strict=True)):
        raise ValueError("The source fits did not share the exact SVD initializer")
    controller,errors = load_controllers(backend,base,masks,np)
    _,t2,_,eps,_,_ = fit_packet(backend,base,np)
    phenotype = {"mean_comp_acts":t2["mean_comp_acts"],"mean_write_rms":t2["mean_write_rms"],"eps":eps}
    anchors = json.loads(ROUTE.read_bytes())["per_backend"][backend]["passes"]
    template = []
    for phase in ("1","2"):
        name = anchors[phase]["static_prefit"]["selected_anchor"]
        if not name.startswith("binary:") or set(name[7:])-{"0","1"} or len(name[7:])!=base["rank"]:
            raise ValueError("The archived CAL static anchor is not the frozen binary roster")
        template.append(np.array([int(x) for x in name[7:]]))
    rows,x,y = state_packet(backend,base,all_selected,check,np)
    focused = [i for i,row in enumerate(rows) if row["arm"]=="focused"]
    rows = [rows[i] for i in focused]
    x,y = (v[focused].astype(np.float64) for v in (x,y))
    dev = [i for i,row in enumerate(rows) if row["split"]=="development"]
    if (len(rows),len(dev))!=(48,32):
        raise ValueError("Wrong focused CAL/DEV source coverage")
    selected = {i:r for i,r in all_selected.items() if r["arm"]=="focused"}
    twins,_,_,closure = read_archive(backend,base,selected,head,check,np)
    _, states, gold, foil = paired_tape(selected,twins,"residual_after_layer")
    for i,r in enumerate(rows):
        world = twins[r["split"],r["arm"],r["root"]][r["world"]]
        for p in (1,2):
            if not np.array_equal(world[1][p,"down_proj_input"],x[i,p-1]):
                raise ValueError("Native write and residual tapes do not attach to the same input")
    phases, raw_private = {}, {}
    for p in (0,1):
        vectors,error,raw = phase_edits(x[:,p],y[:,p],p,{0:initial,1:T1,2:T2},
                                        controller,phenotype,template[p],masks,np)
        phases[str(p+1)] = ratios(vectors,raw,dev,np) | {"common_mass_max_error":error}
        metrics = crossed(vectors)
        z2,mean,sphere = (vectors[k] for k in ("T2_T2_zero","T2_T2_mean","T2_T2_sphere"))
        per_world = metrics | {"pass_mean_increment":mean-z2, "sphere_increment":sphere-mean,
                               "whole_recipe_increment":sphere-z2,
                               "learned_vs_static_zero":z2-vectors["T2_static_zero"],
                               "learned_vs_static_sphere":sphere-vectors["T2_static_sphere"]}
        raw_private[str(p+1)] = {key:np.linalg.norm(v,axis=1).tolist() for key,v in per_world.items()}
    clean = readout(head,states,classes,canonical,gold,foil,check,np)
    saved = native_controls(backend,base,selected,tok,check,np)
    for old,new in zip(saved,clean,strict=True):
        token,peak,logz,correct = old[1]
        if (new["top"]!=token or new["peak"]!=peak or new["correct"]!=bool(correct)
                or abs(new["logz"]-logz)>base["native_logit_abs_tol"]):
            raise ValueError("Native pass-2 full head did not close")
    pending,eligibility = qualified_edits(rows,states,y[:,1],vectors,dose,np)
    edits = {}
    if pending:
        scored = readout(head,[state for *_,state in pending],classes,canonical,
                         [gold[i] for _,i,_,_,_ in pending],[foil[i] for _,i,_,_,_ in pending],check,np)
        edits = {(root,i,a,s):d for (root,i,a,s,_),d in zip(pending,scored,strict=True)}
    summary = endpoints(edits,eligibility,clean,vectors)
    write_new(work/f"{backend}_private.json",{"rows":rows,"qualification":eligibility,"raw_physical_norms":raw_private,
              "clean":clean,"edits":[{"root":root,"i":i,"arm":arm,"sign":sign,**datum}
                                     for (root,i,arm,sign),datum in edits.items()]})
    return {"native_controller_max_error":errors,"native_selected_max_logit_error":closure,
            "clean_heads_exact":len(clean),"signed_edit_heads":len(edits),"phase_write_geometry":phases,
            "dose_matched_pass2":summary}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze",action="store_true")
    cli.add_argument("--snapshot",type=Path)
    cli.add_argument("--work",type=Path)
    args=cli.parse_args()
    registered=seal()
    if args.freeze:
        write_new(MANIFEST,registered)
        return
    if not __debug__ or not args.snapshot or not args.work or json.loads(MANIFEST.read_bytes())!=registered:
        cli.error("Require assertions, the frozen source, the snapshot and a new output directory")
    base,dose=(json.loads(p.read_bytes()) for p in (BASE,DOSE))
    if args.snapshot.name!=base["revision"]:
        cli.error("Wrong snapshot revision")
    with lease(BUDGET) as (check,samples):
        if (version("numpy"),version("tokenizers"))!=("2.5.5","0.20.3"):
            raise ValueError("Pinned NumPy or tokenizer version changed")
        import numpy as np
        from tokenizers import Tokenizer
        from experiments.nan_fit07.pointwise import masks

        banks=load_fits(base,np)
        roster,_=joined(base)
        args.work.mkdir(parents=True,exist_ok=False)
        file=args.snapshot/"tokenizer.json"
        bound(file,base["snapshot_pins"]["tokenizer.json"])
        tok=Tokenizer.from_file(str(file))
        ids=[tok.encode(c,add_special_tokens=False).ids for c in LABELS]
        if any(len(v)!=1 for v in ids):
            raise ValueError("Bad label tokenization")
        canonical=[v[0] for v in ids]
        results={}
        with (args.snapshot/SHARD).open("rb") as stream:
            head=FixedHead(stream,args.snapshot,base,np,check)
            classes=token_classes(tok,head.vocab,np,check)
            if classes[canonical].tolist()!=list(range(len(LABELS))):
                raise ValueError("Bare label class changed")
            for backend in base["backends"]:
                r=evaluate(backend,base,dose,roster,head,tok,classes,canonical,banks,check,masks,args.work,np)
                results[backend]=r
                print(backend,r["dose_matched_pass2"]["closed_roots"],r["phase_write_geometry"]["2"],flush=True)
    assert registered==seal() and "torch" not in sys.modules
    write_new(args.work/"aggregate.json",{"experiment":registered["experiment"],"status":"passed",
               "source_manifest_sha256":digest(MANIFEST),"root_counts":base["roots"],
               "pretrained_decoder_forwards":0,"formal_comparisons":0,"sealed_test_accessed":False,
               "per_backend":results,"resources":{"max_sampled_private_bytes":max(x["private_bytes"] for x in samples),
               "max_sampled_working_set_bytes":max(x["working_set_bytes"] for x in samples),
               "min_sampled_available_RAM_bytes":min(x["available_physical_bytes"] for x in samples)}})


if __name__=="__main__":
    main()
