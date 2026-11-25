"""ROC-06A independent stdlib verification of causal receipts and full null family."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,itertools,json,math
from collections import defaultdict
from pathlib import Path
from statistics import mean,median
SEEDS=(20261117231,20261117233,20261117239,20261117241,20261117251,20261117257)
ARMS=("low_affine4","high_permutation512")
ROUTES=("full_legal_unmatched","input_soft","input_induced_hidden",
        "supervised_observed","supervised_dark","gradient_observed",
        "gradient_dark","pca_observed","random_observed","isotropic")
RADII=(.05,.10,.20)
N=256
ROWS_PER_MODEL=N*len(RADII)*len(ROUTES)
STUDY="ROC-06A-v1-train-only-counterfactual-gradient-observer"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def close(a,b,label,tol=1e-9):
    if a is None or b is None:
        if a is not b:raise AssertionError(label+" undefined mismatch")
        return
    if not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>tol:
        raise AssertionError(f"{label} {a} != {b}")

def p_tail(x,values,lower):
    if x is None or any(v is None for v in values):return None
    return (1+sum(v<=x if lower else v>=x for v in values))/129.

def holm(p):
    out=[None]*6;run=0.
    for index,(v,i) in enumerate(sorted((v,i) for i,v in enumerate(p) if v is not None)):
        run=max(run,min(1.,(6-index)*v))
        out[i]=run
    return out

def verify_cards(receipt):
    cards=receipt["shadow_cards"]
    nulls=receipt["null_rows"]
    if len(cards)!=6 or len(nulls)!=256:
        raise AssertionError("ROC06A 72-card /3072 null census model error")
    classified=0
    mean_b={}
    for shadow in (0,1):
        raw=[r for r in nulls if r["shadow_index"]==shadow]
        if len(raw)!=128 or [r["null_seed"] for r in raw]!=list(range(941001,941129)):
            raise AssertionError("wrong null shadow, seed, rank or count")
        if any(r["rank"]!=7 for r in raw):
            raise AssertionError("wrong rank in random QR null pool")
        vals={k:[v[k] for v in raw] for k in ("a","b","c","s")}
        med={k:median(v) for k,v in vals.items()}
        obs=[r for r in cards if r["shadow_index"]==shadow]
        if {r["basis"] for r in obs}!={"gradient","supervised","pca"}:
            raise AssertionError("wrong observer bases")
        for r in obs:
            if not r["calibration_only"] or r["rank"]!=7 or r["null_distinct"]!=128:
                raise AssertionError("invalid observer card provenance")
            for k in med:
                close(r["null_medians"][k],med[k],"null median",1e-10)
            m=r["metric"]
            tails=[p_tail(m["a"],vals["a"],True),
                   p_tail(m["b"],vals["b"],True),
                   p_tail(m["c"],vals["c"],True),
                   p_tail(m["b"],vals["b"],False),
                   p_tail(m["s"],vals["s"],False),
                   p_tail(m["s"],vals["s"],True)]
            adj=holm(tails)
            for i in range(6):
                close(tails[i],r["one_sided_p"][i],"one-sided null probability")
                close(adj[i],r["holm_q"][i],"Holm correction")
            b=r["ratios"]["b"]
            if b is not None:
                close(b,m["b"]/med["b"],"rank7 B null ratio",1e-9)
                mean_b[(shadow,r["basis"])]=b
            classified+=int(r["tags"]!=["unclassified"])
    return classified,mean_b

def cell(folder,seed,arm,cfg):
    stem=f"roc06a_S{seed}_{arm}"
    path=folder/(stem+"_receipt.json")
    r=json.loads(path.read_text())
    g=folder/(stem+".csv.gz")
    if r["study"]!=STUDY or r["seed"]!=seed or r["arm"]!=arm:
        raise AssertionError("wrong ROC06 model receipt")
    if (r["checkpoint_sha256"]!=cfg["parent"]["original_1920step_checkpoint_sha256"][str(seed)][arm]
        or r["source_dev_sha256"]!=cfg["parent"]["source_sha256"]["development"]
        or r["observer_source_train_sha256"]!=cfg["parent"]["source_sha256"]["train_high512" if arm=="high_permutation512" else "train_low4"]
        or r["preregistration_commit"]!="1c209f6db0f56dd0bb4d4677d1fb22a8cd462a8c"):
        raise AssertionError("model, training observer source, or dev SHA mismatch")
    if (r["model_training_updates_during_causal_analysis"]!=0 or
        r["sealed_test_evaluated"] or
        r["number_of_cases"]!=256 or
        r["per_case_row_count"]!=ROWS_PER_MODEL or
        r["nulls_per_shadow"]!=128 or
        r["native_baseline_max_error"]>1e-5):
        raise AssertionError("model training, native forward or route census error")
    info=r["basis_singulars"]
    if not info["train_only"] or info["train_gradient_count"]!=1024 or info["train_gradient_s7"]<=1e-9:
        raise AssertionError("gradient basis fit not train-only full-rank")
    if sha(g)!=r["per_case_csv_gzip_sha256"]:
        raise AssertionError("compressed causal CSV SHA mismatch")
    nclassified,bratio=verify_cards(r)
    with gzip.open(g,"rt",newline="") as fp:
        rows=list(csv.DictReader(fp))
    if len(rows)!=ROWS_PER_MODEL:
        raise AssertionError("incomplete per-case physical arms")
    seen={}
    case_info={}
    for row in rows:
        i=int(row["case_index"]);rad=float(row["radius"]);route=row["route"]
        if (int(row["seed"])!=seed or row["model_arm"]!=arm or
            row["source_family"]!="roc05_dev_unseen"
            or i not in range(N) or rad not in RADII or route not in ROUTES):
            raise AssertionError("unregistered model/case/radius/route")
        key=(i,rad,route)
        if key in seen:raise AssertionError("duplicate route record")
        seen[key]=row
        label_tuple=(row["source_id"],row["original_label"],row["legal_changed_label"],row["query"])
        if label_tuple[1]==label_tuple[2]:
            raise AssertionError("unchanged legal source edit target")
        if i in case_info and case_info[i]!=label_tuple:
            raise AssertionError("case identity changed between routes")
        case_info[i]=label_tuple
        valid=int(row["valid"])
        dose=float(row["midpoint_l2"])
        if valid not in (0,1) or not math.isfinite(dose) or dose<0:
            raise AssertionError("invalid physical dosage")
        if valid:
            if row["margin_gain"] in ("",None) or not math.isfinite(float(row["margin_gain"])):
                raise AssertionError("valid causal margin missing")
            if route!="full_legal_unmatched" and abs(dose-rad)>2e-5:
                raise AssertionError("unmatched physical hidden radius")
        elif row["margin_gain"] not in ("",None):
            raise AssertionError("abstention has causal effect")
        if int(row["soft_off_manifold"])!=int(route=="input_soft"):
            raise AssertionError("soft-input provenance mismatch")
        if int(row["full_legal_natural_dose"])!=int(route=="full_legal_unmatched"):
            raise AssertionError("natural-dose legal source flag mismatch")
    if len(seen)!=ROWS_PER_MODEL or len(case_info)!=N:
        raise AssertionError("incomplete case × route × radius cells")
    triple=[]
    legal=original=0
    for i in range(N):
        base=seen[(i,.2,"full_legal_unmatched")]
        legal+=int(base["legal_full_pred"]==case_info[i][2])
        original+=int(base["original_pred"]==case_info[i][1])
        entries=[seen[(i,.2,k)] for k in ("gradient_observed","supervised_observed","input_soft")]
        if all(int(z["valid"]) for z in entries):
            triple.append([float(z["margin_gain"]) for z in entries])
    if original!=r["original_correct"] or legal!=r["full_legal_changed_correct"]:
        raise AssertionError("original/edited true-label correctness mismatch")
    n=len(triple)
    if n:
        mg=mean(t[0] for t in triple)
        ms=mean(t[1] for t in triple)
        mi=mean(t[2] for t in triple)
        best=max(ms,mi)
    else:
        mg=ms=mi=best=None
    obs={"paired_eligible":n,"eligibility_fraction":n/N,
         "gradient_gain":mg,"supervised_gain":ms,"soft_input_gain":mi,
         "gradient_minus_supervised":mg-ms if n else None,
         "gradient_minus_soft_input":mg-mi if n else None,
         "best_reference_gain":best,"gradient_minus_best":mg-best if n else None,
         "best_reference":("supervised_observed" if ms>=mi else "input_soft") if n else None}
    for k,v in obs.items():
        if isinstance(v,(float,type(None))) and k!="best_reference":
            close(v,r["primary"].get(k),f"{seed}/{arm}/{k}")
        elif r["primary"].get(k)!=v:
            raise AssertionError("reported primary category discrepancy "+k)
    return {"seed":seed,"arm":arm,"primary":obs,"original_correct":original,
            "legal_changed_correct":legal,"classified_cards":nclassified,
            "B_ratios":bratio,"source_cases":case_info,
            "train_gradient_energy7":info["train_gradient_explained_energy7"],
            "gradient_vs_supervised_cosines":info["gradient_vs_supervised_cosines"],
            "receipt_sha256":sha(path),"csv_sha256":sha(g)}

def analyze(folder,config_path):
    cfg=json.loads(config_path.read_text())
    if cfg["study"]!=STUDY or cfg["parent"]["original_train_run"]!=44101721734:
        raise AssertionError("wrong registered primary")
    cells={}
    for seed in SEEDS:
        for arm in ARMS:
            cells[(seed,arm)]=cell(folder,seed,arm,cfg)
    keys=next(iter(cells.values()))["source_cases"]
    if any(c["source_cases"]!=keys for c in cells.values()):
        raise AssertionError("dev cases not paired across original frozen models")
    diffs=[]
    perseed=[]
    for seed in SEEDS:
        high=cells[(seed,"high_permutation512")]
        low=cells[(seed,"low_affine4")]
        p=high["primary"]
        diffs.append(p["gradient_minus_best"])
        perseed.append({"seed":seed,"high":p,"low":low["primary"],
              "high_original_correct":high["original_correct"],
              "high_legal_changed_correct":high["legal_changed_correct"],
              "high_gradient_energy7":high["train_gradient_energy7"],
              "high_subspace_cosines":high["gradient_vs_supervised_cosines"]})
    available=all(p["high"]["paired_eligible"]>=205 for p in perseed)
    if not available or any(d is None or not math.isfinite(d) for d in diffs):
        gate="ABSTAIN";avg=pos=extreme=pvalue=None
    else:
        avg=mean(diffs)
        pos=sum(d>0 for d in diffs)
        distribution=[mean(sig*d for sig,d in zip(signs,diffs))
                      for signs in itertools.product((-1,1),repeat=6)]
        extreme=sum(value>=avg-1e-15 for value in distribution)
        pvalue=extreme/64
        gate="PASS" if pos>=5 and avg>0 and pvalue<=.05 else "FAIL"
    primary_mean_route={k:mean(p["high"][k] for p in perseed if p["high"][k] is not None)
                        for k in ("gradient_gain","supervised_gain","soft_input_gain")}
    return {"study":STUDY,"scope":"exposed ROC05 development, frozen ROC05A first-run model weights",
            "original_roc02_sealed_evaluated":False,
            "roc05_reserved_evaluated":False,"new_training_updates":0,
            "model_cells":12,"model_seed_count":6,"cases_per_model":256,
            "physical_case_routes":len(cells)*ROWS_PER_MODEL,
            "null_records":len(cells)*256,"observer_null_cards":len(cells)*6,
            "one_sided_holm_values":len(cells)*6*6,
            "classified_cards":sum(c["classified_cards"] for c in cells.values()),
            "high_six_seed_effects":diffs,"per_seed":perseed,
            "high_mean_route_gains":primary_mean_route,
            "primary_mean_gradient_minus_best":avg,
            "primary_positive_seed_count":pos,
            "exact_signflip_extreme_assignments":extreme,
            "exact_signflip_total":64,"exact_one_sided_p":pvalue,
            "primary_eligibility_met":available,"primary_gate":gate,
            "original_model_receipts":[{k:v for k,v in c.items() if k not in ("source_cases","B_ratios")}
                                       for c in cells.values()]}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--result-dir",type=Path,required=True)
    ap.add_argument("--config",type=Path,default=Path("configs/roc06a_train_counterfactual_gradient_observer_v1.json"))
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    z=analyze(a.result_dir,a.config)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(z,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:v for k,v in z.items() if k not in ("original_model_receipts","per_seed")},sort_keys=True,indent=2))
if __name__=="__main__":
    main()
