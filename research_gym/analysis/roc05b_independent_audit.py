"""ROC05B stdlib independent causal receipt census and six-seed exact test."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,itertools,json,math
from collections import defaultdict
from pathlib import Path
from statistics import mean

SEEDS=(20261117231,20261117233,20261117239,20261117241,20261117251,20261117257)
ARMS=("low_affine4","high_permutation512")
ROUTES=("full_legal_unmatched","input_soft","input_induced_hidden",
        "supervised_observed","supervised_dark","pca_observed",
        "random_observed","isotropic")
RADII=(.05,.10,.20)
N=256
TOTAL=6*2*256*3*8

def sha(path:Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def audit(folder:Path,config_path:Path):
    cfg=json.loads(config_path.read_text())
    if cfg["study"]!="ROC-05B-v1-matched-hidden-input-causal-rule-intervention":
        raise AssertionError("study identity")
    per_seed=[]
    all_count=0
    n_unclassified=0
    for seed in SEEDS:
        byarm={}
        for arm in ARMS:
            recfile=folder/f"roc05b_S{seed}_{arm}_receipt.json"
            receipt=json.loads(recfile.read_text())
            if receipt["seed"]!=seed or receipt["arm"]!=arm:
                raise AssertionError("model identity mismatch")
            if receipt["checkpoint_sha256"]!=cfg["parent"]["frozen_1920step_checkpoint_sha256"][str(seed)][arm]:
                raise AssertionError("first-run frozen model SHA mismatch")
            if receipt["source_dev_sha256"]!=cfg["parent"]["exact_source_sha256"]["development"]:
                raise AssertionError("source SHA mismatch")
            if receipt["model_training_updates_during_causal_analysis"]!=0 or receipt["sealed_test_evaluated"]:
                raise AssertionError("forbidden training or sealed test access")
            if receipt["number_of_cases"]!=N or receipt["per_case_row_count"]!=N*len(RADII)*len(ROUTES):
                raise AssertionError("missing case count")
            if receipt["nulls_per_shadow"]!=128 or len(receipt["shadow_cards"])!=4:
                raise AssertionError("incomplete null-calibrated geometry")
            for card in receipt["shadow_cards"]:
                if (card["rank"]!=7 or card["shadow_index"] not in (0,1)
                    or card["basis"] not in ("supervised","pca")
                    or card["null_distinct"]!=128 or not card["calibration_only"]):
                    raise AssertionError("wrong local observer null card")
                if len(card["one_sided_p"])!=6 or len(card["holm_q"])!=6:
                    raise AssertionError("wrong six-direction Holm family")
                n_unclassified+=int(card["tags"]==["unclassified"])
            gzpath=folder/f"roc05b_S{seed}_{arm}.csv.gz"
            if sha(gzpath)!=receipt["per_case_csv_gzip_sha256"]:
                raise AssertionError("corrupt per-case result gzip")
            with gzip.open(gzpath,"rt",newline="") as f:
                records=list(csv.DictReader(f))
            if len(records)!=N*len(RADII)*len(ROUTES):
                raise AssertionError("per-case data incomplete")
            all_count+=len(records)
            bykey={}
            for x in records:
                i=int(x["case_index"]);r=float(x["radius"]);route=x["route"]
                if (int(x["seed"])!=seed or x["model_arm"]!=arm
                    or not 0<=i<N or r not in RADII or route not in ROUTES):
                    raise AssertionError("wrong per-case key")
                key=(i,r,route)
                if key in bykey:
                    raise AssertionError("duplicate route/dose/case")
                bykey[key]=x
                if (x["source_family"]!="roc05_dev_unseen"
                    or len(x["source_id"])!=64 or x["original_label"]==x["legal_changed_label"]):
                    raise AssertionError("wrong source provenance or legal changed target")
                valid=int(x["valid"])
                if valid not in (0,1):
                    raise AssertionError("invalid validity flag")
                if valid and x["margin_gain"]=="":
                    raise AssertionError("missing valid causal output")
                if valid and (not math.isfinite(float(x["margin_gain"]))
                              or not math.isfinite(float(x["midpoint_l2"]))):
                    raise AssertionError("nonfinite causal output")
                if valid and route not in ("full_legal_unmatched",) and abs(float(x["midpoint_l2"])-r)>2e-5:
                    raise AssertionError("physically unmatched L2 route")
                if int(x["soft_off_manifold"])!=int(route=="input_soft"):
                    raise AssertionError("soft-input validity provenance mismatch")
            if len(bykey)!=N*len(RADII)*len(ROUTES):
                raise AssertionError("missing route/case combination")
            paired=[]
            obs=[]
            softened=[]
            for i in range(N):
                a=bykey[(i,.2,"supervised_observed")]
                b=bykey[(i,.2,"input_soft")]
                if a["source_id"]!=b["source_id"]:
                    raise AssertionError("paired case identity changed")
                if int(a["valid"]) and int(b["valid"]):
                    ga=float(a["margin_gain"]);gb=float(b["margin_gain"])
                    paired.append(ga-gb);obs.append(ga);softened.append(gb)
            p=receipt["primary"]
            if p["paired_eligible"]!=len(paired):
                raise AssertionError("paired eligibility mismatch")
            if abs(p["eligibility_fraction"]-len(paired)/N)>1e-12:
                raise AssertionError("paired eligible fraction")
            if paired:
                if any(abs(x-y)>1e-9 for x,y in ((mean(paired),p["primary_observed_minus_soft"]),
                    (mean(obs),p["primary_rank7_gain"]),(mean(softened),p["primary_soft_input_gain"]))):
                    raise AssertionError("primary paired seed effects diverge from per-case rows")
            byarm[arm]={
              "paired_eligible":len(paired),"mean_observed_gain":mean(obs) if obs else None,
              "mean_soft_input_gain":mean(softened) if softened else None,
              "mean_observed_minus_soft":mean(paired) if paired else None,
              "original_correct":receipt["original_correct"],
              "legal_changed_correct":receipt["full_legal_changed_correct"],
              "shadow_B_to_null_supervised_mean":mean(
                  x["ratios"]["b"] for x in receipt["shadow_cards"] if x["basis"]=="supervised")
            }
        per_seed.append({"seed":seed,"arms":byarm,
            "primary_diff":byarm["high_permutation512"]["mean_observed_minus_soft"],
            "paired_eligible_high":byarm["high_permutation512"]["paired_eligible"]})
    if all_count!=TOTAL:
        raise AssertionError("total per-case record census")
    eligible=all(r["paired_eligible_high"]>=205 for r in per_seed)
    diffs=[r["primary_diff"] for r in per_seed]
    if all(x is not None for x in diffs):
        observed=mean(diffs)
        outcomes=[mean(a*d for a,d in zip(signs,diffs))
                  for signs in itertools.product((-1.,1.),repeat=6)]
        ge=sum(x>=observed-1e-15 for x in outcomes)
        p=ge/64
        pos=sum(x>0 for x in diffs)
    else:
        observed=None;ge=None;p=None;pos=None
    gate=("ABSTAIN" if not eligible or observed is None
          else ("PASS" if pos>=5 and observed>0 and p<=.05 else "FAIL"))
    return {
      "study":"ROC-05B-v1-matched-hidden-input-causal-rule-intervention",
      "scientific_status":"exposed-development causal diagnostic",
      "model_count":12,"unique_initialization_seeds":6,
      "cases_per_model":N,"radiuses":list(RADII),"routes":list(ROUTES),
      "per_case_rows":all_count,"complete_null_cards":48,
      "unclassified_cards":n_unclassified,
      "per_seed":per_seed,"primary_six_seed_effects":diffs,
      "primary_mean_observed_minus_input":observed,
      "primary_eligibility_all_six_at_least_80pct":eligible,
      "primary_positive_seed_count":pos,
      "exact_64_sign_extreme_count":ge,
      "one_sided_exact_signflip_p":p,
      "primary_gate":gate,
      "original_roc02_sealed_evaluated":False,
      "roc05_reserved_test_evaluated":False,
      "new_training_updates":0
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--receipt-dir",type=Path,required=True)
    p.add_argument("--config",type=Path,default=Path("configs/roc05b_matched_rule_causal_v1.json"))
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    r=audit(a.receipt_dir,a.config)
    a.out.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:v for k,v in r.items() if k!="per_seed"},sort_keys=True,indent=2))

if __name__=="__main__":
    main()
