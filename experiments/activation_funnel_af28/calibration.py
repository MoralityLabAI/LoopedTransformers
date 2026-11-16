"""AF-28 prospective, grammar-disjoint STOP logistic calibration.

Only publicly visible frozen-model scores/heads and host-current facts enter
the gate features. STOP truth is used as a supervised target ONLY for train
and development. Heldout labels are never touched during fit or threshold
selection. All optimizers are deterministic full-batch balanced BCE.
"""
from __future__ import annotations
import json,hashlib
import numpy as np

FEATURE_NAMES=(
    "direct_stop_minus_best_legal",
    "root_reach_logit_div4",
    "root_distance_logit_div4",
    "vi2_stop_minus_best_legal_div8",
    "legal_count_minus4_div4",
    "known_fact_count_minus4_div4",
)
DIM=6
UPDATES=600
STEP=0.1
L2=0.01
THRESHOLDS=tuple(round(i*0.05,2) for i in range(1,20))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sig(x):
    x=np.asarray(x,dtype=np.float64)
    return np.where(x>=0,1/(1+np.exp(-np.maximum(x,0))),
                    np.exp(np.minimum(x,0))/(1+np.exp(np.minimum(x,0))))


def feature_row(plan,known,legal):
    legal=tuple(legal)
    if not legal:
        raise ValueError("source STOP gate only scored with legal rules")
    if tuple(sorted(legal))!=legal or len(set(legal))!=len(legal):
        raise ValueError("host physical legal-rule mask unordered or repeated")
    if any(a<0 or a>=14 for a in legal):
        raise ValueError("illegal Horn action in feature extraction")
    if "direct" not in plan["scores"] or "vi2" not in plan["scores"]:
        raise ValueError("original Direct/VI2 score outputs absent")
    direct=plan["scores"]["direct"]
    vi2=plan["scores"]["vi2"]
    nodes=plan["trace"]["nodes"]
    root=nodes[0]
    if tuple(sorted(root["state"]))!=tuple(sorted(known)):
        raise ValueError("source model root is not current public Horn state")
    if len(direct)!=15 or len(vi2)!=15:
        raise ValueError("original frozen model requires 15 action scores")
    values=(
        direct[14]-max(direct[a] for a in legal),
        float(root["reach_logit"])/4,
        float(root["distance_logit"])/4,
        (vi2[14]-max(vi2[a] for a in legal))/8,
        (len(legal)-4)/4,
        (len(known)-4)/4,
    )
    v=np.asarray(values,dtype=np.float64)
    if v.shape!=(DIM,) or not np.isfinite(v).all():
        raise ValueError("infinite or wrong source STOP features")
    return np.clip(v,-8,8).tolist()


def _weighted_targets(labels):
    y=np.asarray(labels,dtype=np.float64)
    if y.ndim!=1 or len(y)<2 or not ((y==0)|(y==1)).all():
        raise ValueError("calibration labels must be nonempty 0/1")
    yes=(y==1).sum()
    no=(y==0).sum()
    if not yes or not no:
        raise ValueError("calibration requires both negative and positive goals")
    weight=np.where(y==1,0.5/yes,0.5/no)
    return y,weight


def fit(rows,feature_indices):
    feature_indices=tuple(feature_indices)
    if not feature_indices or any(i<0 or i>=DIM for i in feature_indices):
        raise ValueError("invalid registered calibration feature subset")
    Xraw=np.asarray([r["features"] for r in rows],dtype=np.float64)
    y,w=_weighted_targets([r["stop_target"] for r in rows])
    if Xraw.shape!=(len(rows),DIM) or not np.isfinite(Xraw).all():
        raise ValueError("source training features invalid")
    if len(set(r["split"] for r in rows))!=1 or rows[0]["split"]!="train":
        raise ValueError("STOP calibration fitting requires train-only samples")
    Xraw=Xraw[:,feature_indices]
    mean=Xraw.mean(axis=0)
    std=Xraw.std(axis=0)
    std=np.maximum(std,1e-6)
    X=np.clip((Xraw-mean)/std,-8.,8.)
    A=np.column_stack([np.ones(len(rows)),X])
    coef=np.zeros(A.shape[1],dtype=np.float64)
    trajectory=[]
    for step in range(UPDATES):
        z=A@coef
        p=sig(z)
        gradient=A.T@(w*(p-y))
        gradient[1:]+=2*L2*coef[1:]
        norm=np.linalg.norm(gradient)
        if norm>10:
            gradient=gradient*(10/norm)
        coef-=STEP*gradient
        if step in (0,59,119,239,359,479,599):
            zz=A@coef
            # Weighted BCE with numerically stable logaddexp.
            loss=float(sum(w*(np.logaddexp(0,zz)-y*zz))+
                       L2*sum(coef[1:]**2))
            trajectory.append({"step":step+1,"balanced_bce_l2":loss})
    if not np.isfinite(coef).all():
        raise FloatingPointError("nonfinite trained STOP gate coefficients")
    return {
        "feature_indices":list(feature_indices),
        "feature_names":[FEATURE_NAMES[i] for i in feature_indices],
        "train_only_mean":mean.tolist(),
        "train_only_std":std.tolist(),
        "weights_bias_first":coef.tolist(),
        "updates":UPDATES,"step":STEP,"l2":L2,
        "train_rows":len(rows),
        "train_label_positive":int(sum(y)),
        "train_label_negative":int(len(y)-sum(y)),
        "loss_receipt":trajectory,
    }


def predict(model,features):
    selected=np.asarray(features,dtype=np.float64)[model["feature_indices"]]
    mean=np.asarray(model["train_only_mean"],dtype=np.float64)
    std=np.asarray(model["train_only_std"],dtype=np.float64)
    a=np.clip((selected-mean)/std,-8.,8.)
    z=float(np.asarray(model["weights_bias_first"])[0]+
            np.dot(a,model["weights_bias_first"][1:]))
    return float(sig(z))


def select_threshold(model,dev_rows):
    if (len(dev_rows)!=192 or
        len({r["case_id"] for r in dev_rows})!=192 or
        any(r["split"]!="dev" for r in dev_rows)):
        raise ValueError("registered threshold selection requires 192 dev roots only")
    pos=[r for r in dev_rows if r["stop_target"]==0]
    neg=[r for r in dev_rows if r["stop_target"]==1]
    if (len(pos),len(neg))!=(80,112):
        raise ValueError("development truth/negative balance changed")
    entries=[]
    for t in THRESHOLDS:
        p_continue=sum(predict(model,r["features"])<t for r in pos)
        n_stop=sum(predict(model,r["features"])>=t for r in neg)
        entries.append({
            "threshold":t,"continue_on_provable":p_continue,
            "correct_immediate_STOP_on_unprovable":n_stop,
            "passes_dev_positive_continuation_floor":p_continue>=72,
        })
    eligible=[r for r in entries if r["passes_dev_positive_continuation_floor"]]
    best=max(eligible,key=lambda r:(r["correct_immediate_STOP_on_unprovable"],
                                    r["threshold"])) if eligible else entries[-1]
    return {
        "chosen_threshold":best["threshold"],
        "dev_chosen_positive_continue":best["continue_on_provable"],
        "dev_chosen_correct_negative_STOP":best["correct_immediate_STOP_on_unprovable"],
        "used_eligible_dev_floor":bool(eligible),
        "dev_grid_receipt":entries,
    }


def make_models(train_rows,dev_rows):
    if len(train_rows)<192:
        raise ValueError("incomplete source train STOP sample set")
    learned=fit(train_rows,tuple(range(DIM)))
    simple=fit(train_rows,(0,))
    learned["dev"]=select_threshold(learned,dev_rows)
    simple["dev"]=select_threshold(simple,dev_rows)
    return {"CAL":learned,"THR":simple}


def choose(prob,threshold):
    if not 0<=prob<=1 or threshold not in THRESHOLDS:
        raise ValueError("invalid fixed calibrated probability/threshold")
    return prob>=threshold
