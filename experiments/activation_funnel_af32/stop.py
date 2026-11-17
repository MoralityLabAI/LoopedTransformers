"""AF-32 train-only learned STOP over public source-feature summaries.

No source truth, bitset BFS or symbolic fixed point enters feature_row().
Labels are supplied only by TRAIN physical-BFS in make_rows(); DEV labels
choose a frozen threshold. Held decisions use only a learned ranker, current
public graph and proven bits. Exact Horn certificate is a separate arm.
"""
from __future__ import annotations
import json
import math
from . import features, gym, ranker

DIM = 10
FEATURE_NAMES = (
    "min_reverse_goal_hops", "max_reverse_goal_hops",
    "mean_reverse_goal_hops", "min_reverse_blocked_cost",
    "max_reverse_blocked_cost", "mean_reverse_blocked_cost",
    "maximum_train_ranker_logit", "legal_rule_fraction",
    "known_atom_fraction", "mean_reverse_goal_ancestor",
)
UPDATES, STEP, L2 = 600, .1, .01
THRESHOLDS = tuple(round(i * .05, 2) for i in range(1, 20))


def _sigmoid(z):
    if z >= 0:
        return 1/(1+math.exp(-z))
    e=math.exp(z)
    return e/(1+e)


def feature_row(public, state, trained_ranker):
    """Public graph and proven physical bits only. No closure/search oracle."""
    vectors, work = features.extract(public, state)
    if not vectors:
        return ([-5.,-5.,-5.,-5.,-5.,-5.,-5.,0.,
                 len(state)/gym.ATOMS,0.], work)
    hops=[v[0] for v in vectors.values()]
    costs=[v[1] for v in vectors.values()]
    logits=[ranker.score(trained_ranker,v) for v in vectors.values()]
    values=[
        min(hops),max(hops),sum(hops)/len(hops),
        min(costs),max(costs),sum(costs)/len(costs),
        max(logits),len(vectors)/gym.RULES,
        len(state)/gym.ATOMS,
        sum(v[2] for v in vectors.values())/len(vectors),
    ]
    if len(values)!=DIM or not all(math.isfinite(a) for a in values):
        raise AssertionError("AF32 STOP public feature invalid")
    return values, work


def make_rows(split,trained_ranker,roots_only=False):
    if split not in ("train","dev"):
        raise ValueError("AF32 STOP supervised rows are TRAIN/DEV only")
    rows=[]
    for case in gym.cases_for(split):
        public=gym.public_features(case)
        state=frozenset(public["initial_physical"])
        steps=1 if roots_only else 3
        by_slot={slot:head for slot,prem,head in public["rules_physical"]}
        for t in range(steps):
            vector,work=feature_row(public,state,trained_ranker)
            truth=gym.public_bitset_bfs(public,state)
            rows.append({
                "split":split,"state_index":t,
                "case_source_sha256":gym.digest(public),
                "state_sha256":gym.digest(sorted(state)),
                "features":vector, "stop_target":int(truth is None),
            })
            legal=features.enabled(features.parse(public),state)
            if t+1 < steps:
                if not legal:
                    raise AssertionError("AF32 expected train successor absent")
                chosen=legal[0]
                state=state|{by_slot[chosen]}
                if public["goal_slot"] in state:
                    raise AssertionError("AF32 train STOP sample includes terminal proof")
    expected=192 if roots_only else 576
    if len(rows)!=expected:
        raise AssertionError("AF32 supervised STOP source row count changed")
    return rows


def fit(seed,trained_ranker,train=None,dev=None):
    if seed not in ranker.SEEDS or trained_ranker["seed"] != seed:
        raise ValueError("source training seed mismatch")
    train=make_rows("train",trained_ranker) if train is None else train
    dev=make_rows("dev",trained_ranker,True) if dev is None else dev
    if (len(train)!=576 or len(dev)!=192 or
        any(r["split"]!="train" for r in train) or
        any(r["split"]!="dev" or r["state_index"]!=0 for r in dev)):
        raise ValueError("AF32 STOP train/dev boundary violated")
    y=[r["stop_target"] for r in train]
    n1=sum(y);n0=len(y)-n1
    if not n1 or not n0 or n1!=n0:
        raise AssertionError("AF32 train STOP labels not balanced")
    matrix=[r["features"] for r in train]
    if any(len(v)!=DIM for v in matrix):
        raise ValueError("AF32 STOP input dimension invalid")
    mean=[sum(v[j] for v in matrix)/len(matrix) for j in range(DIM)]
    std=[max((sum((v[j]-mean[j])**2 for v in matrix)/len(matrix))**.5,1e-6)
         for j in range(DIM)]
    xs=[norm(v,mean,std) for v in matrix]
    w=[0.]*(DIM+1)
    trajectory=[]
    for epoch in range(UPDATES):
        grad=[0.]*(DIM+1)
        loss=0.
        for x,target in zip(xs,y):
            z=w[0]+sum(a*b for a,b in zip(w[1:],x))
            p=_sigmoid(z)
            coeff=(.5/n1 if target else .5/n0)
            err=coeff*(p-target)
            grad[0]+=err
            for j in range(DIM):
                grad[j+1]+=err*x[j]
            if epoch in (0,99,199,299,399,499,599):
                loss += coeff*(math.log1p(math.exp(-abs(z)))+
                              max(z,0)-target*z)
        for j in range(1,DIM+1):
            grad[j]+=2*L2*w[j]
        if epoch in (0,99,199,299,399,499,599):
            trajectory.append({"update":epoch+1,"balanced_BCE_pre_update":loss})
        for j in range(DIM+1):
            w[j]-=STEP*grad[j]
    model={
        "experiment":"AF-32","seed":seed,
        "feature_names":list(FEATURE_NAMES),
        "mean":mean,"std":std,"weights_bias_first":w,
        "train_rows":len(train),"train_label_STOP":n1,
        "train_label_continue":n0,
        "train_sha256":gym.digest(train),
        "dev_sha256":gym.digest(dev),
        "updates":UPDATES,"step":STEP,"l2":L2,
        "train_loss_receipt":trajectory,
        "ranker_training_sha256":trained_ranker["training_row_sha256"],
    }
    p=[(r["stop_target"],predict(model,r["features"])) for r in dev]
    pos=[prob for label,prob in p if label==0]
    neg=[prob for label,prob in p if label==1]
    if len(pos)!=96 or len(neg)!=96:
        raise AssertionError("AF32 DEV roots unbalanced")
    grid=[]
    for t in THRESHOLDS:
        cont=sum(prob<t for prob in pos)
        stops=sum(prob>=t for prob in neg)
        grid.append({"threshold":t,"provable_continued":cont,
                     "correct_negative_STOP":stops,
                     "eligible":cont>=80})
    eligible=[x for x in grid if x["eligible"]]
    selected=max(eligible,key=lambda x:(x["correct_negative_STOP"],
                                        x["threshold"])) if eligible else grid[-1]
    model["dev_threshold"]=selected["threshold"]
    model["dev_choice"]=selected
    model["dev_grid"]=grid
    model["DEV_labels_seen_for_threshold_only"]=True
    return model


def norm(vector,mean,std):
    return tuple(max(-8.,min(8.,(v-m)/s))
                 for v,m,s in zip(vector,mean,std))


def predict(model,vector):
    x=norm(vector,model["mean"],model["std"])
    z=model["weights_bias_first"][0]+sum(
        w*a for w,a in zip(model["weights_bias_first"][1:],x))
    return _sigmoid(z)


def choose(model,public,state,trained_ranker):
    feats,work=feature_row(public,state,trained_ranker)
    p=predict(model,feats)
    return (p>=model["dev_threshold"]),p,work


if __name__=="__main__":
    raise SystemExit("Use experiments.activation_funnel_af32.evaluate for immutable full source runs")
