"""AF-26 read-only diagnostic for failed original AF-25 VI-2 parity.

This does not alter AF-26 source-locked policies or the registered outcome.
It reads an immutable original AF-25 seed archive and original AF-23 weights,
reruns AF-25's exact original three-mode model evaluator, and reports the
first divergence between the original scored and fresh trace, if any.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import torch

from experiments.activation_funnel_af25 import gym
from experiments.activation_funnel_af25 import evaluate as af25

AF25_ZIP={
    29:"28d82740b0c9b72d304fb236a4ca71919d839dc1deb2a1e787e489f93b282618",
    103:"b07f2c4156ed3c626dadf188e7566c597a04f7957e771f8b8467674c81faeb27",
    131:"4c9411ad6cb9eea47ee1e5ae439ecfb3efb4e134bcef05aeced75081eebbe5e0",
}


def first_change(old,new):
    a,b=old["actions"],new["actions"]
    for k in range(max(len(a),len(b))):
        if k>=min(len(a),len(b)):
            return {"step":k,"kind":"action_length_changed",
                    "old_len":len(a),"new_len":len(b)}
        s,t=a[k],b[k]
        if s["action"] != t["action"] or s["before"] != t["before"]:
            out={
                "step":k,"kind":"action_or_state_diverged",
                "same_before":s["before"]==t["before"],
                "old_action":s["action"],"new_action":t["action"],
                "old_enabled":s["enabled"],"new_enabled":t["enabled"],
                "old_state":s["before"],"new_state":t["before"],
            }
            if s["before"]==t["before"]:
                old_sc,new_sc=s["decision_scores"],t["decision_scores"]
                out["max_source_score_absolute_difference"]=max(
                    abs(x-y) for x,y in zip(old_sc,new_sc)
                )
                out["old_sorted_actions"]=sorted(
                    range(len(old_sc)),key=lambda i:(-old_sc[i],i)
                )[:4]
                out["new_sorted_actions"]=sorted(
                    range(len(new_sc)),key=lambda i:(-new_sc[i],i)
                )[:4]
                out["old_top4_scores"]=[old_sc[i] for i in out["old_sorted_actions"]]
                out["new_top4_scores"]=[new_sc[i] for i in out["new_sorted_actions"]]
                out["old_forward_nodes"]=s["plan_trace"]["nodes_forward_count"]
                out["new_forward_nodes"]=t["plan_trace"]["nodes_forward_count"]
            return out
    return None


def diagnose(seed,model_dir,original_af25_zip,out):
    if seed not in AF25_ZIP:
        raise ValueError("unregistered diagnostic-only seed")
    path=Path(original_af25_zip)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=AF25_ZIP[seed]:
        raise ValueError("immutable original AF25 seed ZIP differs")
    with zipfile.ZipFile(path) as archive:
        receipt=json.loads(archive.read("evaluation.json"))
        original_rows=[json.loads(x) for x in archive.read("episodes.jsonl").splitlines()]
    if (receipt["experiment"],receipt["seed"])!=("AF-25",seed):
        raise ValueError("wrong original scored seed")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model,provenance=af25.load_frozen_original(model_dir,seed)
    old_index={(r["split"],r["mode"],r["case_id"]):r for r in original_rows}
    output={
        "scope":"source-only original action parity diagnostic",
        "scientific_verdict":"UNSCORED",
        "seed":seed,"original_scored_ci":43829157457,
        "model_source":provenance,
        "splits":{},
    }
    for split in gym.SPLITS:
        fresh=af25.neural_rollout(model,gym.cases_for(split))
        modes={}
        for mode in ("direct","vi1","vi2"):
            original_success=receipt["scores"][split][mode]["solved"]
            fresh_success=sum(r["solved"] for r in fresh[mode] if r["goal_truth"])
            differences=[]
            for new in fresh[mode]:
                old=old_index[(split,mode,new["case_id"])]
                diff=first_change(old,new)
                if diff is not None:
                    differences.append({
                        "case_id":new["case_id"],
                        "original_solved":old["solved"],
                        "fresh_solved":new["solved"],
                        "difference":diff,
                    })
            modes[mode]={
                "original_solved":original_success,
                "fresh_solved":fresh_success,
                "number_diverged_case_paths":len(differences),
                "number_different_completion_labels":sum(x["original_solved"]!=x["fresh_solved"] for x in differences),
                "first_12_divergences":differences[:12],
            }
        output["splits"][split]=modes
    Path(out).write_text(json.dumps(output,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "seed":seed,
        "d5":output["splits"]["held_depth5"]["vi2"]["fresh_solved"],
        "d5_original":output["splits"]["held_depth5"]["vi2"]["original_solved"],
        "d5_divergent_cases":output["splits"]["held_depth5"]["vi2"]["number_diverged_case_paths"],
        "d6":output["splits"]["held_depth6"]["vi2"]["fresh_solved"],
    }))
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True,choices=AF25_ZIP)
    p.add_argument("--original-zip-dir",type=Path,required=True)
    p.add_argument("--af25-episode-zip",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    diagnose(a.seed,a.original_zip_dir,a.af25_episode_zip,a.out)


if __name__=="__main__":
    main()
