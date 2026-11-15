"""AF-17 post-hoc diagnostic, executed after frozen scientific evaluation.

For fixed Horn topology, flip one initial fact located outside the model's
two-update receptive field of the goal atom. Reproduce trained STOP,
reachability and estimated distance outputs from original weights.
"""
import json
from pathlib import Path
import torch
from experiments.activation_funnel_af17 import experiment as ex, gym
from experiments.activation_funnel_af05 import gym as base


def run():
    torch.set_num_threads(2)
    rows=[]
    for seed in ex.SEEDS:
        _,models=ex.load_models(Path('/mnt/data/AF17_local_run')/f's{seed}/training',seed)
        for arm,model in models.items():
            for d in (3,4,5):
                a=gym.Case('held_depth5','dual_fallback',d,'and',(0,1,1,1),55)
                b=gym.Case('held_depth5','dual_fallback',d,'and',(1,1,1,1),55)
                x,k,mask=gym.observations([a,b],
                    [frozenset(gym.initial_facts(a)),frozenset(gym.initial_facts(b))])
                with torch.no_grad():scores,dist,reach=model.full(x,k)
                rows.append({'seed':seed,'arm':arm,'depth':d,
                    'first_input_provable':base.GOAL in gym.closure(a),
                    'second_input_provable':base.GOAL in gym.closure(b),
                    'stop_logit_difference':float(scores[0,-1]-scores[1,-1]),
                    'reachability_logit_difference':float(reach[0]-reach[1]),
                    'distance_logit_difference':float(dist[0]-dist[1]),
                    'max_other_rule_logit_difference':float((scores[0,:-1]-scores[1,:-1]).abs().max())})
    assert len(rows)==48
    assert all(not x['first_input_provable'] and x['second_input_provable'] for x in rows)
    assert all(x['stop_logit_difference']==0 and x['reachability_logit_difference']==0
               and x['distance_logit_difference']==0 for x in rows)
    assert any(x['max_other_rule_logit_difference']>0.01 for x in rows)
    out={'experiment':'AF-17','scope':'posthoc causal-receptive-field diagnostic',
         'frozen_training_scientific_source_commit':'89adc61cbef49b8879efbef8796e6e74208666b0',
         'trained_seeds':list(ex.SEEDS),'trained_arms':list(ex.ARMS),
         'pairs':len(rows),'goal_truth_flips':len(rows),'zero_stop_difference':len(rows),
         'zero_distance_difference':len(rows),'zero_reach_difference':len(rows),
         'rows':rows}
    return out

if __name__=='__main__':
    r=run();dest=Path('/mnt/data/AF17_local_run/posthoc_receptive_field.json')
    ex.write(dest,r)
    print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
