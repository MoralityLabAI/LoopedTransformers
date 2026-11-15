"""AF-21 independently audited preregistered four-seed science gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from . import experiment as ex

SEEDS=ex.SEEDS

def aggregate(root,out):
 root=Path(root);seed_results={};verification={
  'host_proof_rows':0,'counterfactual_rows':0,'forward_replays':0,
  'pair_forward_replays':0,'independent_train_pair_oracle_checks':0}
 for seed in SEEDS:
  folder=root/f's{seed}';training=folder/'training';evaluation=folder/'evaluation'
  tr=json.loads((training/'train_receipt.json').read_text())
  ev=json.loads((evaluation/'evaluation.json').read_text())
  audited=json.loads((folder/'independent_audit.json').read_text())
  if not audited['verified'] or audited['false_accepted_proofs']!=0:
   raise ValueError('AF21 independent proof audit failed')
  if ev['source_hashes']!=ex.source_hashes() or tr['source_hashes']!=ex.source_hashes() or \
    audited['evaluation_sha256']!=ex.sha(evaluation/'evaluation.json') or \
    audited['training_sha256']!=ex.sha(training/'train_receipt.json'):
   raise ValueError('AF21 pinned source or artifact hashes mismatch')
  for k in verification:verification[k]+=audited[k]
  eligible=all(tr['fit'][name][split]['proof_success']>=.65 for name in ('bce_l4','greedy_l4','greedy_rule_l4') for split in ('train','dev'))
  depth_results={};gates=[]
  for split in ('held_depth4','held_depth5'):
   p=ev['scores']['greedy_rule_l4'][split];b=ev['scores']['bce_l4'][split]
   q=ev['pairs']['greedy_rule_l4'][split];qb=ev['pairs']['bce_l4'][split]
   passed=(p['proof_success']>=b['proof_success']+.05-1e-10 and
       p['proof_success']>=.70 and
       p['early_stop_positive']<=b['early_stop_positive']+.02+1e-10 and
       q['flip_joint_order_accuracy']>=.75 and
       q['connected_mean_abs_reach_delta']<=min(.10,qb['connected_mean_abs_reach_delta']+.02) and
       q['connected_mean_abs_stop_delta']<=min(.10,qb['connected_mean_abs_stop_delta']+.02) and
       p['false_accepted_proofs']==0)
   depth_results[split]={
    'bce_success':b['proof_success'],'greedy_rule_success':p['proof_success'],
    'gain':p['proof_success']-b['proof_success'],
    'bce_early_stop':b['early_stop_positive'],'greedy_rule_early_stop':p['early_stop_positive'],
    'flip_joint_order_accuracy':q['flip_joint_order_accuracy'],
    'connected_reach_delta':q['connected_mean_abs_reach_delta'],
    'connected_stop_delta':q['connected_mean_abs_stop_delta'],
    'bce_connected_reach_delta':qb['connected_mean_abs_reach_delta'],
    'bce_connected_stop_delta':qb['connected_mean_abs_stop_delta'],
    'registered_pass':passed if eligible else None}
   gates.append(passed)
  verdict='inconclusive' if not eligible else 'pass' if all(gates) else 'fail'
  seed_results[str(seed)]={'eligible':eligible,'verdict':verdict,
    'train_dev':{name:tr['fit'][name] for name in ex.CONFIGS},
    'depths':depth_results,
    'held_layout':{name:{'proof_success':ev['scores'][name]['held_layout']['proof_success'],
      'flip_accuracy':ev['pairs'][name]['held_layout']['flip_joint_order_accuracy']}
       for name in ex.CONFIGS},
    'arms':{sp:{name:ev['scores'][name][sp]['proof_success'] for name in ex.CONFIGS}
        for sp in ('held_depth4','held_depth5')}}
 overall='fail' if any(x['verdict']=='fail' for x in seed_results.values()) else \
  'pass' if all(x['verdict']=='pass' for x in seed_results.values()) else 'inconclusive'
 result={'experiment':'AF-21','scientific_verdict':overall,'engineering_pass':True,
   'seed_results':seed_results,'verification':verification,
   'source_hashes':ex.source_hashes(),'protocol_sha256':ex.sha(ex.HERE/'PROTOCOL.json'),
   'manifest_sha256':ex.check_manifest()}
 ex.write(out,result)
 return result


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
 p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 print(json.dumps(aggregate(a.root,a.out)))
if __name__=='__main__':main()
