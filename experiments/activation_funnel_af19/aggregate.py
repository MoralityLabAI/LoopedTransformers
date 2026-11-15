"""Source-bound AF-19 scientific gate and cross-seed engineering audit."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from . import experiment as ex


def aggregate(root,out):
    root=Path(root);all_results={};verified={'episodes':0,'forward_replays':0,
        'independent_bfs_cost_checks':0,'cone_probe_replays':0}
    any_fail=False
    for seed in ex.SEEDS:
        folder=root/f's{seed}'
        train=folder/'training';eval_dir=folder/'evaluation'
        a=json.loads((folder/'independent_audit.json').read_text())
        scores=json.loads((eval_dir/'evaluation.json').read_text())
        tr=json.loads((train/'train_receipt.json').read_text())
        if not a['passed'] or scores['source_hashes']!=ex.hashes() or tr['source_hashes']!=ex.hashes():
            raise ValueError('AF19 independent seed verification failed')
        if scores['train_sha256']!=ex.sha(train/'train_receipt.json') or a['evaluation_sha256']!=ex.sha(eval_dir/'evaluation.json'):
            raise ValueError('AF19 aggregate receipt mismatch')
        for k in verified:verified[k]+=a[k]
        qualify=all(tr['fit'][c][part]['proof_success']>=.65
            for c in ex.CONFIGS for part in ('train','dev'))
        arms={};eligible_fail=False
        for split in ('held_depth4','held_depth5'):
            q=scores['scores'];local2=q['local_l2'][split];anc2=q['ancestor_l2'][split]
            local4=q['local_l4'][split];anc4=q['ancestor_l4'][split]
            ok=(anc2['proof_success']>=local2['proof_success']+.05 and
                anc4['proof_success']>=local4['proof_success']+.05 and
                anc2['proof_success']>=.70 and anc4['proof_success']>=.70 and
                anc2['false_accepted_proofs']==anc4['false_accepted_proofs']==0 and
                anc2['early_stop_positive']<=local2['early_stop_positive']+.02 and
                anc4['early_stop_positive']<=local4['early_stop_positive']+.02)
            arms[split]={'local_l2':local2['proof_success'],'ancestor_l2':anc2['proof_success'],
                'local_l4':local4['proof_success'],'ancestor_l4':anc4['proof_success'],
                'gain_l2':anc2['proof_success']-local2['proof_success'],
                'gain_l4':anc4['proof_success']-local4['proof_success'],
                'pass':ok if qualify else None}
        probes=_load_cone(eval_dir)
        if not all(p['irrelevant_perturbation_max']<=1e-5 for p in probes):
            raise ValueError('padded-rule stability gate failed')
        true_pair=[p for p in probes if p['config']=='ancestor_l2' and p['outside_local_cone']]
        local_pair=[p for p in probes if p['config']=='local_l2' and p['outside_local_cone']]
        mechanism=(len(true_pair)>=2 and all(p['stop_difference']>1e-4 for p in true_pair)
              and all(p['stop_difference']<=1e-6 for p in local_pair))
        verdict='inconclusive' if not qualify else 'pass' if all(x['pass'] for x in arms.values()) and mechanism else 'fail'
        if verdict=='fail':any_fail=True
        all_results[str(seed)]={'eligible':qualify,'verdict':verdict,'heldout':arms,
            'mechanism_gate':mechanism,
            'fit':{k:v['dev'] for k,v in tr['fit'].items()},
            'causal_probes':{k:[r for r in probes if r['config']==k] for k in ex.CONFIGS}}
    overall='fail' if any_fail else 'pass' if all(x['verdict']=='pass' for x in all_results.values()) else 'inconclusive'
    result={'experiment':'AF-19','scientific_verdict':overall,'engineering_pass':True,
        'seed_results':all_results,'verification':verified,'source_hashes':ex.hashes(),
        'protocol_sha256':ex.sha(ex.HERE/'PROTOCOL.json'),'manifest_sha256':ex.check_manifest()}
    ex.write(out,result);return result

def _load_cone(folder):
    return [json.loads(l) for l in (folder/'causal_cone.jsonl').read_text().splitlines()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    print(json.dumps(aggregate(a.root,a.out)))
if __name__=='__main__':main()
