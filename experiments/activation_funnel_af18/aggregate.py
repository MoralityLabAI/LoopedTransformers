"""Source-bound AF-18 scientific gate and cross-seed engineering audit."""
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
            raise ValueError('AF18 independent seed verification failed')
        if scores['train_sha256']!=ex.sha(train/'train_receipt.json') or a['evaluation_sha256']!=ex.sha(eval_dir/'evaluation.json'):
            raise ValueError('AF18 aggregate receipt mismatch')
        for k in verified:verified[k]+=a[k]
        qualify=all(tr['fit'][c][part]['proof_success']>=.65 for c in ('l2_cap16','l6_cap16','l8_cap16')
                    for part in ('train','dev'))
        arms={}
        for split in ('held_depth4','held_depth5'):
            x=scores['scores'];base=x['l2_cap16'][split];deep=x['l6_cap16'][split];top=x['l8_cap16'][split]
            cap10=x['l2_cap10'][split]
            keep=(deep['proof_success']>=base['proof_success']+.05 and
                  top['proof_success']>=base['proof_success']+.05 and
                  deep['proof_success']>=.70 and top['proof_success']>=.70 and
                  deep['false_accepted_proofs']==top['false_accepted_proofs']==0 and
                  deep['early_stop_positive']<=base['early_stop_positive']+.02 and
                  top['early_stop_positive']<=base['early_stop_positive']+.02)
            arms[split]={'two_loop_cap10':cap10['proof_success'],
                'two_loop_cap16':base['proof_success'],
                'four_loop_cap16':x['l4_cap16'][split]['proof_success'],
                'six_loop_cap16':deep['proof_success'],
                'eight_loop_cap16':top['proof_success'],
                'six_minus_two':deep['proof_success']-base['proof_success'],
                'eight_minus_two':top['proof_success']-base['proof_success'],
                'pass':keep if qualify else None}
        verdict='inconclusive' if not qualify else 'pass' if all(x['pass'] for x in arms.values()) else 'fail'
        if verdict=='fail':any_fail=True
        all_results[str(seed)]={'eligible':qualify,'verdict':verdict,'heldout':arms,
            'fit':{k:v['dev'] for k,v in tr['fit'].items()},
            'cone':{k:[r for r in _load_cone(eval_dir) if r['config']==k]
                for k in ex.CONFIGS}}
    overall='fail' if any_fail else 'pass' if all(x['verdict']=='pass' for x in all_results.values()) else 'inconclusive'
    result={'experiment':'AF-18','scientific_verdict':overall,'engineering_pass':True,
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
