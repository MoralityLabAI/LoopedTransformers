"""AF-22 all-seed preregistered science verdict and independent audit merge."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from . import experiment as ex


def aggregate(root,out):
    root=Path(root);seeds={};checked={'proof_rows':0,'planner_nodes_checked':0,
        'checkpoint_forward_replays':0,'false_accepted_proofs':0}
    for seed in ex.SEEDS:
        folder=root/f's{seed}'
        training=folder/'training';evaluation=folder/'evaluation'
        tr=json.loads((training/'train_receipt.json').read_text())
        ev=json.loads((evaluation/'evaluation.json').read_text())
        audit=json.loads((folder/'independent_audit.json').read_text())
        if not audit['verified'] or audit['source_hashes']!=ex.hashes() or \
           audit['evaluation_sha256']!=ex.sha(evaluation/'evaluation.json') or \
           audit['training_sha256']!=ex.sha(training/'train_receipt.json') or \
           tr['source_hashes']!=ex.hashes() or ev['source_hashes']!=ex.hashes():
            raise ValueError('AF22 seed archive failed independent source-binding check')
        for key in checked:checked[key]+=audit[key]
        eligible=all(tr['fit'][sp]['proof_success']>=.65 for sp in ('train','dev'))
        depth={};passed=[]
        for sp in ('held_depth4','held_depth5'):
            rec=ev['scores'][sp];direct=rec['direct'];vi1=rec['vi1'];vi2=rec['vi2']
            gate=(vi2['proof_success']>=direct['proof_success']+.05-1e-10 and
                  vi2['proof_success']>=.70 and
                  vi2['early_stop_positive']<=direct['early_stop_positive']+.02+1e-10 and
                  vi2['proof_success']>=vi1['proof_success']-.03-1e-10 and
                  vi2['false_accepted_proofs']==0)
            passed.append(gate)
            depth[sp]={'direct':direct,'vi1':vi1,'vi2':vi2,
                       'gain':vi2['proof_success']-direct['proof_success'],
                       'registered_pass':gate if eligible else None}
        verdict='inconclusive' if not eligible else 'pass' if all(passed) else 'fail'
        seeds[str(seed)]={'eligible':eligible,'verdict':verdict,'depth':depth,
            'train_dev':tr['fit'],
            'held_layout':ev['scores']['held_layout']}
    overall=('fail' if any(x['verdict']=='fail' for x in seeds.values())
             else 'pass' if all(x['verdict']=='pass' for x in seeds.values())
             else 'inconclusive')
    result={'experiment':'AF-22','scientific_verdict':overall,'engineering_pass':True,
            'seed_results':seeds,'verification':checked,
            'source_hashes':ex.hashes(),'manifest_sha256':ex.check_manifest(),
            'protocol_sha256':ex.sha(ex.HERE/'PROTOCOL.json')}
    ex.write(out,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(aggregate(a.root,a.out)))
if __name__=='__main__':main()
