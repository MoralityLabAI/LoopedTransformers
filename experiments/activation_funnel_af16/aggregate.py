"""Frozen AF-16 primary gate over every independently checked seed and family."""
from __future__ import annotations
from pathlib import Path
import argparse,json
from . import experiment as ex

def aggregate(root,out):
    root=Path(root);results={};counts={'episodes':0,'forward_replays':0};failed=False
    for seed in ex.SEEDS:
        tr=root/f's{seed}/training/train_receipt.json'
        ev=root/f's{seed}/evaluation/evaluation.json'
        ar=root/f's{seed}/evaluation/independent_audit.json'
        if not all(p.is_file() for p in (tr,ev,ar)):raise FileNotFoundError('missing AF16 required seed receipt')
        t=json.loads(tr.read_text());e=json.loads(ev.read_text());a=json.loads(ar.read_text())
        if not a['passed'] or a['seed']!=seed or a['training_sha256']!=ex.sha(tr) or a['evaluation_sha256']!=ex.sha(ev):
            raise ValueError('AF16 independent result receipt invalid')
        eligible=all(t['fit'][arm][split]['proof_success']>=(.75 if split=='train' else .65)
                     for arm in ('imitation','frontier') for split in ('train','dev'))
        scores={};seed_fail=False
        for split in ('novel_depth4','novel_depth5'):
            b=e['scores']['imitation'][split];f=e['scores']['frontier'][split]
            extra=e['scores']['imitation_extra'][split];fixed=e['scores']['fixed'][split]
            gain=f['proof_success']-b['proof_success']
            gain_over_more=f['proof_success']-extra['proof_success']
            passed=(gain>=.05-1e-12 and gain_over_more>=.02-1e-12 and
                  f['proof_success']>=.70 and f['false_accepted']==0 and
                  f['proof_success']>=fixed['proof_success']-.03)
            if eligible and not passed:seed_fail=True
            scores[split]={'imitation':b['proof_success'],'frontier':f['proof_success'],
                   'imitation_extra':extra['proof_success'],'fixed':fixed['proof_success'],
                   'gain':gain,'gain_over_more':gain_over_more,'passed':passed if eligible else None}
        verdict='inconclusive' if not eligible else ('fail' if seed_fail else 'pass')
        results[str(seed)]={'eligible':eligible,'verdict':verdict,'scores':scores,'fit':t['fit']}
        if verdict=='fail':failed=True
        counts['episodes']+=a['episodes'];counts['forward_replays']+=a['forward_replays']
    overall='fail' if failed else ('pass' if all(s['verdict']=='pass' for s in results.values()) else 'inconclusive')
    result={'experiment':'AF-16','scientific_verdict':overall,'engineering_pass':True,
          'seed_results':results,'counts':counts,'source_hashes':ex.hashes(),
          'protocol_sha256':ex.sha(ex.HERE/'PROTOCOL.json'),
          'manifest_sha256':ex.check_manifest()}
    ex.write(out,result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();print(json.dumps(aggregate(a.root,a.out)))
if __name__=='__main__':main()
