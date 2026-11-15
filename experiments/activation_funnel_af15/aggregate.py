"""AF-15 independent registered verdict over complete source-bound seed roster."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from . import experiment as ex


def aggregate(root,out):
    root=Path(root);results={};episodes=0;replays=0
    for seed in ex.SEEDS:
        tr=root/f's{seed}/training/train_receipt.json'
        ev=root/f's{seed}/evaluation/evaluation.json'
        audit=root/f's{seed}/evaluation/independent_audit.json'
        if not all(f.is_file() for f in (tr,ev,audit)):raise FileNotFoundError('AF-15 missing one seed artifact')
        t=json.loads(tr.read_text());e=json.loads(ev.read_text());a=json.loads(audit.read_text())
        if not a['passed'] or a['seed']!=seed or a['training_sha256']!=ex.sha(tr) or a['evaluation_sha256']!=ex.sha(ev):
            raise ValueError('AF-15 audit provenance mismatch')
        eligible=all(t['fit'][arm][split]['proof_success']>= (.80 if split=='train' else .75)
                     for arm in ex.ARMS for split in ('train','dev'))
        families={};failure=False
        for split in ('fresh_depth4','fresh_depth5'):
            b=e['scores']['imitation'][split];c=e['scores']['frontier'][split];fixed=e['scores']['fixed'][split]
            gain=c['proof_success']-b['proof_success']
            passed=(gain>=.05-1e-12 and c['proof_success']>=.80 and
                    c['false_accepted']==0 and c['proof_success']>=fixed['proof_success']-.03)
            if eligible and not passed:failure=True
            families[split]={'imitation':b['proof_success'],'frontier':c['proof_success'],
              'fixed':fixed['proof_success'],'gain':gain,
              'false_accepted':c['false_accepted'],'passed':passed if eligible else None}
        result='inconclusive' if not eligible else ('fail' if failure else 'pass')
        results[str(seed)]={'eligible':eligible,'verdict':result,'families':families,
            'fit':t['fit'],'source_sha256':ex.sha(tr)}
        episodes+=a['episodes'];replays+=a['forward_replays']
    verdict='fail' if any(d['verdict']=='fail' for d in results.values()) else (
        'pass' if all(d['verdict']=='pass' for d in results.values()) else 'inconclusive')
    overall={'experiment':'AF-15','registered_verdict':verdict,'engineering_pass':True,
        'seed_results':results,'verified_rows':episodes,'sampled_forward_replays':replays,
        'source_hashes':ex.hashes(),'protocol_sha256':ex.sha(ex.PROTOCOL),
        'fresh_manifest_sha256':ex.check_manifest()}
    ex.write(out,overall);return overall


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(aggregate(a.root,a.out)))
if __name__=='__main__':main()
