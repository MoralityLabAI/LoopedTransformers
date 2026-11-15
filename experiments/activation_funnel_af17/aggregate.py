"""AF-17 frozen scientific gate; separately stores engineering verification."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from . import experiment as ex


def aggregate(root,out):
    root=Path(root);all_results={};total={'episodes':0,'forward_replays':0,'independent_cost_label_checks':0}
    any_failed=False
    for seed in ex.SEEDS:
        tr=root/f's{seed}/training/train_receipt.json'
        ev=root/f's{seed}/evaluation/evaluation.json'
        au=root/f's{seed}/evaluation/independent_audit.json'
        if not all(p.is_file() for p in (tr,ev,au)):raise FileNotFoundError('missing required AF17 seed receipt')
        t=json.loads(tr.read_text());e=json.loads(ev.read_text());a=json.loads(au.read_text())
        if not a['passed'] or a['seed']!=seed or a['train_sha256']!=ex.sha(tr) or a['evaluation_sha256']!=ex.sha(ev):
            raise ValueError('AF17 independent receipt mismatch')
        eligible=all(t['fit'][arm][split]['proof_success']>=(.75 if split=='train' else .65)
          for arm in ('stop','bellman') for split in ('train','dev'))
        results={};fail=False
        for split in ('held_depth4','held_depth5'):
            bel=e['scores']['bellman'][split];stop=e['scores']['stop'][split]
            extra=e['scores']['imitation_extra'][split];fixed=e['scores']['fixed'][split]
            delta=bel['proof_success']-stop['proof_success']
            delta_extra=bel['proof_success']-extra['proof_success']
            passed=(delta>=.05-1e-12 and delta_extra>=.02-1e-12 and
                    bel['proof_success']>=.75 and bel['proof_success']>=fixed['proof_success']-.03 and
                    bel['early_stop_positive']<=.05 and bel['false_accepted']==0)
            if eligible and not passed:fail=True
            results[split]={'bellman':bel['proof_success'],'stop':stop['proof_success'],
               'imitation':e['scores']['imitation'][split]['proof_success'],
               'imitation_extra':extra['proof_success'],'fixed':fixed['proof_success'],
               'gain_over_stop':delta,'gain_over_extra':delta_extra,
               'bellman_early_stop':bel['early_stop_positive'],'passed':passed if eligible else None}
        verdict='inconclusive' if not eligible else 'fail' if fail else 'pass'
        if verdict=='fail':any_failed=True
        all_results[str(seed)]={'eligible':eligible,'verdict':verdict,'heldout':results,'fit':t['fit']}
        for key in total:total[key]+=a[key]
    overall='fail' if any_failed else 'pass' if all(x['verdict']=='pass' for x in all_results.values()) else 'inconclusive'
    result={'experiment':'AF-17','scientific_verdict':overall,'engineering_pass':True,
          'seed_results':all_results,'counts':total,'protocol_sha256':ex.sha(ex.HERE/'PROTOCOL.json'),
          'manifest_sha256':ex.check_manifest(),'source_hashes':ex.hashes()}
    ex.write(out,result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(aggregate(a.root,a.out)))
if __name__=='__main__':main()
