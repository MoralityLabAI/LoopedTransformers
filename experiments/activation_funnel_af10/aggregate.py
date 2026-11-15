"""AF-10 deterministic two-seed source-bound independent audit aggregation."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from .diagnostic import ARTIFACTS,ARMS,SPLITS,FROZEN_SOURCE


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(root:Path):
    result={'experiment':'AF-10','scope':'exploratory_frozen_model_mechanism',
            'original_scientific_source':FROZEN_SOURCE,'all_integrity_checks_pass':True,
            'seeds':{},'original_models':0,'rows':0,'checkpoint_replay_calls':0,
            'structural_noninterference_errors':0,'primary_scientific_claim':'none'}
    for seed in (17,29):
        folder=root/f'af10-seed-{seed}'
        audit=json.loads((folder/'independent_audit.json').read_text())
        summary=json.loads((folder/'summary.json').read_text())
        if (audit['experiment'],audit['seed'],audit['verified'])!=('AF-10',seed,True):
            raise ValueError('incomplete independently checked seed')
        if audit['summary_sha256']!=sha(folder/'summary.json') or \
            audit['retained_rows_sha256']!=sha(folder/'cut_replay.jsonl') or \
            summary['row_sha256']!=audit['retained_rows_sha256']:
            raise ValueError('tampered source-bound audit roster')
        if audit['source_sha256']!=ARTIFACTS[seed]['sha256'] or \
           summary['source']['original_source_commit']!=FROZEN_SOURCE:
            raise ValueError('wrong original AF-09 frozen-model identity')
        if audit['model_groups']!=len(ARMS)*len(SPLITS) or audit['rows']!=4320:
            raise ValueError('incomplete original arm and input donor roster')
        if audit['checkpoints_replayed']!=90 or audit['independent_affine_max_error']>2e-5:
            raise ValueError('incomplete independent native/NumPy replay')
        for arm in ARMS:
            if set(summary['models'][arm])!=set(SPLITS):raise ValueError('heldout family missing')
            for split,s in summary['models'][arm].items():
                if s['cut_noninterference_error_max']!=0 or \
                  s['cut_protected_correct']!=s['prefix_protected_correct']:
                    raise ValueError('structural causal cut not invariant')
        result['seeds'][str(seed)]={'artifact_sha256':audit['source_sha256'],
                                    'summary_sha256':audit['summary_sha256'],
                                    'row_sha256':audit['retained_rows_sha256'],
                                    'independent_affine_max_error':audit['independent_affine_max_error'],
                                    'models':{arm:{split:{key:s[key] for key in (
                                        'pairs','prefix_protected_correct_rate','ordinary_protected_correct_rate',
                                        'cut_protected_correct_rate','ordinary_composed_goal_correct_rate',
                                        'cut_composed_goal_correct_rate','prefix_pair_fraction',
                                        'cut_goal_accuracy_conditional_on_correct_prefix_pair')}
                                        for split,s in summary['models'][arm].items()}
                                    for arm in ARMS}}
        result['rows']+=audit['rows'];result['original_models']+=len(ARMS)
        result['checkpoint_replay_calls']+=audit['checkpoints_replayed']
    assert (result['rows'],result['original_models'],result['checkpoint_replay_calls'])==(8640,6,180)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=aggregate(a.root)
    a.out.write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('experiment','all_integrity_checks_pass','rows',
                                     'original_models','checkpoint_replay_calls')}))
if __name__=='__main__':main()
