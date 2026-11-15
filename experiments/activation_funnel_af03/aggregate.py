"""AF-03 cross-seed receipt validation and preregistered architecture-level gate."""
import argparse
import json
from pathlib import Path

import hashlib

SEEDS=(17,29,43)
WIDTHS=(1,2)

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1 << 20),b''):
            h.update(chunk)
    return h.hexdigest()

def save_json(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def aggregate(root):
    root=Path(root)
    rows=[]
    for seed in SEEDS:
        for width in WIDTHS:
            dirs=list(root.glob(f'af03-s{seed}-k{width}-*'))
            if len(dirs)!=1:
                raise ValueError(f'missing or duplicate seed={seed} code width={width} artifact')
            folder=dirs[0]
            report_path=folder/'eval'/'report.json'
            train_path=folder/'train'/'train_receipt.json'
            report=json.loads(report_path.read_text())
            train=json.loads(train_path.read_text())
            if (report['experiment'],report['seed'],report['token_code_width'])!=('AF-03',seed,width):
                raise ValueError('report identity mismatch')
            if (train['scope'],train['seed'],train['token_code_width'])!=('primary',seed,width):
                raise ValueError('nonprimary training receipt')
            if report['train_receipt_sha256']!=file_hash(train_path):
                raise ValueError('report does not bind training receipt')
            if report['source_sha256']!=train['af02_source_sha256']:
                raise ValueError('report does not bind frozen AF-02 source')
            for filename,digest in report['file_hashes'].items():
                if file_hash(folder/'eval'/filename)!=digest:
                    raise ValueError('held-out predictions / intervention records changed')
            for arm,files in train['weights'].items():
                for ext, key in (('pt','torch'),('npz','named_arrays')):
                    if file_hash(folder/'train'/f'{arm}.{ext}')!=files[key]['sha256']:
                        raise ValueError('frozen model changed')
            rows.append({'seed':seed,'width':width,'carry':5*width,
                         'gate':report['primary_gate'],
                         'eligible':report['gate_eligible'],
                         'native_causal_cell_grad_l2':train['training']['causal']['native_cell_first_step_grad_l2'],
                         'checks':report['checks'],
                         'scores':{arm:{family:report['arms'][arm][family]
                                         for family in ('depth_context','delay4','delay5','delay8')}
                                   for arm in ('task','status','causal')}})
    primary=[r for r in rows if r['width']==2]
    if all(r['gate']=='pass' for r in primary):
        overall='pass'
    elif any(r['gate']=='fail' for r in primary):
        overall='fail'
    else:
        overall='inconclusive'
    return {'experiment':'AF-03','aggregate_primary_10_coordinate_gate':overall,
            'primary_width':2,'stress_width':1,
            'coverage':{'models':18,'seeds':list(SEEDS),'widths':list(WIDTHS),
                        'arms':['task','status','causal'],'all_matrix_receipts_verified':True},
            'runs':rows,
            'caveat':'Synthetic delay5 and delay8 are extra tensor-only waits and carry no fresh original-host Lean certificate.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    result=aggregate(args.root)
    save_json(args.out,result)
    print(json.dumps({'aggregate_primary_gate':result['aggregate_primary_10_coordinate_gate'],
                      'seeds':[(r['seed'],r['width'],r['gate']) for r in result['runs']]}),flush=True)


if __name__=='__main__':main()
