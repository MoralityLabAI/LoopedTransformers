"""COE-09 strictly post-registration causal replay of a SHA-pinned pretrained TRM.

Runs a selected direction from COE-07's fit-only probe on the held-out Sudoku
boards and compares matched-radius rotations along seeded random directions.
The probe direction, visit=4, dose=0.05 rad, both signs, and random baselines
are fixed before heldout inference. Row-space/parity audits are mandatory.
Native weights must be downloaded and SHA-256 checked by COE-07's loader.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
try:
    from .native_label_replay import (SOURCE_MODEL_SHA,SOURCE_SCHEMA,FIT_STRATA,TEST_STRATA,REPS,load_native)
    from .causal_geometry import row_basis,rotate_vector,prepare_direction,geometry_certificate
except ImportError:
    from native_label_replay import (SOURCE_MODEL_SHA,SOURCE_SCHEMA,FIT_STRATA,TEST_STRATA,REPS,load_native)
    from causal_geometry import row_basis,rotate_vector,prepare_direction,geometry_certificate

HEAD_NAME='cyclic_gain'
VISIT=4
FINAL=8
ANGLE=0.05
CONTROL_SEEDS=(303,317)
SECTORS=('dark','visible','scratch')


def end_logits(model,zh,zl,emb,ar,target):
    for _ in range(FINAL-VISIT):zh,zl=model.act_step(zh,zl,emb)
    return model.lm_head(zh[ar,target]).float()[:,2:11]


def summary(logits,truth):
    pred=logits.argmax(-1)
    true=logits[torch.arange(len(truth),device=logits.device),truth]
    other=logits.clone();other[torch.arange(len(truth),device=logits.device),truth]=float('-inf')
    margins=true-other.max(-1).values
    return {'correct':[int(x) for x in (pred==truth).cpu().tolist()],
            'predictions':[int(x) for x in pred.cpu().tolist()],
            'margins':[float(x) for x in margins.cpu().tolist()]}


def compare(clean,candidate):
    a=clean['correct'];b=candidate['correct']
    return {'clean_correct':sum(a),'intervention_correct':sum(b),
            'clean_to_wrong':sum(x==1 and y==0 for x,y in zip(a,b)),
            'wrong_to_correct':sum(x==0 and y==1 for x,y in zip(a,b)),
            'prediction_changes':sum(x!=y for x,y in zip(clean['predictions'],candidate['predictions'])),
            'mean_margin_change':sum(y-x for x,y in zip(clean['margins'],candidate['margins']))/len(a)}


@torch.inference_mode()
def experiment(model, manifest,receipt):
    from mechinterp.loopscope02.scripts import trm_sudoku as TS
    rows=[x for k in TEST_STRATA for x in manifest['strata'][k]]
    assert len(rows)==18
    device=next(model.parameters()).device
    inputs=TS.encode_questions([x['question'] for x in rows],device)
    ar=torch.arange(len(rows),device=device)
    index=torch.tensor([int(x['target_idx']) for x in rows],device=device)+model.puzzle_emb_len
    truth=torch.tensor([int(x['digit'])-1 for x in rows],device=device,dtype=torch.long)
    emb=model.encode_inputs(inputs)
    zh,zl=model.init_states(len(rows))
    for _ in range(VISIT):zh,zl=model.act_step(zh,zl,emb)
    clean_logits=end_logits(model,zh.clone(),zl.clone(),emb,ar,index)
    clean=summary(clean_logits,truth)
    basis=row_basis(model.lm_head.weight.detach().float())
    out={}
    w=model.lm_head.weight.detach().float()
    for sector in SECTORS:
        h=receipt['representations'][sector][HEAD_NAME]
        raw=torch.tensor(h['direction'],device=device,dtype=torch.float32)
        selected=prepare_direction(raw,w,sector)
        runs={}
        for kind in ('selected','random303','random317'):
            if kind=='selected': vec=selected
            else:
                generator=torch.Generator(device='cpu').manual_seed(int(kind[6:]))
                vec=torch.randn((len(rows),w.shape[1]),generator=generator).to(device)
            for sign in (-1,1):
                hz=zh.clone();lz=zl.clone()
                audits=[]
                for i in range(len(rows)):
                    state=(zl if sector=='scratch' else zh)[i,index[i],:].float()
                    d=vec if kind=='selected' else vec[i]
                    moved=rotate_vector(state,d,basis if sector in ('visible','dark') else None,
                                        sector=sector,angle=sign*ANGLE)
                    cert=geometry_certificate(state,moved,w,sector)
                    if cert['relative_norm_change']>2e-5:raise ValueError('off-sphere intervention')
                    if sector=='dark' and cert['max_head_logit_change']>2e-4:
                        raise ValueError('dark-fiber intervention altered visible logits')
                    audits.append(cert)
                    if sector=='scratch':lz[i,index[i]]=moved.to(lz.dtype)
                    else:hz[i,index[i]]=moved.to(hz.dtype)
                after=summary(end_logits(model,hz,lz,emb,ar,index),truth)
                runs[f'{kind}_{sign:+d}']={**compare(clean,after),
                    'max_relative_norm_error':max(c['relative_norm_change'] for c in audits),
                    'max_head_logit_change':(max(c['max_head_logit_change'] for c in audits)
                                              if sector!='scratch' else None),
                    'mean_displacement_l2':sum(c['displacement_l2'] for c in audits)/len(audits)}
        out[sector]=runs
    return {'schema':'COE-09-native-targeted-causal-v1',
            'site':'evaluator-designated target cell','dose_radians':ANGLE,
            'injection_after_visit':VISIT,'final_visit':FINAL,
            'direction_head':HEAD_NAME,'null_seeds':list(CONTROL_SEEDS),
            'fit_strata':list(FIT_STRATA),'heldout_strata':list(TEST_STRATA),
            'n_heldout_boards':len(rows),'clean':clean,'interventions':out,
            'scope':'predefined causal perturbations, fixed selected readouts, no heldout head fitting'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',required=True,type=Path)
    p.add_argument('--activations',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--threads',type=int,default=4)
    opt=p.parse_args()
    if not 1<=opt.threads<=16:raise ValueError('invalid threads')
    if opt.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    torch.set_num_threads(opt.threads)
    receipt=json.loads(opt.receipt.read_text(encoding='utf8'))
    _arrays=load_native(opt.activations,receipt)
    from mechinterp.phase_gauge07.score import select_manifest
    from mechinterp.loopscope02.scripts import trm_sudoku as TS
    repo_root=Path(__file__).resolve().parents[2]
    bundle=json.loads((repo_root/'mechinterp/loopscope02/results/benchmark_bundle.json').read_text())
    manifest=select_manifest(bundle)
    if manifest['manifest_sha256']!=receipt['source_benchmark_sha256']:
        raise ValueError('source native manifest mismatch')
    model,sha=TS.load_trm_checkpoint('gaoxin492',opt.device,dtype=torch.float32)
    if sha!=SOURCE_MODEL_SHA:raise ValueError('checkpoint pin mismatch')
    r=experiment(model,manifest,receipt)
    r['source_checkpoint_sha256']=sha
    r['source_native_receipt_sha256']=hashlib.sha256(opt.receipt.read_bytes()).hexdigest()
    r['source_native_activation_sha256']=hashlib.sha256(opt.activations.read_bytes()).hexdigest()
    opt.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'schema':r['schema'],'clean_correct':sum(r['clean']['correct']),
                      'effects':{k:{arm:val['intervention_correct'] for arm,val in v.items()}
                                 for k,v in r['interventions'].items()}},indent=2))
if __name__=='__main__':main()
