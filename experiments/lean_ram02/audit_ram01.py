"""Independently verify frozen RAM-01 checkpoint hashes and cached dev scores."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import model_data as m
from train_local import split_data,write_json
from resources import ResourceGuard

def run(data,weights,out):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    train,dev,audit=split_data(data)
    expected={'fc12_seed17.pt':'55bc16117e045279ba0e723c171e45b4fe05b4c6eb8a1b9709dd1b59cc9987a2',
              'fc12_seed29.pt':'0e11d74cf1be7889caf824671099d052884dc3722211e773dd126d233671a805'}
    assert audit['original_fc12_checkpoint_hashes']==expected
    reports=list(weights.rglob('training_report.json'))
    assert len(reports)==1
    report=json.loads(reports[0].read_text(encoding='utf8'))
    root=reports[0].parent
    results=[]
    for phase in ('development','final'):
        for record in report[phase]:
            checkpoint=root/phase/record['checkpoint']
            sha=m.digest(checkpoint)
            assert sha==record['checkpoint_sha256']
            data=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert data['state_sha256']==m.STATE_SHA and data['effect_sha256']==m.EFFECT_SHA
            model=m.CandidatePolicy(m.Config(**data['config']))
            model.load_state_dict(data['state_dict']);model.eval()
            score=m.cached_metrics(model,dev) if phase=='development' else None
            if score is not None:
                assert score['top1_executable']==record['metrics']['top1_executable'],(record['name'],record['seed'])
            results.append(dict(phase=phase,name=record['name'],seed=record['seed'],checkpoint_sha256=sha,
                parameters=sum(p.numel() for p in model.parameters()),cached_development=score))
    assert len(results)==16
    receipt=dict(experiment='RAM02_RAM01_independent_artifact_audit',checkpoints=results,
                 train_data_audit=audit,all_checkpoint_hashes_verified=True,
                 all_eight_development_top1_counts_reproduced=True,
                 torch=str(torch.__version__),source_training_report_sha256=m.digest(reports[0]),
                 note='Read-only reproduction; no original checkpoint or compiler outcome is modified.')
    write_json(out,receipt)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    with ResourceGuard(a.out.with_suffix('.resources.jsonl'),torch):
        run(a.data,a.weights,a.out)
