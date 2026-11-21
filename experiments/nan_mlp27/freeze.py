"""Freeze the complete conditional scope before any new pretrained forward."""
from datetime import datetime,timezone
from pathlib import Path
import ast
from .common import ROOT,RUNS,load_sources,intervention_rows,read,sha,write

def main():
    assert not RUNS.exists()
    from experiments.nan_full26.common import verify as original
    parent=original();public,pins=load_sources();cfg=read(ROOT/'config.json');old=read('experiments/nan_full26/config.json')
    for key,value in old.items():
        if key not in ('experiment','capacity_shapes'):assert cfg[key]==value,key
    assert len(public)==1549 and len(intervention_rows(public))==72
    bindings=read('experiments/nan_full26/tokenizer_receipt.json')['contexts']
    assert max(x['tokens'] for x in bindings.values())+256<=9216
    assert max(bindings[x['id']]['tokens'] for x in intervention_rows(public))+256<=4608
    engineering=read(ROOT/'engineering_receipt.json');assert engineering['status']=='passed' and engineering['pretrained_model_forwards']==engineering['GPU_allocations']==0
    assert engineering['closed_synthetic_host_episodes']==22 and engineering['geometry_pairs']==12
    for name,h in engineering['source_sha256'].items():assert sha(ROOT/name)==h,name
    cpu=read('reports/nanbeige_bpd_gym/nan_mlp27_cpu_engineering.json');assert cpu['status']=='passed' and len(cpu['cases'])==12
    assert cpu['adapter_sha256']==sha('mechinterp/nanbeige_bpd_gym/mlp_workspace27.py')
    assert cpu['fixture_sha256']==sha('mechinterp/nanbeige_bpd_gym/check_mlp_workspace27.py')
    files=dict(parent['files'])
    files['experiments/nan_full26/SOURCE_MANIFEST.json']=sha('experiments/nan_full26/SOURCE_MANIFEST.json')
    for p in ROOT.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='SOURCE_MANIFEST.json':
            if p.suffix in ('.py','.json','.md') or p.name=='.gitattributes':files[p.as_posix()]=sha(p)
    for name in ('mechinterp/nanbeige_bpd_gym/mlp_workspace27.py','mechinterp/nanbeige_bpd_gym/check_mlp_workspace27.py',
        'reports/nanbeige_bpd_gym/nan_mlp27_cpu_engineering.json','.github/workflows/nan-mlp27-cpu.yml'):
        files[name]=sha(name)
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text())
    write(ROOT/'SOURCE_MANIFEST.json',dict(experiment='NAN-MLP-27',frozen_UTC=datetime.now(timezone.utc).isoformat(),
        files=files,source_pins=pins,parent_FULL26_manifest_sha256=sha('experiments/nan_full26/SOURCE_MANIFEST.json'),
        baseline_contexts_per_backend=1549,AB_pairs_per_backend=216,distinct_root_ids=len({r['root'] for r in public}),
        new_pretrained_started=False,conditional_resource_qualification=True,task_score_selection=False,
        original_neural_replies_reused=0,original_workers_stopped=False,
        checkpoint_revision=cfg['checkpoint_revision'],adapter_commit=parent['adapter_commit'],
        BPD_commit=parent['BPD_commit'],EvalMath_commit=parent['EvalMath_commit'],
        exposure='Earlier development results and CTX21 NF4 capacity stop exposed; BF16 CTX21 running. No STORY23/FULL26/new MLP27 neural outcome is used to select tasks.',
        actual_GPU_capacity_unmeasured=True,formal_inferential_comparisons=0,sealed_test_accessed=False))
    print(dict(seal_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),bindings=len(files),baseline=1549,AB_pairs=216,pretrained_forwards=0))

if __name__=='__main__':main()
