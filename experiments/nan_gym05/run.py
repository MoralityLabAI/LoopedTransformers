"""Offline model worker: public prompts only, no corpus readers or host labels."""
import argparse
from collections import OrderedDict
import json
import os
import shutil
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from experiments.nan_bpd01.backend import load_backend
from experiments.nan_comp02.head import OperatorHead
from experiments.nan_comp03.prompt import aliases
from experiments.nan_diag04.telemetry import Telemetry
from mechinterp.dark_fiber_gyms.arg03_nanbeige import assert_oracle_isolated
from .common import read,rows,write,sha,prompt_ids
from .observe import TowerObserver
from .decode import generate
from .linear import extract

def filename(root):return 'root_'+root.replace(':','_')

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('bf16','nf4'),required=True)
    p.add_argument('--compactification-root',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--runs',required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args();start=time.perf_counter();root=Path(__file__).parent
    cfg=read(root/'config.json');seal=read(root/'SOURCE_MANIFEST.json');runs=Path(a.runs);out=runs/a.backend;out.mkdir(parents=True,exist_ok=True)
    for name,h in seal['critical_files_sha256'].items():
        if sha(root/name)!=h:raise ValueError('Registered critical source changed: '+name)
    for name,h in seal['model_dependencies_sha256'].items():
        if sha(root.parents[1]/name)!=h:raise ValueError('Registered model dependency changed: '+name)
    if sha(runs/'cohort/public.jsonl')!=seal['combined_public_sha256']:raise ValueError('Public prompt cohort changed')
    if sha(Path(a.snapshot)/'tokenizer_config.json')!=seal['tokenizer_config_sha256']:raise ValueError('Tokenizer changed')
    if sha(runs/'engineering/gpu_fixture.json')!=seal['gpu_fixture_sha256']:raise ValueError('Engineering fixture changed')
    public=rows(runs/'cohort/public.jsonl');grouped=OrderedDict()
    for r in public:
        assert set(r)=={'id','root','split','suite','prompt','schema','publication'}
        grouped.setdefault(r['root'],[]).append(r)
    completed=[];prior=None
    if (out/'receipt.json').exists():
        if not a.resume:raise FileExistsError('A prior run must be explicitly resumed')
        prior=read(out/'receipt.json')
        assert not prior.get('attempts') and prior['source_manifest_sha256']==sha(root/'SOURCE_MANIFEST.json')
        completed=list(prior['completed_roots']);write(out/'initial_execution_receipt.json',prior)
        for key in completed:
            saved=read(out/(filename(key)+'.json'))
            assert saved['root']==key and sha(out/saved['state_file'])==saved['state_file_sha256']
        incomplete=out/'initial_incomplete_artifacts'
        for key in grouped:
            if key in completed:continue
            for suffix in ('.json','_states.npz'):
                artifact=out/(filename(key)+suffix)
                if artifact.exists():
                    incomplete.mkdir(exist_ok=True);shutil.copyfile(artifact,incomplete/artifact.name)
        linear=out/'linear_baselines'
        if linear.exists() and not (linear/'manifest.json').exists():
            shutil.copytree(linear,out/'linear_baselines_initial_attempt')
    elif a.resume:raise FileNotFoundError('No prior receipt to resume')
    torch.manual_seed(cfg['source_seed']);torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']);torch.cuda.reset_peak_memory_stats()
    meta={'backend':a.backend,'checkpoint_revision':cfg['checkpoint_revision'],'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),
        'public_sha256':sha(runs/'cohort/public.jsonl'),'pid':os.getpid(),'torch':torch.__version__,'commands':vars(a),
        'public_prompts_only':True,'KV_cache':False,'native_loops':2,'state_site':'residual_after_layer',
        'BPD_transfer_attempted':False,'sealed_test_accessed':False,'placement':{'embeddings':'CPU BF16','head':'CPU BF16 chunks','decoder':'one BF16 resident plus stage' if a.backend=='bf16' else 'CUDA NF4 double quantization, BF16 compute'}}
    status='running';failure=None
    with Telemetry(out/('memory_samples_resume.json' if prior else 'memory_samples.json')):
        try:
            assert_oracle_isolated();print('Loading verified '+a.backend+' for NAN-GYM-05',flush=True)
            loop,oldhead,tok,entry=load_backend(a.compactification_root,a.snapshot,a.backend,chunk_rows=cfg['vocabulary_chunk_rows'])
            head=OperatorHead(oldhead.weight,aliases(tok),chunk_rows=cfg['vocabulary_chunk_rows']);observer=TowerObserver(loop,cfg['layers_zero_based'])
            meta['eos_token_id']=tok.tok.eos_token_id;meta['aliases']=aliases(tok);write(out/'manifest.json',meta)
            if not (out/'linear_baselines/manifest.json').exists():extract(loop,out/'linear_baselines',a.backend,cfg)
            for key,tasks in grouped.items():
                if key in completed:continue
                if time.perf_counter()-start>cfg['maximum_backend_seconds']:status='time_budget_reached';break
                mark=time.perf_counter();print(a.backend+' root '+key+' starting',flush=True)
                seq={r['id']:prompt_ids(tok,a.snapshot,r['prompt']) for r in tasks}
                assert len(seq)<=cfg['maximum_group_contexts'] and max(map(len,seq.values()))<=cfg['maximum_prompt_tokens']
                records,observed=generate(observer,head,tok.tok,tasks,seq,cfg);arrays={};index=[]
                for i,((task_id,phase,layer,site),tensor) in enumerate(observed.items()):
                    array_key='v'+str(i);arrays[array_key]=tensor.numpy();index.append({'task_id':task_id,'loop':phase,'layer_zero_based':layer,'site':site,'array_key':array_key})
                state_file=out/(filename(key)+'_states.npz');np.savez_compressed(state_file,**arrays)
                write(out/(filename(key)+'.json'),{'root':key,'split':tasks[0]['split'],'suite':tasks[0]['suite'],'publication':tasks[0]['publication'],
                    'records':records,'observations':index,'state_file':state_file.name,'state_file_sha256':sha(state_file)})
                completed.append(key);assert_oracle_isolated();torch.cuda.empty_cache()
                print(a.backend+' root '+key+' saved in '+str(round(time.perf_counter()-mark,1))+'s',flush=True)
            else:status='completed'
            observer.close()
        except Exception as e:
            status='failed';failure={'type':type(e).__name__,'message':str(e)};traceback.print_exc()
        finally:
            assert_oracle_isolated();torch.cuda.synchronize()
            receipt=meta|{'status':status,'failure':failure,'completed_roots':completed,'elapsed_seconds':round(time.perf_counter()-start,3),
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
                'head_remained_cpu':bool('head' in locals() and head.weight.device.type=='cpu')}
            if prior:
                receipt['attempts']=[prior,dict(receipt)];receipt['elapsed_seconds']+=prior['elapsed_seconds']
                for k in ('peak_cuda_allocated_bytes','peak_cuda_reserved_bytes'):receipt[k]=max(prior[k],receipt[k])
            write(out/'receipt.json',receipt);print(json.dumps({k:receipt[k] for k in ('status','failure','elapsed_seconds','completed_roots')}),flush=True)
    if status!='completed':raise SystemExit(2)

if __name__=='__main__':main()
