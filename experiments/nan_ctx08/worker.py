"""One native model, resident unchanged BF16 head, source-only long inputs."""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from experiments.nan_bpd01.backend import load_backend
from experiments.nan_comp02.head import OperatorHead
from experiments.nan_comp03.prompt import aliases
from experiments.nan_diag04.telemetry import Telemetry
from experiments.nan_gym05.common import prompt_ids
from experiments.nan_gym05.observe import TowerObserver
from experiments.nan_fit07.io import read,rows,sha,write
from mechinterp.dark_fiber_gyms.arg03_nanbeige import assert_oracle_isolated
from mechinterp.nanbeige_bpd_gym.resident_head import mount,check_parity,retain_decoder_layers


class BudgetReached(RuntimeError):pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('nf4','bf16'),required=True);p.add_argument('--snapshot',required=True)
    p.add_argument('--compactification-root',required=True);p.add_argument('--runs',default='runs/nan_ctx08');a=p.parse_args()
    root=Path(__file__).parent;cfg=read(root/'config.json');seal=read(root/'SOURCE_MANIFEST.json')
    for name,h in seal['worker_files_sha256'].items():assert sha(root/name)==h,name
    for name,h in seal['model_dependencies_sha256'].items():assert sha(name)==h,name
    token_receipt=read(root/'fixtures/tokenizer_receipt.json')
    assert sha(Path(a.snapshot)/'tokenizer_config.json')==token_receipt['tokenizer_config_sha256']
    public=rows(root/'fixtures/public.jsonl');assert len(public)==291
    out=Path(a.runs)/a.backend;out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    torch.set_num_threads(1);torch.manual_seed(2026112008);torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction'])
    torch.cuda.reset_peak_memory_stats();status='failed';completed=[];levels=[];failure=None;parity=[];observer=None;peaks={'allocated':0,'reserved':0}
    def keep_peaks():
        peaks['allocated']=max(peaks['allocated'],torch.cuda.max_memory_allocated())
        peaks['reserved']=max(peaks['reserved'],torch.cuda.max_memory_reserved())
    def budget():
        if time.monotonic()-started>cfg['maximum_backend_seconds']:raise BudgetReached('Registered backend budget reached')
    with Telemetry(out/'memory_samples.json'):
        try:
            assert_oracle_isolated();loop,legacy,tok,_=load_backend(a.compactification_root,a.snapshot,a.backend,chunk_rows=cfg['scorer_chunk_rows'])
            if loop.stage_layer is not None:loop.stage_layer.requires_grad_(False).eval()
            if a.backend=='bf16':retain_decoder_layers(loop,cfg['BF16_resident_decoder_layers'])
            cpu_head=OperatorHead(legacy.weight,aliases(tok),chunk_rows=cfg['scorer_chunk_rows']);head=mount(loop,cpu_head);del legacy
            observer=TowerObserver(loop,cfg['layers_zero_based'])
            write(out/'manifest.json',{'backend':a.backend,'pid':os.getpid(),'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),
                'checkpoint_revision':cfg['checkpoint_revision'],'head_dtype':'BF16','head_quantized':False,
                'head_device':str(head.weight.device),'head_bytes':head.weight.numel()*head.weight.element_size(),
                'embedding_device':str(loop.base.embed_tokens.weight.device),'decoder_GPU_layers':loop.n_gpu_layers,
                'CPU_threads':torch.get_num_threads(),'head_transfer_per_score':'GPU-to-GPU chunks; no CPU vocabulary-weight transfer',
                'host_gold_opened':False,'sealed_test_accessed':False})
            indices={r['id']:i for i,r in enumerate(public)}
            def execute(row):
                nonlocal cpu_head
                budget();ids=prompt_ids(tok,a.snapshot,row['prompt']);bound=token_receipt['contexts'][row['id']]
                assert len(ids)==bound['tokens'] and hashlib.sha256(json.dumps(ids).encode()).hexdigest()==bound['initial_prompt_ids_sha256']
                assert len(ids)<=row['context_budget'];keep_peaks();torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
                with torch.inference_mode():
                    two,observed,one=observer.forward({'context':ids},capture=True);phases=[]
                    for t,h in enumerate((one[0],two[0])):
                        if cpu_head is not None:parity.append(check_parity(cpu_head,head,h))
                        scored=head.score(h);token=scored['free_top_token_id']
                        phases.append({'loop':t+1,'score':scored,'free_token_text':tok.tok.decode([token],skip_special_tokens=False,clean_up_tokenization_spaces=False)})
                if cpu_head is not None:
                    del cpu_head;cpu_head=None;gc.collect()
                    write(out/'resident_head_parity.json',{'checks':parity,'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json')})
                ordinal=indices[row['id']];name=f'case_{ordinal:04d}';arrays={};index=[]
                for i,((_,phase,layer,site),tensor) in enumerate(observed.items()):
                    key='v'+str(i);arrays[key]=tensor.numpy();index.append(dict(loop=phase,layer_zero_based=layer,site=site,array_key=key))
                statefile=out/(name+'_states.npz')
                with statefile.open('xb') as stream:np.savez_compressed(stream,**arrays)
                record={k:row[k] for k in ('id','root','split','context_budget','position','task','world','copy_control')}
                record.update(phases=phases,initial_prompt_ids=ids,initial_prompt_ids_sha256=bound['initial_prompt_ids_sha256'],
                    elapsed_seconds=time.monotonic()-begin,CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),state_file=statefile.name,state_file_sha256=sha(statefile),state_index=index)
                write(out/(name+'.json'),record);completed.append(row['id'])
                keep_peaks()
                write(out/'progress.json',{'completed_contexts':len(completed),'current_context_budget':row['context_budget'],
                    'updated_UTC':datetime.now(timezone.utc).isoformat()})
                print({'backend':a.backend,'completed_contexts':len(completed),'context_budget':row['context_budget'],'tokens':len(ids)},flush=True)
            for length in cfg['context_budgets']:
                level={'context_budget':length,'status':'running','DEV_contexts_completed':0}
                try:
                    execute(next(r for r in public if r['split']=='calibration' and r['context_budget']==length))
                    for row in public:
                        if row['split']=='development' and row['context_budget']==length:execute(row);level['DEV_contexts_completed']+=1
                    level['status']='completed'
                except torch.cuda.OutOfMemoryError:
                    level['status']='CUDA_OOM_capacity_stop';keep_peaks();torch.cuda.empty_cache();levels.append(level);status='capacity_stopped';break
                levels.append(level);write(out/f'level_{length}_receipt.json',level)
            else:status='completed'
            assert_oracle_isolated()
        except Exception as error:
            status='budget_reached' if isinstance(error,BudgetReached) else 'failed'
            failure={'type':type(error).__name__,'message':str(error)};traceback.print_exc()
        finally:
            if observer is not None:observer.close()
            keep_peaks()
            write(out/'receipt.json',{'backend':a.backend,'status':status,'failure':failure,'completed_contexts':len(completed),
                'registered_contexts':291,'levels':levels,'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),
                'elapsed_seconds':time.monotonic()-started,'CUDA_peak_allocated_bytes':peaks['allocated'],
                'CUDA_peak_reserved_bytes':peaks['reserved'],'head_residency':'CUDA BF16; unchanged precision',
                'head_parity_checks':len(parity),'host_gold_opened':False,'sealed_test_accessed':False,'BPD_transfer_attempted':False})
    if status=='failed':raise SystemExit(2)


if __name__=='__main__':main()
