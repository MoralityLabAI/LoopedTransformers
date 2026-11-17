"""Bounded model-only development runner. No EvalMath/labels in this process."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import torch
from .backend import load_backend,candidate_inputs
from .edits import dose_edit,iter_phase_directions,prefix_hidden_error
from .head import scores_at_temperatures
from mechinterp.dark_fiber_gyms.arg03_nanbeige import assert_oracle_isolated,public_rows

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8');tmp.replace(path)

class MemorySampler:
    def __init__(self,out):self.out=out;self.stop=threading.Event();self.samples=[]
    def sample(self):
        record={'elapsed_seconds':round(time.perf_counter()-self.start,3),'pid':os.getpid()}
        try:
            r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=10,creationflags=0x08000000)
            record['driver_device_global_mib']=[int(v.strip()) for v in r.stdout.strip().split(',')]
        except Exception as e:record['driver_error']=repr(e)
        try:
            cmd=f"$m=Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Where-Object {{$_.Name -like 'pid_{os.getpid()}_*'}}; @($m | Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted) | ConvertTo-Json -Compress"
            r=subprocess.run(['powershell','-NoProfile','-Command',cmd],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20,creationflags=0x08000000)
            record['windows_process_gpu_memory']=json.loads(r.stdout) if r.stdout.strip() else []
            if r.returncode:record['shared_error']=r.stderr
        except Exception as e:record['shared_error']=repr(e)
        self.samples.append(record);write(self.out,self.samples)
    def worker(self):
        while not self.stop.wait(15):self.sample()
    def __enter__(self):
        self.start=time.perf_counter();self.sample();self.thread=threading.Thread(target=self.worker,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join(timeout=35);self.sample()

def summarize_observations(records):
    groups={}
    for r in records:
        key=f"loop{r['phase']}/layer{r['layer']}/{r['site']}"
        groups.setdefault(key,[]).append(r)
    result={}
    for key,rows in groups.items():
        vals=torch.cat([r['values'] for r in rows],dim=0).float()
        norms=torch.cat([r['token_l2'] for r in rows],dim=0).float()
        mx=torch.cat([r['token_max_abs'] for r in rows],dim=0).float()
        bins={}
        for name,sl in [('all',slice(None)),('context8',slice(-11,-3)),('scored3',slice(-3,None)),('last',slice(-1,None))]:
            v=vals[:,sl,:]
            bins[name]={'channel_mean':v.mean((0,1)).tolist(),'channel_std':v.std((0,1),unbiased=False).tolist(),
                        'channel_mean_abs':v.abs().mean((0,1)).tolist(),'mean_token_l2':float(norms[:,sl].mean()),
                        'max_token_abs':float(mx[:,sl].max())}
        result[key]={'channels':rows[0]['channels'],'positions':vals.shape[1],'prefix_rows':vals.shape[0],'bins':bins,
                     'channel_mean_by_token':vals.mean(0).tolist()}
    return result

def checked_resume(out,pairs,meta,enabled):
    receipt=out/'receipt.json'
    if not enabled:
        if receipt.exists():raise FileExistsError('Existing receipt: choose a new out directory or explicit --resume')
        return None,[]
    prior=json.loads(receipt.read_text(encoding='utf-8'))
    if prior.get('attempts'):raise RuntimeError('One bounded resume is supported; inspect receipts before further launches')
    for key in ['public_sha256','pairs_sha256','config_sha256','backend','checkpoint_revision']:
        if prior[key]!=meta[key]:raise ValueError(f'Resume source mismatch: {key}')
    completed=prior['completed_cases'];expected={pair[role] for pair in pairs for role in ['recipient','donor']}
    if not set(completed)<=expected:raise ValueError('Unexpected completed case ID')
    for pair in pairs:
        present=[pair[role] in completed for role in ['recipient','donor']]
        if any(present) and not all(present):raise ValueError('Partial pair receipt requires manual audit')
        if all(present):
            for role in ['recipient','donor']:
                cid=pair[role];r=json.loads((out/(cid.replace(':','_')+'.json')).read_text())
                if (r['case_id'],r['root'],r['split'])!=(cid,pair['root'],pair['split']):raise ValueError('Case receipt binding mismatch')
    return prior,list(completed)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--compactification-root',required=True);p.add_argument('--snapshot',required=True)
    p.add_argument('--cohort',required=True);p.add_argument('--backend',choices=['bf16','nf4'],required=True);p.add_argument('--out',required=True)
    p.add_argument('--resume',action='store_true',help='Finish only pending fixed pairs after one bounded attempt; no new cases')
    a=p.parse_args();start=time.perf_counter();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cfgpath=Path(__file__).with_name('config.json');cfg=json.loads(cfgpath.read_text(encoding='utf-8'))
    fixture=out.parent/'engineering'/'gpu_fixture.json'
    if not fixture.exists() or json.loads(fixture.read_text())['status']!='passed':raise RuntimeError('GPU engineering fixture must pass first')
    torch.manual_seed(cfg['seed']);torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']);torch.cuda.reset_peak_memory_stats()
    source={str(p.relative_to(Path(__file__).parent)):sha(p) for p in Path(__file__).parent.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.md'] and '__pycache__' not in str(p)}
    cohort=Path(a.cohort);rows={r['case_id']:r for r in public_rows(cohort/'public.jsonl')};pairs=json.loads((cohort/'pairs.json').read_text())
    meta={'schema_version':1,'backend':a.backend,'checkpoint_revision':cfg['checkpoint_revision'],'source_sha256':source,
          'public_sha256':sha(cohort/'public.jsonl'),'pairs_sha256':sha(cohort/'pairs.json'),'config_sha256':sha(cfgpath),
          'quantization':{'format':'bitsandbytes NF4','double_quantization':True,'compute_dtype':'bfloat16'} if a.backend=='nf4' else None,
          'placement':{'head':'CPU, vocabulary-row chunks to GPU','embeddings':'CPU','decoder':'all NF4 CUDA' if a.backend=='nf4' else 'one resident BF16 layer plus one CUDA staging layer'},
          'pid':os.getpid(),'torch':torch.__version__,'candidate_microbatch':1,'native_loops':[1,2],
          'sealed_test_accessed':False,'oracle_imported':False,'checkpoint_verified_before_load':True,'commands':vars(a)}
    prior,completed=checked_resume(out,pairs,meta,a.resume)
    if prior and len(completed)==2*len(pairs):print('All fixed pairs already complete',flush=True);return
    if prior:shutil.copyfile(out/'receipt.json',out/'initial_execution_receipt.json')
    replacement=[];status='running';failure=None
    with MemorySampler(out/('memory_samples_resume.json' if prior else 'memory_samples.json')):
        try:
            print(f'Loading verified {a.backend} checkpoint with CPU output head',flush=True)
            loop,head,tok,entry=load_backend(a.compactification_root,a.snapshot,a.backend,chunk_rows=cfg['vocabulary_chunk_rows'])
            meta['operator_ids']=dict(zip('INSR',head.ids));meta['decoder_quantized_linears']=loop.quantized_linears
            write(out/('manifest_resume.json' if prior else 'manifest.json'),meta)
            for pairindex,pair in enumerate(pairs):
                if all(pair[role] in completed for role in ['recipient','donor']):continue
                if time.perf_counter()-start>cfg['maximum_backend_seconds']:status='time_budget_reached';break
                print(f"{a.backend} root {pair['root']} {pair['split']} starting",flush=True)
                pair_start=time.perf_counter();cache={}
                for role in ['recipient','donor']:
                    row=rows[pair[role]];ids,promptlen=candidate_inputs(tok,head.ids and dict(zip('INSR',head.ids)),a.snapshot,row)
                    emb,ctx=loop.prepare(ids);h1=loop.step(emb,ctx,1,observe=True);obs=list(loop.observations)
                    h2=loop.step(h1,ctx,2,observe=True);obs+=list(loop.observations)
                    records=dict(zip(['native1','native2','open_loop'],head.score([h1,h2,(2*h1.float()-emb.float()).to(h1.dtype)])))
                    swapped,_=candidate_inputs(tok,dict(zip('INSR',head.ids)),a.snapshot,row,variant='swap')
                    se,sc=loop.prepare(swapped);sh=loop.step(se,sc,1);sr=head.score([sh])[0]
                    records['equal_compute']={'scores':[(x+y)/2 for x,y in zip(records['native1']['scores'],sr['scores'])],
                                              'description':'mean canonical/swap one-loop log scores; two decoder traversals'}
                    del se,sh
                    if pair['split']=='development' and cfg['development_roots'].index(pair['root']) in cfg['phase_intervention_root_indices'] and role=='recipient':
                        bare,_=candidate_inputs(tok,dict(zip('INSR',head.ids)),a.snapshot,row,native_chat=False)
                        be,bc=loop.prepare(bare);bh1=loop.step(be,bc,1);bh2=loop.step(bh1,bc,2)
                        records['bare_native2']=head.score([bh2])[0];del be,bh1,bh2
                    summary=summarize_observations(obs)
                    rec={'case_id':row['case_id'],'root':pair['root'],'split':pair['split'],'role':role,'family':pair['family'],
                         'prompt_tokens':promptlen,'records':records,'phase_observations':summary,'interventions':{}}
                    torch.save(obs,out/(row['case_id'].replace(':','_')+'_activations.pt'))
                    if pair['split']=='calibration':
                        key='loop2/layer21/loop_boundary_post_norm'
                        replacement.append(summary[key]['bins']['context8']['channel_mean'][0])
                    cache[role]=(h1,h2,ctx,rec)
                    del emb,obs
                if pair['split']=='calibration':
                    write(out/'calibration_phase.json',{'replacement_loop2_channel_mean':sum(replacement)/len(replacement),
                        'calibration_cases':len(replacement),'development_cases_used':0,'rule':'mean postnorm loop2 context8 channel2279; CAL only'})
                else:
                    if not (out/'calibration_phase.json').exists():raise RuntimeError('Calibration must precede development')
                    if cfg['development_roots'].index(pair['root']) in cfg['phase_intervention_root_indices']:
                        h1,h2,ctx,rec=cache['recipient'];dh1,dh2,_,drec=cache['donor']
                        replacement_mean=json.loads((out/'calibration_phase.json').read_text())['replacement_loop2_channel_mean']
                        for name,direction in iter_phase_directions(h1,cfg,pair['root'],replacement_mean):
                            edited,dose=dose_edit(h1,direction,cfg['physical_relative_dose'],cfg['maximum_relative_dose_error'])
                            current=head.score([h1,edited])[1];nextstate=loop.step(edited,ctx,2);delayed=head.score([h2,nextstate])[1]
                            if max(prefix_hidden_error(edited),prefix_hidden_error(nextstate))!=0:raise RuntimeError('Candidate-prefix state consistency failed')
                            cosine=float(torch.nn.functional.cosine_similarity((nextstate.float()-h2.float()).flatten(),(dh2.float()-h2.float()).flatten(),dim=0))
                            rec['interventions'][name]={'dose':dose,'current':current,'delayed':delayed,'next_state_l2':float((nextstate.float()-h2.float()).norm()),'cosine_to_donor_next_displacement':cosine}
                            del edited,nextstate
                        # Actual checkpoint closure, not merely a random-model fixture.
                        transferred=loop.step(dh1,ctx,2)
                        closure=float((transferred.float()-dh2.float()).abs().max())
                        rec['interventions']['full_donor_state']={'delayed':head.score([h2,transferred])[1],
                            'state_closure_max_error':closure,'physical_l2':float((dh1.float()-h1.float()).norm()),'dose_matched':False}
                        rec['interventions']['zero_write']={'current':rec['records']['native1'],'delayed':rec['records']['native2'],'actual_l2':0.,'zero_dose':True}
                        rec['interventions']['source_input']={'delayed':drec['records']['native2'],'description':'native donor public input, no state intervention'}
                        del transferred,direction
                for role,(_,_,_,rec) in cache.items():
                    write(out/(rec['case_id'].replace(':','_')+'.json'),rec);completed.append(rec['case_id'])
                print(f"{a.backend} root {pair['root']} saved in {time.perf_counter()-pair_start:.1f}s; peak alloc {torch.cuda.max_memory_allocated()/2**30:.3f} GiB",flush=True)
                del cache;torch.cuda.empty_cache()
            else:status='completed'
        except Exception as e:
            status='failed';failure={'type':type(e).__name__,'message':str(e)}
            import traceback;traceback.print_exc()
        finally:
            assert_oracle_isolated();torch.cuda.synchronize()
            receipt=meta|{'status':status,'failure':failure,'completed_cases':completed,'elapsed_seconds':round(time.perf_counter()-start,3),
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
                'head_remained_cpu':bool('head' in locals() and head.weight.device.type=='cpu')}
            if prior:
                attempt=dict(receipt);attempt['completed_cases']=[cid for cid in completed if cid not in prior['completed_cases']]
                receipt['attempts']=[prior,attempt]
                receipt['elapsed_seconds']=round(prior['elapsed_seconds']+attempt['elapsed_seconds'],3)
                for key in ['peak_cuda_allocated_bytes','peak_cuda_reserved_bytes']:receipt[key]=max(prior[key],attempt[key])
            write(out/'receipt.json',receipt);print(json.dumps({k:receipt[k] for k in ['status','failure','completed_cases','elapsed_seconds','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes']}),flush=True)
    if status!='completed':raise SystemExit(2)

if __name__=='__main__':main()
