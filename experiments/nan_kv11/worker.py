"""Actual checkpoint cache comparison; no task grading or donor intervention."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import torch
from experiments.nan_bpd01.backend import load_backend
from experiments.nan_comp02.head import OperatorHead
from experiments.nan_comp03.prompt import aliases
from experiments.nan_gym05.common import prompt_ids
from experiments.nan_gym05.observe import TowerObserver
from mechinterp.dark_fiber_gyms.arg03_nanbeige import assert_oracle_isolated
from mechinterp.nanbeige_bpd_gym.native_cached import NativeTwoLoopCache
from mechinterp.nanbeige_bpd_gym.resident_head import mount,retain_decoder_layers,full_logits,check_parity
from experiments.nan_recomp09.common import target_digest
from .common import read,sha,write,public_rows,verify


class BudgetReached(RuntimeError):pass


def driver_sample():
    try:
        command=f"ConvertTo-Json -Compress -InputObject @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Where-Object {{ $_.Name -like 'pid_{os.getpid()}_*' }} | Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted)"
        values=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW))
        nv=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW).strip().split(',')
        return dict(status='sampled',WDDM_worker_rows=values,nvidia_device_used_MiB=int(nv[0]),nvidia_total_MiB=int(nv[1]),scope='instantaneous counters; shared usage may include runtime/transfer memory; not exact residency/peak proof')
    except Exception as error:return {'status':'unavailable','error_type':type(error).__name__}


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('bf16','nf4'),required=True);p.add_argument('--snapshot',required=True)
    p.add_argument('--compactification-root',required=True);p.add_argument('--runs',default='runs/nan_kv11');a=p.parse_args()
    root=Path(__file__).parent;seal=verify(root);cfg=read(root/'config.json');public,dev=public_rows();out=Path(a.runs)/a.backend;out.mkdir(parents=True,exist_ok=False)
    cache=out/'HF_cache';cache.mkdir();os.environ['HF_HOME']=str(cache.resolve());os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
    torch.set_num_threads(1);torch.manual_seed(cfg['source_seed']);torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']);torch.cuda.reset_peak_memory_stats()
    started=time.monotonic();status='failed';failure=None;completed=0;cached=None;observer=None;attempted=0
    def budget():
        if time.monotonic()-started>cfg['maximum_backend_seconds']:raise BudgetReached('Registered between-step backend budget reached')
    def event(**fields):
        with (out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(fields)+'\n')
        print(fields,flush=True)
    try:
        assert_oracle_isolated();loop,legacy,tok,_=load_backend(a.compactification_root,a.snapshot,a.backend,chunk_rows=cfg['vocabulary_chunk_rows'])
        if loop.stage_layer is not None:loop.stage_layer.requires_grad_(False).eval()
        if a.backend=='bf16':retain_decoder_layers(loop,cfg['BF16_resident_decoder_layers'])
        cpu=OperatorHead(legacy.weight,aliases(tok),chunk_rows=cfg['vocabulary_chunk_rows']);head=mount(loop,cpu);del legacy
        before=target_digest(loop);observer=TowerObserver(loop,cfg['layers_zero_based']);cached=NativeTwoLoopCache(loop,observe_layers=cfg['layers_zero_based'])
        ids=prompt_ids(tok,a.snapshot,next(r for r in public if r['split']=='calibration')['prompt'])
        with torch.inference_mode():
            two,_,one=observer.forward({'context':ids},capture=False);parity=[check_parity(cpu,head,h) for h in (one[0],two[0])]
            cached.prefill(ids);cached.reset()
        write(out/'head_parity.json',parity);del cpu,two,one;gc.collect()
        write(out/'driver_after_warmup.json',driver_sample())
        write(out/'manifest.json',dict(backend=a.backend,source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'),target_parameter_sha256=before,
            contexts=130,source_families=41,head_device=str(head.weight.device),CPU_threads=torch.get_num_threads(),host_labels_opened=False,sealed_test_accessed=False))
        for ordinal,row in enumerate(dev):
            budget();attempted+=1;initial=prompt_ids(tok,a.snapshot,row['prompt']);history=list(initial);generated=[];steps=[];cached.reset()
            record={k:row[k] for k in ('id','root','suite','split')};record.update(copy_control=row['id'].endswith(':copy'),initial_prompt_ids=initial,initial_prompt_ids_sha256=hashlib.sha256(json.dumps(initial).encode()).hexdigest())
            try:
                for step in range(cfg['maximum_generated_tokens']):
                    budget();arrays={}
                    with torch.inference_mode():
                        torch.cuda.synchronize();mark=time.monotonic();two,observed,one=observer.forward({'context':history},capture=True)
                        reference=[one[0][:,-1:],two[0][:,-1:]];ref_logits=[full_logits(head.weight,h,loop.device,cfg['vocabulary_chunk_rows']) for h in reference]
                        torch.cuda.synchronize();reference_seconds=time.monotonic()-mark
                        torch.cuda.synchronize();mark=time.monotonic();states,tape=cached.prefill(history,capture=True) if step==0 else cached.step(generated[-1],capture=True)
                        cand_logits=[full_logits(head.weight,h,loop.device,cfg['vocabulary_chunk_rows']) for h in states];torch.cuda.synchronize();cached_seconds=time.monotonic()-mark
                        assert observed.keys()==tape.keys() and len(tape)==40
                        phases=[];prefill_exact=True
                        for t,(h,c,rl,cl) in enumerate(zip(reference,states,ref_logits,cand_logits),1):
                            same=torch.equal(h,c) and torch.equal(rl,cl);prefill_exact &= same
                            phases.append(dict(loop=t,normalized_boundary_relative_l2=float((c.float()-h.float()).norm()/h.float().norm().clamp_min(1e-12)),
                                logit_max_abs=float((rl-cl).abs().max()),free_argmax_equal=int(rl.argmax())==int(cl.argmax()),logits_bit_exact=torch.equal(rl,cl)))
                            arrays[f'reference_boundary_{t}']=h.flatten().float().cpu().numpy();arrays[f'cached_boundary_{t}']=c.flatten().float().cpu().numpy()
                            arrays[f'reference_logits_{t}']=rl.numpy();arrays[f'cached_logits_{t}']=cl.numpy()
                        for k in sorted(observed):
                            _,t,l,site=k;arrays[f'reference_{t}_{l}_{site}']=observed[k].numpy();arrays[f'cached_{t}_{l}_{site}']=tape[k].numpy()
                            prefill_exact &= torch.equal(observed[k],tape[k])
                        token=int(ref_logits[1].argmax());candidate=int(cand_logits[1].argmax());generated.append(token)
                    file=out/f'case_{ordinal:04d}_step_{step:02d}.npz'
                    with file.open('xb') as f:np.savez_compressed(f,**arrays)
                    steps.append(dict(step=step,history_tokens=len(history),reference_token_id=token,cached_free_token_id_on_reference_history=candidate,
                        phases=phases,reference_seconds=reference_seconds,cached_seconds=cached_seconds,cache_bytes=cached.cache_bytes(),cache_sequence_length=cached.length,
                        prefill_all_sites_and_logits_bit_exact=bool(prefill_exact) if step==0 else None,artifact_file=file.name,artifact_sha256=sha(file)))
                    del arrays,two,one,reference,ref_logits,states,tape,cand_logits,observed,h,c,rl,cl
                    if step==0 and not prefill_exact:raise AssertionError('Initial cached prefill is not bit exact; preserve this condition')
                    if token==tok.tok.eos_token_id:break
                    history.append(token)
                record.update(status='completed',steps=steps,reference_generated_ids=generated,reference_ended_EOS=generated[-1]==tok.tok.eos_token_id,
                    all_forced_history_argmaxes_equal=all(s['reference_token_id']==s['cached_free_token_id_on_reference_history'] for s in steps))
                write(out/f'case_{ordinal:04d}.json',record);completed+=1;event(backend=a.backend,completed_contexts=completed,reference_tokens=len(generated))
            except Exception as error:
                record.update(status='failed',steps=steps,reference_generated_ids=generated,failure_type=type(error).__name__);write(out/f'case_{ordinal:04d}.json',record);raise
        assert target_digest(loop)==before;assert_oracle_isolated();status='completed'
    except Exception as error:
        status='budget_reached' if isinstance(error,BudgetReached) else 'failed';failure={'type':type(error).__name__,'message':str(error)};traceback.print_exc()
    finally:
        write(out/'driver_terminal.json',driver_sample())
        if cached is not None:cached.close()
        if observer is not None:observer.close()
        write(out/'receipt.json',dict(backend=a.backend,status=status,failure=failure,completed_contexts=completed,attempted_contexts=attempted,
            source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'),elapsed_seconds=time.monotonic()-started,
            CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            target_parameter_sha256_unchanged=before if status=='completed' else None,sealed_test_accessed=False,BPD_donor_transfer_attempted=False))
    if status!='completed':raise SystemExit(2)


if __name__=='__main__':main()
