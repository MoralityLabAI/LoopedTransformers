"""Fresh NF4 full gym under the explicit RAM28 loader/resource condition."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback
from .common import ROOT,RUNS,THREADS,append,read,write,sha,verify
from experiments.nan_mlp27.common import load_sources,intervention_rows,validate_packages

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',required=True,choices=('nf4',))
    for name in ('snapshot','compactification-root','bpd-root','artifact-root'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    write(RUNS/'nf4_runtime.json',dict(pid=os.getpid(),backend='nf4',artifact_root=a.artifact_root,condition='NAN-RAM-28'))
    (RUNS/'nf4_runtime_ready.signal').touch(exist_ok=False)
    for key in THREADS:os.environ[key]='1'
    os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
    os.environ['HF_HOME']=str((RUNS/'HF_cache').resolve())
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();seal=verify();cfg=read(ROOT/'config.json')
    assert subprocess.check_output(['git','-C',a.compactification_root,'rev-parse','HEAD'],text=True).strip()==seal['adapter_commit']
    public,pins=load_sources();assert pins==seal['source_pins']
    out=Path(a.artifact_root)/a.backend;out.mkdir(parents=True,exist_ok=False)
    write(out/'manifest.json',dict(experiment='NAN-RAM-28',method_source_manifest_sha256=sha('experiments/nan_mlp27/SOURCE_MANIFEST.json'),pid=os.getpid(),backend=a.backend,source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
        baseline_registered=1549,AB_pairs_registered=216,source26_replies_reused=0,CPU_threads=1,priority='BelowNormal',
        oracle_opened=False,sealed_test_accessed=False))
    import experiments.nan_bpd01.backend as backend_module
    from .loader import load_backend
    backend_module.load_backend=load_backend
    from experiments.nan_mlp27.engine import Engine as MethodEngine
    from .resources import soft_guard
    class Engine(MethodEngine):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            from .loader import EXPECTED_TARGET
            assert self.before==EXPECTED_TARGET and self.loop.n_gpu_layers==22
            sample=soft_guard(self.cfg)
            write(self.out/'resource_after_load.json',dict(memory=sample,decoder_GPU_layers=22,
                head_device=str(self.head.weight.device),embedding_device=str(self.loop.base.embed_tokens.weight.device),
                parameter_sha256=self.before,pretrained_forwards=0,source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json')))
            append(RUNS/'resource_events.jsonl',dict(event='model_loaded_and_weight_checked',memory=sample))
        def capture(self,*args,**kwargs):
            soft_guard(self.cfg)
            result=super().capture(*args,**kwargs)
            soft_guard(self.cfg)
            return result
    engine=None;status='failed';failure=None;baseline=ab=0;started=time.monotonic()
    try:
        validate_packages('GPU')
        engine=Engine(a,cfg,out);engine.preflight(public[0]);append(out/'events.jsonl',dict(event='preflight_closed'))
        index=0
        while index<len(public):
            selected=[public[index]];first=engine.ids(selected[0])
            if index+1<len(public):
                nxt=engine.ids(public[index+1])
                if engine.admitted(2,max(len(first),len(nxt))+256):selected.append(public[index+1])
            engine.generate(selected,['native']*len(selected),index,'baseline');index+=len(selected);baseline=index
            append(out/'events.jsonl',dict(event='baseline_batch_closed',completed=baseline,registered=1549))
        for route in cfg['AB_routes']:
            for row in intervention_rows(public):
                assert engine.admitted(2,len(engine.ids(row))+256),'registered_AB_capacity_unavailable'
                engine.generate([row,row],['native',route],ab,'AB');ab+=1
                append(out/'events.jsonl',dict(event='AB_pair_closed',completed=ab,registered=216,route=route))
        status='completed'
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error));traceback.print_exc()
        if type(error).__name__=='HostMemoryBudgetReached':status='host_memory_budget_reached'
        elif type(error).__name__=='OutOfMemoryError':status='capacity_stopped'
        elif str(error)=='registered_padded_capacity_gate_failed_before_scored_inference':status='capacity_not_admitted'
    finally:
        terminal=dict(status=status,failure=failure,baseline_completed=baseline,baseline_registered=1549,
            AB_pairs_completed=ab,AB_pairs_registered=216,source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
            elapsed_seconds_including_load=time.monotonic()-started,source26_replies_reused=0,neural_retries=0,sealed_test_accessed=False)
        if engine:
            from experiments.nan_recomp09.common import target_digest
            engine.observer.close()
            if engine.instrument:engine.instrument.close()
            engine.workspace.close();after=target_digest(engine.loop)
            terminal.update(target_before=engine.before,target_after=after,target_unchanged=engine.before==after,
                CUDA_peak_allocated_bytes=engine.t.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=engine.t.cuda.max_memory_reserved(),
                artifact_bytes=engine.bytes,completed_captures=engine.forward_count,capture_attempts=engine.capture_attempts,
                loaded_model_elapsed_seconds=time.monotonic()-engine.started)
            if after!=engine.before:
                status='target_changed';terminal['status']=status
                terminal['failure']=dict(type='TargetChanged',message='Final parameter bytes differ; preserve failure')
        write(out/'receipt.json',terminal);append(out/'events.jsonl',dict(event='worker_terminal',status=status))
    if status!='completed':raise SystemExit(2)

if __name__=='__main__':main()
