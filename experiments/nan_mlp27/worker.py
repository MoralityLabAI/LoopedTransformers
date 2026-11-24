"""Fresh full-scope capacity condition; no source26 neural response is reused."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback
from .common import ROOT,RUNS,THREADS,append,read,write,sha,verify,load_sources,intervention_rows,validate_packages

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',required=True,choices=('nf4','bf16'))
    for name in ('snapshot','compactification-root','bpd-root','artifact-root'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    for key in THREADS:os.environ[key]='1'
    os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();seal=verify();cfg=read(ROOT/'config.json')
    assert read(RUNS/'qualification.json')['status']=='qualified_capacity_failure'
    assert subprocess.check_output(['git','-C',a.compactification_root,'rev-parse','HEAD'],text=True).strip()==seal['adapter_commit']
    public,pins=load_sources();assert pins==seal['source_pins']
    out=Path(a.artifact_root)/a.backend;out.mkdir(parents=True,exist_ok=False)
    write(out/'manifest.json',dict(pid=os.getpid(),backend=a.backend,source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
        baseline_registered=1549,AB_pairs_registered=216,source26_replies_reused=0,CPU_threads=1,priority='BelowNormal',
        oracle_opened=False,sealed_test_accessed=False))
    from .engine import Engine
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
        if type(error).__name__=='OutOfMemoryError':status='capacity_stopped'
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
        write(out/'receipt.json',terminal);append(out/'events.jsonl',dict(event='worker_terminal',status=status))
    if status!='completed':raise SystemExit(2)

if __name__=='__main__':main()
