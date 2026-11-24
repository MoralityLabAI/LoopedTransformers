"""Metadata-only, authenticated one-shot follower; never restarts inference."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT,RUNS,THREADS,Handle,read,write,append,verify,closed_entries,select


def main():
    p=argparse.ArgumentParser()
    for name in ('python','snapshot','artifact-root'):p.add_argument('--'+name,required=True)
    p.add_argument('--model-pid',required=True,type=int);p.add_argument('--queue-pid',required=True,type=int)
    p.add_argument('--publish',action='store_true');a=p.parse_args()
    for name in THREADS:os.environ[name]='1'
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();verify();cfg=read(ROOT/'registration.json');flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    RUNS.mkdir(parents=True,exist_ok=False)
    def event(name,**fields):
        row=dict(UTC=datetime.now(timezone.utc).isoformat(),event=name,**fields)
        append(RUNS/'follower_events.jsonl',row);print(json.dumps(row),flush=True)
    # Validate command identities immediately before acquiring held handles.
    for pid,needle in ((a.model_pid,'experiments.nan_ram28.worker --backend nf4'),(a.queue_pid,'experiments.nan_ram28.queue ')):
        query=f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}';if ($p.Name -ne 'python.exe' -or $p.CommandLine -notlike '*-m {needle}*') {{exit 3}}"
        subprocess.run(['powershell','-NoProfile','-Command',query],check=True,creationflags=flags)
    model=Handle(a.model_pid);queue=Handle(a.queue_pid)
    try:
        assert model.alive() and queue.alive()
        write(RUNS/'follower_identity.json',dict(pid=os.getpid(),model_pid=a.model_pid,model_creation=model.creation,
            queue_pid=a.queue_pid,queue_creation=queue.creation,one_shot=True,sealed_test_accessed=False))
        event('authenticated_handles',model_pid=a.model_pid,queue_pid=a.queue_pid)
        base=Path(a.artifact_root)/'nf4';last=None
        while True:
            entries=select(closed_entries(base),cfg['ordered_source_ids']);count=len(entries)
            if count!=last:event('closed_story_coverage',closed=count,registered=72);last=count
            if count==72 or not model.alive():break
            model.alive(30000)
        overlap=model.alive()
        if not overlap:
            event('waiting_canonical_supervisor_and_host',closed=count)
            while queue.alive(30000):pass
        from experiments.nan_ram28.resources import memory
        while memory()['available_physical_bytes']<cfg['minimum_start_available_RAM_bytes']:
            event('waiting_RAM_admission',available_physical_bytes=memory()['available_physical_bytes'])
            if overlap and not model.alive():
                overlap=False
                while queue.alive(30000):pass
            time.sleep(30)
        command=[a.python,'-X','utf8','-B','-u','-m','experiments.nan_storyaudit30.run',
                 '--artifact-root',a.artifact_root,'--snapshot',a.snapshot]
        if overlap:command+=['--model-pid',str(a.model_pid),'--model-creation',str(model.creation)]
        with (RUNS/'audit.log').open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,creationflags=flags)
            event('bounded_audit_started',launcher_pid=child.pid,overlap_live_model=overlap)
            code=child.wait();event('bounded_audit_terminal',exit_code=code)
        if code:return
        from .run import OUTPUTS
        assert read(RUNS/'completion.json')['status'] in ('complete_story_block','partial_closed_story_block')
        if a.publish:
            for command in (['git','add','--sparse','--',*OUTPUTS],
                ['git','commit','--only','-m','Publish early bounded RAM28 storyworld audit','--',*OUTPUTS],
                ['git','push','origin','HEAD:refs/heads/research/nanbeige-bpd-gym-suite-towers']):
                subprocess.run(command,check=True,creationflags=flags)
            event('early_report_pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        event('one_shot_follower_finished',continuous_goal_remains_active=True)
    finally:model.close();queue.close()


if __name__=='__main__':main()
