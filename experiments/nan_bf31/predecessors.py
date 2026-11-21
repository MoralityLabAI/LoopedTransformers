"""Hold creation-bound read-only handles; never terminate a predecessor."""
import json
import subprocess
from .common import read,write,RUNS


def wait(cfg,event):
    from experiments.nan_storyaudit30.common import Handle
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0);handles=[];observed=[]
    specs=[('model','experiments.nan_ram28.worker --backend nf4'),('queue','experiments.nan_ram28.queue '),
           ('story_follower','experiments.nan_storyaudit30.follow ')]
    try:
        for label,needle in specs:
            pid=cfg['predecessor_'+label+'_pid'];creation=cfg['predecessor_'+label+'_creation']
            q=f"Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
            raw=subprocess.check_output(['powershell','-NoProfile','-Command',q],text=True,creationflags=flags).strip()
            if not raw:observed.append(dict(label=label,pid=pid,status='not_present'));continue
            p=json.loads(raw);assert p['Name']=='python.exe' and '-m '+needle in p['CommandLine'],'Predecessor PID command changed; inspect before admission'
            h=Handle(pid,creation);handles.append((label,h));observed.append(dict(label=label,pid=pid,creation=h.creation,status='authenticated_read_only_handle'))
        write(RUNS/'predecessors.json',dict(observations=observed,termination_rights=False))
        last=None
        while True:
            live=[label for label,h in handles if h.alive()]
            if not live:break
            if live!=last:event('waiting_predecessor_actual_handles',live=live);last=live
            next(h for label,h in handles if label in live).alive(30000)
    finally:
        for _,h in handles:h.close()
    # Includes canonical hosts and the early grading child, if any survived a
    # predecessor launcher. It never mistakes a closed marker for process exit.
    q=r"@(Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match '-m experiments\.nan_[^\s]+\.(worker|server|host|run)(\s|$)'} | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress"
    raw=subprocess.check_output(['powershell','-NoProfile','-Command',q],text=True,creationflags=flags).strip()
    assert not raw or not json.loads(raw),'A known model/host/audit remains; inspect without overlap'
    event('predecessors_and_known_model_host_audit_lanes_ended')
