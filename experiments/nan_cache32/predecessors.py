"""Creation-bound actual predecessor handles, including BF31 revision2 host lane."""
import json
import os
import subprocess
from .common import RUNS,write


def ensure_inventory_clear(exclude_pid=None):
    exclude=os.getpid() if exclude_pid is None else exclude_pid
    q=(r"@(Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.ProcessId -ne "+str(exclude)+
       r" -and $_.CommandLine -match '-m experiments[.]nan_[^ ]+[.](worker|server|host|run)( |$)'} | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress")
    raw=subprocess.check_output(['powershell','-NoProfile','-Command',q],text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).strip()
    assert not raw or not json.loads(raw),'A known model/host/audit is still active'


def wait(cfg,event):
    from experiments.nan_storyaudit30.common import Handle
    handles=[];records=[];flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    try:
        for spec in cfg['predecessors']:
            pid=spec['pid'];q=f"Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' | Select-Object Name,CommandLine | ConvertTo-Json -Compress"
            raw=subprocess.check_output(['powershell','-NoProfile','-Command',q],text=True,creationflags=flags).strip()
            if not raw:records.append(dict(**spec,status='not_present'));continue
            process=json.loads(raw)
            assert process['Name']=='python.exe' and spec['needle'] in process['CommandLine'],'Predecessor PID identity changed'
            handle=Handle(pid,int(spec['creation']));handles.append((spec['label'],handle));records.append(dict(**spec,status='authenticated_read_only_handle'))
        write(RUNS/'predecessors.json',dict(observations=records,termination_rights=False))
        previous=None
        while True:
            live=[label for label,handle in handles if handle.alive()]
            if not live:break
            if live!=previous:event('waiting_authenticated_predecessors',live=live);previous=live
            next(handle for label,handle in handles if label in live).alive(30000)
    finally:
        for label,handle in handles:handle.close()
    ensure_inventory_clear();event('all_predecessor_and_known_model_host_lanes_ended')
