"""Keep foreign model/host processes blocked; recognize only our host launcher."""
import json
import ntpath
import os
import re
import subprocess
import sys

KNOWN=re.compile(r'(?:^| )-m experiments[.]nan_[^ ]+[.](worker|server|host|run)(?: |$)')
HOST=re.compile(r'^-X utf8 -B -u -m experiments[.]nan_cache32[.]host(?: |$)')


def arguments(command):
    if not isinstance(command,str) or any(c in command for c in '\r\n\0'):return None
    match=re.fullmatch(r'(?:"[^"\r\n]+"|[^\s"]+)\s+(.+)',command)
    return match[1] if match else None


def owned_host_launcher(own,parent,expected_launcher):
    """Conservative identity checks, never an exemption for arbitrary parents."""
    if not own or not parent:return False
    if any(type(v.get(k)) is not int for v,k in ((own,'pid'),(own,'parent_pid'),(parent,'pid'),(own,'creation'),(parent,'creation'))):return False
    if own['parent_pid']!=parent['pid'] or not 0<parent['creation']<own['creation']:return False
    if own.get('name','').lower()!='python.exe' or parent.get('name','').lower()!='python.exe':return False
    if ntpath.normcase(ntpath.normpath(parent.get('exe') or ''))!=ntpath.normcase(ntpath.normpath(expected_launcher)):return False
    own_args=arguments(own.get('command'));parent_args=arguments(parent.get('command'))
    return bool(own_args and own_args==parent_args and HOST.match(own_args))


def blockers(records,own_pid,expected_launcher):
    bypid={r['pid']:r for r in records};assert len(bypid)==len(records),'Duplicate process PID'
    own=bypid.get(own_pid);excluded={own_pid}
    parent=bypid.get(own['parent_pid']) if own else None
    if owned_host_launcher(own,parent,expected_launcher):excluded.add(parent['pid'])
    return [r for r in records if r['pid'] not in excluded and KNOWN.search(r.get('command') or '')]


def collect():
    """Read process metadata and bind our actual/parent creation times to handles."""
    command="@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine) | ConvertTo-Json -Compress"
    raw=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=30).strip()
    values=json.loads(raw) if raw else []
    if isinstance(values,dict):values=[values]
    records=[dict(pid=v['ProcessId'],parent_pid=v['ParentProcessId'],name=v['Name'],exe=v['ExecutablePath'],command=v['CommandLine'],creation=0) for v in values]
    own=next((v for v in records if v['pid']==os.getpid()),None)
    assert own is not None and own['parent_pid']==os.getppid(),'Current process identity missing or changed'
    from experiments.nan_storyaudit30.common import Handle
    for record in records:
        if record['pid'] not in (own['pid'],own['parent_pid']):continue
        try:
            handle=Handle(record['pid'])
            try:
                if handle.alive():record['creation']=handle.creation
            finally:handle.close()
        except OSError:pass  # Missing/ambiguous parent is never exempted.
    return records


def ensure_clear(exclude_pid=None):
    assert exclude_pid in (None,os.getpid()),'May exclude only the actual calling process'
    remaining=blockers(collect(),os.getpid(),ntpath.join(sys.prefix,'Scripts','python.exe'))
    assert not remaining,'A known model/host/audit is still active: '+str([r['pid'] for r in remaining])
