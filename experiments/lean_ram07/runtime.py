"""RAM-05 resource-guarded Lean compiler sessions; local experiment scope only."""
from __future__ import annotations
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import time
_guard_spec=importlib.util.spec_from_file_location('ram03_shared_resource_guard',Path(__file__).resolve().parents[1]/'lean_ram02'/'resources.py')
_guard_module=importlib.util.module_from_spec(_guard_spec)
_guard_spec.loader.exec_module(_guard_module)
ResourceGuard=_guard_module.ResourceGuard
ResourceLimit=_guard_module.ResourceLimit
guarded_process=_guard_module.guarded_process
stop_tree=_guard_module.stop_tree
import ram07_core as c

class ProbeTimeout(RuntimeError):pass

class Session:
    def __init__(self,lean,original,theorem,out,guard,timeout=10.):
        self.out,self.guard,self.lean=Path(out),guard,lean
        self.out.mkdir(parents=True,exist_ok=False)
        self.source=self.out/'Session.lean'
        self.source.write_text(c.make_source(original,theorem,[],server=True),encoding='utf8',newline='\n')
        self.number=0;self.closed=False
        self.log=(self.out/'stdout.log').open('wb')
        self.started=time.perf_counter()
        env=dict(os.environ,RAM03_MAILBOX=str(self.out.resolve()))
        try:
            self.proc=subprocess.Popen([lean,str(self.source.resolve())],stdin=subprocess.DEVNULL,
              stdout=self.log,stderr=subprocess.STDOUT,env=env,
              creationflags=0x08000000 if os.name=='nt' else 0)
            ready=self._response(timeout,0)
            if not ready.get('ready'):raise RuntimeError('server failed to start: '+repr(ready))
        except BaseException:
            if hasattr(self,'proc'):self.close()
            else:self.log.close()
            raise
        self.initial=ready['state']
        self.startup_seconds=time.perf_counter()-self.started

    def _response(self,timeout,number):
        end=time.perf_counter()+max(0.,timeout)
        path=self.out/f'response-{number}.json'
        while True:
            self.guard.check()
            remaining=end-time.perf_counter()
            if remaining<=0:self.close();raise ProbeTimeout('Lean request timed out')
            if path.exists():
                try:return json.loads(path.read_text(encoding='utf8'))
                except (PermissionError,FileNotFoundError):pass
            if self.proc.poll() is not None:
                self.log.flush()
                failure=(self.out/'stdout.log').read_text(encoding='utf8',errors='replace')[-1500:]
                self.close()
                raise ProbeTimeout('Lean process exited unexpectedly; outcome unknown: '+failure)
            time.sleep(min(.005,remaining))

    def _send(self,number,value):
        path=self.out/f'request-{number}.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(value,ensure_ascii=False),encoding='utf8',newline='\n');tmp.replace(path)

    def query(self,actions,timeout=3.):
        if any(not c.valid_script(s) for s in actions):raise ValueError('invalid action prefix')
        if self.closed:raise ProbeTimeout('server already closed')
        self.number+=1;began=time.perf_counter()
        self._send(self.number,dict(id=self.number,actions=actions,stop=False))
        try:
            response=self._response(timeout,self.number)
        except ProbeTimeout as exc:
            c.append_json(self.out/'unknown_compiler_observations.jsonl',dict(
                id=self.number,actions=list(actions),classification='unknown',
                reason=str(exc)[-1500:],request_timeout_seconds=timeout))
            raise
        if response.get('id')!=self.number:raise RuntimeError('Lean protocol desynchronized: '+repr(response))
        result=response['result'];result['wall_seconds']=time.perf_counter()-began
        c.append_json(self.out/'requests.jsonl',dict(id=self.number,actions=actions,result=result))
        return result

    def close(self):
        if self.closed:return
        self.closed=True
        if self.proc.poll() is None:
            try:self._send(self.number+1,dict(id=self.number+1,actions=[],stop=True));self.proc.wait(timeout=.3)
            except (OSError,subprocess.TimeoutExpired):
                if self.proc.poll() is None:stop_tree(self.proc)
        self.log.close()
    def __enter__(self):return self
    def ensure_clean(self):
        if self.proc.returncode!=0:
            raise RuntimeError('probe source did not compile cleanly: '+(self.out/'stdout.log').read_text(encoding='utf8',errors='replace')[-2000:])
    def __exit__(self,*args):
        self.close()
        if args[0] is None:self.ensure_clean()

def certify(lean,original,theorem,actions,prefix,guard,timeout):
    source=c.make_source(original,theorem,actions,server=False)
    return compile_source(lean,source,prefix,guard,timeout)

def compile_source(lean,source,prefix,guard,timeout):
    prefix=Path(prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    path=prefix.with_suffix('.lean');log=prefix.with_suffix('.log')
    path.write_text(source,encoding='utf8',newline='\n')
    result=guarded_process([lean,str(path.resolve())],log,guard,timeout)
    gate,axioms=c.axiom_audit(result['log'])
    accepted=result['returncode']==0 and not result['timed_out'] and gate and 'sorryAx' not in result['log'] and 'error:' not in result['log']
    result.pop('log')
    return dict(**result,accepted=accepted,axiom_gate=gate,axioms=axioms,source_file=str(path),
      source_sha256=c.digest(path),log_file=str(log),log_sha256=c.digest(log))
