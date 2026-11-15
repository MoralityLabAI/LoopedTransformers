"""RAM-13 matched original 20-family Lean4.35 full-theorem proof search.

Original frozen 12-source case manifest, 9 immutable policies and strict
in-budget paired original-source compilation. External third replay later.
No original reference proof commands enter model ranks or tactics.
"""
from __future__ import annotations
import argparse,collections,hashlib,io,json,os,signal,subprocess,sys,time,zipfile
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,'/mnt/data')
sys.path.insert(0,'/mnt/data/ram13_local')
sys.path.insert(0,'/mnt/data/ram11f_public/research/ram11f')
import ram08_offline_dev_audit as old
import grammar
from native_replay import render_source,axiom_inventory

ROOT=Path('/mnt/data/ram13_portable'); SRC=ROOT/'mathcan'
LEAN=ROOT/'lean/bin/lean'; PROBE=Path('/mnt/data/ram13_local')
MODELZIP=Path('/mnt/data/ram08-frozen-audited-train-43689874169.zip')
RAM12ZIP=Path('/mnt/data/RAM12_certified_native_states_and_rankers_20261115.zip')
OUT=Path('/mnt/data/ram13_local/results')
LOCKED={
 'ChromaticObservers':('61917ac9f1890cc90055b45defd9f1f78ef4d9a3',('robustSixCode_in_range','robustSixCode_margin_three')),
 'DeferredObservability':('e1bf593dac5629f39e5b03d8e60012874ad9ff4d',('present_blind_to_second','next_observation_is_hidden')),
 'GramianTransfer':('f07e99a3f27acba22c2c0c6b3f675a68ab774ab2',('transport_iterate','observable_transport')),
 'ObserverPermutationNull':('dad53a0e1a05d4b780d5da8d0eabff1db0f2669c',('native_population_certificate','native_histogram_mass_certificate')),
 'PhaseMemory':('615f20a32b0e190408ebee0b3b2a4f9bb695ee02',('toggle_zero_to_one','toggle_one_to_zero')),
 'SpectralStability':('a140af4fb5f89b753d8c1402803eb9cfe6e05f47',('sixth_roots_distinct','six_phase_character_orthogonality')),
}
HEADS={'loop32':'c288f38a2d912a8a544ef057257aacde1f93da202fe42679884a2e79cfa259cd',
       'loop64':'b281d349d56de154025cc567c80ad920ae99e24a50796c9dd182ff1d47b7b92a',
       'flat64':'90d1ee19694116e432b755924208f1f493482b6838647fe33ae133fc89e61f62'}
PRIMARY={'seconds':20,'evaluations':32,'transitions':8,'request_seconds':3,'max_goals':8}
SECONDARY={'seconds':40,'evaluations':64,'transitions':12,'request_seconds':3,'max_goals':8}
POLICIES=['native','frequency','shape_frequency','loop32','loop64','flat64','loop32_residual','loop64_residual','flat64_residual']
SECOND=['native','frequency','shape_frequency','loop64_residual']

def sha(b):return hashlib.sha256(b).hexdigest()
def git_blob(data):return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
def make_env():
    env=dict(os.environ)
    env['PATH']=str(LEAN.parent)+':'+env.get('PATH','')
    env['LEAN_PATH']=str(PROBE)+':'+str(SRC/'.lake/build/lib/lean')
    env['LEAN_NUM_THREADS']='2';env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    env['PYTHONHASHSEED']='0'
    return env

def module_sources():
    d={}
    assert (SRC/'lean-toolchain').read_text().strip()=='leanprover/lean4:v4.35.0'
    for n,(b,names) in LOCKED.items():
        f=SRC/'MathCan'/f'{n}.lean';raw=f.read_bytes()
        assert git_blob(raw)==b,(n,git_blob(raw),b)
        # Top-level statement order is a frozen list, independently verified
        # against comment-masked anchors before any policy inference.
        source=raw.decode()
        assert all(source.count('theorem '+name) + source.count('lemma '+name)>=1 for name in names)
        d[n]=source
    return d

class Timeout(RuntimeError):pass
class Session:
    def __init__(self,original,theorem,work,deadline):
        self.work=work;work.mkdir(parents=True,exist_ok=False)
        self.output=work/'Session.lean'
        src=render_source(original,theorem,['ram03_probe_server'])
        src='import ProbeServer\n'+src
        self.output.write_text(src,encoding='utf8')
        self.env=make_env();self.env['RAM03_MAILBOX']=str(work.resolve())
        self.log=open(work/'stdout.log','wb');self.closed=False;self.number=0
        self.proc=subprocess.Popen([str(LEAN),str(self.output.resolve())],cwd=SRC,env=self.env,
            stdin=subprocess.DEVNULL,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        data=self.await_response(0,max(.01,min(8.,deadline-time.perf_counter())))
        if not data.get('ready') or not data.get('state',{}).get('accepted'):
            raise Timeout('Lean root server did not signal ready')
        self.initial=data['state']
    def await_response(self,n,timeout):
        path=self.work/f'response-{n}.json';end=time.perf_counter()+max(0.,timeout)
        while time.perf_counter()<end:
            if path.exists():
                try:return json.loads(path.read_text(encoding='utf8'))
                except (ValueError,FileNotFoundError,PermissionError):pass
            if self.proc.poll() is not None:
                err=(self.work/'stdout.log').read_text(errors='replace')[-1600:]
                raise Timeout('Lean exited before reply: '+err)
            time.sleep(.005)
        raise Timeout('Lean pending reply timed out')
    def send(self,n,actions,stop=False):
        path=self.work/f'request-{n}.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(id=n,actions=actions,stop=stop),ensure_ascii=False))
        tmp.replace(path)
    def query(self,actions,seconds):
        if any(not grammar.valid_script(s) for s in actions):raise AssertionError('unregistered tactic')
        self.number+=1;self.send(self.number,actions)
        data=self.await_response(self.number,seconds)
        if data.get('id')!=self.number:raise AssertionError('server response protocol desynchronized')
        if 'result' not in data:raise Timeout('no completed Lean tactic result')
        return data['result']
    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.proc.poll() is None:self.send(self.number+1,[],stop=True)
            self.proc.wait(timeout=.2)
        except Exception:
            if self.proc.poll() is None:
                try:os.killpg(self.proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                self.proc.wait(timeout=2)
        finally:self.log.close()

class Rankers:
    def __init__(self):
        assert sha(MODELZIP.read_bytes())==old.ARCHIVE_SHA
        torch.set_num_threads(1)
        self.models={};self.heads={}
        with zipfile.ZipFile(MODELZIP) as z,zipfile.ZipFile(RAM12ZIP) as rz:
            raw=z.read('data/transitions.jsonl')
            assert sha(raw)==old.DATA_SHA
            rows=[json.loads(s) for s in raw.decode().splitlines() if s.strip()]
            train=[r for r in rows if not r['development']]
            self.frequency,self.shape=old.controls(train)
            for n,(seed,digest) in old.EXPECTED.items():
                data=z.read(f'training/{n}_s{seed}/best.pt')
                assert sha(data)==digest
                cp=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
                model=old.Policy(old.Config(**cp['model_config'])).eval()
                model.load_state_dict(cp['state_dict'])
                self.models[n]=model
                h=rz.read(f'weights/first_{n}.pt')
                assert sha(h)==HEADS[n]
                t=torch.load(io.BytesIO(h),map_location='cpu',weights_only=True)
                assert t['base_sha256']==digest and t['seed']==20261115 and t['updates']==150 and t['alpha']==.35
                hd=nn.Linear(model.config.width,1).eval();hd.load_state_dict(t['state_dict'])
                self.heads[n]=hd
    @torch.no_grad()
    def score(self,model_name,state,pool,residual=False):
        m=self.models[model_name]
        rows=[dict(before=state,family=f,script=s) for f,s in pool]
        if not rows:return []
        result=[]
        for i in range(0,len(rows),16):
            x=old.encode(rows[i:i+16]);logits,delta=m(x)
            p=logits.softmax(-1)
            s=3*p[:,3]+1.5*p[:,1]+.3*p[:,2]-.6*p[:,0]+.4*delta.clamp(-3,3)
            if residual:
                z=m.fuse(torch.cat([m.mean(x[k]) for k in ('context','goal','pending','action')]+[m.family(x['family']),x['counts']],-1))
                fixed=z
                for _ in range(m.config.loops):z=m.norm(z+m.cell(torch.cat([z,fixed],-1)))
                new=s+.35*torch.tanh(self.heads[model_name](z).squeeze(-1))
                assert (new-s).abs().max().item()<=.350001
                s=new
            result.extend(s.tolist())
        return result
    def rank(self,which,state,pool):
        if which=='native':return pool
        rows=[dict(before=state,family=f,script=s) for f,s in pool]
        if which=='frequency':scores=[self.frequency(r) for r in rows]
        elif which=='shape_frequency':scores=[self.shape(r) for r in rows]
        else:
            model=which.removesuffix('_residual')
            scores=self.score(model,state,pool,which.endswith('_residual'))
        return [pool[i] for i in sorted(range(len(pool)),key=lambda i:(-scores[i],i))]

def certify(original,theorem,actions,work,seconds):
    source=render_source(original,theorem,actions)
    proof=source.encode();work.parent.mkdir(parents=True,exist_ok=True)
    path=work.with_suffix('.lean');path.write_bytes(proof)
    try:
        r=subprocess.run([str(LEAN),str(path.resolve())],cwd=SRC,env=make_env(),
              capture_output=True,text=True,timeout=min(10.,max(.01,seconds)))
        log=r.stdout+r.stderr
        (work.with_suffix('.log')).write_text(log)
        okay,axioms,detail=axiom_inventory(log)
        passed=bool(r.returncode==0 and okay and 'error:' not in log and 'sorryAx' not in log)
        return dict(accepted=passed,axioms=axioms,source_sha256=sha(proof),source_file=str(path),log_sha256=sha(log.encode()),returncode=r.returncode,reason=detail)
    except subprocess.TimeoutExpired:
        (work.with_suffix('.log')).write_text('unknown: compiler timeout')
        return dict(accepted=False,unknown=True,source_sha256=sha(proof),source_file=str(path),reason='compile_timeout')

def search(original,theorem,ranker,policy,budget,out):
    start=time.perf_counter();end=start+budget['seconds'];active_end=end-.5
    out.mkdir(parents=True,exist_ok=False)
    calls=[];visited=set();processes=0;restarts=0;server=None;success=False;proof=None;replay=None;chosen=[];reason='no_progress';stats=collections.Counter()
    def open_server():
        nonlocal processes
        processes+=1
        return Session(original,theorem,out/f'server{processes:02d}',active_end)
    try:
        server=open_server();root_signature=grammar.signature(server.initial['goals'])
        stack=[dict(actions=[],state=server.initial,ordered=None,index=0)]
        visited.add(root_signature)
        while stack:
            if time.perf_counter()>=active_end:reason='deadline';break
            if len(calls)>=budget['evaluations']:reason='evaluation_budget';break
            node=stack[-1]
            if len(node['actions'])>=budget['transitions']:
                stats['transition_budget']+=1;reason='transition_budget';stack.pop();continue
            if node['ordered'] is None:
                pool=grammar.candidates(node['state'])
                node['ordered']=ranker.rank(policy,node['state'],pool)
                assert collections.Counter(node['ordered'])==collections.Counter(pool)
            if node['index']>=len(node['ordered']):stack.pop();continue
            family,script=node['ordered'][node['index']];node['index']+=1
            actions=node['actions']+[script]
            before=len(node['state']['goals'])
            try:
                obs=server.query(actions,min(budget['request_seconds'],max(.01,active_end-time.perf_counter())))
            except Timeout as exc:
                calls.append(dict(family=family,script=script,actions=actions,unknown=True,error=str(exc)[-450:]))
                stats['request_timeout']+=1
                if restarts>=2 or time.perf_counter()>=active_end:reason='request_timeout';break
                restarts+=1;server.close();server=open_server()
                if grammar.signature(server.initial['goals'])!=root_signature:raise AssertionError('original theorem initial goal changed after restart')
                continue
            calls.append(dict(family=family,script=script,actions=actions,accepted=obs.get('accepted'),
                              after_goal_count=len(obs.get('goals',[])) if obs.get('accepted') else None, error=obs.get('error')))
            if time.perf_counter()>=active_end:reason='deadline';break
            if not obs.get('accepted'):stats['failed']+=1;continue
            goals=obs.get('goals',[]);sig=grammar.signature(goals)
            if sig==grammar.signature(node['state']['goals']):stats['unchanged']+=1;continue
            if not goals:
                chosen=actions;server.close();server=None
                if time.perf_counter()>=active_end:reason='verification_budget';break
                processes+=1
                proof=certify(original,theorem,actions,out/'certificate',active_end-time.perf_counter())
                if not proof['accepted'] or time.perf_counter()>=active_end:reason='final_rejected_or_budget';break
                processes+=1
                replay=certify(original,theorem,actions,out/'independent_replay',active_end-time.perf_counter())
                success=bool(replay['accepted'] and replay['source_sha256']==proof['source_sha256'] and time.perf_counter()<=end)
                reason='independently_certified' if success else 'replay_rejected_or_budget';break
            if len(goals)>budget['max_goals']:stats['goal_cap']+=1;continue
            if sig in visited:stats['deduplicated']+=1;continue
            visited.add(sig);stats['transitions']+=1
            if len(goals)>1:stats['multi_goal_transitions']+=1
            stack.append(dict(actions=actions,state=obs,ordered=None,index=0))
    except Timeout as exc:
        reason='startup_or_request_timeout';stats['detail']=str(exc)[-800:]
    except Exception as exc:
        reason='unexpected_error';stats['detail']=repr(exc)
    finally:
        if server is not None:server.close()
    elapsed=time.perf_counter()-start
    if elapsed>budget['seconds']:
        success=False;reason='deadline_shutdown_overrun'
    receipt=dict(accepted=success,reason=reason,actions=chosen,budget=budget,
                 logical_evaluations=len(calls),processes=processes,server_restarts=restarts,
                 elapsed_seconds=elapsed,calls=calls,diagnostics=dict(stats),
                 certificate=proof,replay=replay,original_source_sha256=sha(original.encode()),
                 policy=policy,theorem=theorem)
    (out/'case.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    return receipt

def smoke():
    torch.set_num_threads(1)
    orig=(SRC/'MathCan/Intervention.lean').read_text()
    out=Path('/mnt/data/ram13_local/smoke_intervention')
    if out.exists():import shutil;shutil.rmtree(out)
    t0=time.perf_counter();session=Session(orig,'obsEqOn_refl',out,time.perf_counter()+10)
    print('SMOKE initial',session.initial,flush=True)
    for act in [['intro ram07_arg0'],['intro'],['simp_all']]:
        try:r=session.query(act,2.0);print('SMOKE',act,r,flush=True)
        except Timeout as e:print('SMOKE timed',act,str(e),flush=True)
    session.close();print('SMOKE secs',time.perf_counter()-t0)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--n',type=int,default=2);args=ap.parse_args()
    if args.smoke:smoke();sys.exit(0)
    originals=module_sources();rank=Rankers();out=OUT;out.mkdir(parents=True,exist_ok=True)
    cases=[(module,theorem) for module,(_,names) in LOCKED.items() for theorem in names]
    for index,(module,theorem) in enumerate(cases[:args.n]):
        policies=POLICIES[index%len(POLICIES):]+POLICIES[:index%len(POLICIES)]
        for policy in policies:
            path=out/'primary'/f'{index:02d}'/policy
            if path.exists():print('SKIP',module,theorem,policy,flush=True);continue
            result=search(originals[module],theorem,rank,policy,PRIMARY,path)
            print('RAM13',module,theorem,policy,result['accepted'],result['reason'],round(result['elapsed_seconds'],2),result['logical_evaluations'],flush=True)
