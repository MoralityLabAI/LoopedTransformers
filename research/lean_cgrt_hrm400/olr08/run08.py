"""OLR08 staged CPU native search gym. No post-command candidate execution.
Use launch.py to apply resource limits. See RUN.md for the prospective gates.
"""
from __future__ import annotations
import argparse, collections, hashlib, heapq, itertools, json, math, os, pathlib, queue, random, re, subprocess, sys, threading, time, zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import probe
from search08 import actions,reprice,FREE,RANKED

HOME=Path(__file__).resolve().parent
SCOPE='pre_command_session_v1'; COMMIT='6caaee842e9495688c1567e78c0e68dbb96942aa'
SEED=20270129; STEP_CAP=300; RPC_CAP=1200; HB_CAP=512_000_000; DEPTH=3
REPLAY_WARM_HB=20000
# New files, no SInt. Calibration and fresh evaluation use distinct files;
# their families were held out from the reused referee TRAIN cohort.
FRESH={ 'calibration':{'Int':['Cooper','OfNat'],'Array':['BinSearch','FinRange'],'Fin':['Iterate']},
        'evaluation':{'BitVec':['Basic','BasicAux','Lemmas','Bitblast','Folds'],'Array':['DecidableEq','GetLit','InsertionSort','Lex','Monadic','QSort','Range','Set','Subarray'],'Vector':['Attach','Count','DecidableEq','FinRange','InsertIdx','Lex','MapIdx','Monadic','OfFn','Range','Zip'],'UInt':['BasicAux','Log2']} }
# BitVec is family-new. Array/Vector/Fin/UInt have prior spent files, so report new-file and new-family separately.

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):probe.write(p,x)
def close_cert(r):return r.get('kernel') is True and isinstance(r.get('axioms'),list) and set(r['axioms'])<=probe.AXIOMS

class Lean:
    def __init__(self,exe,timeout=900):
        self.p=subprocess.Popen([str(exe),'Lean'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',bufsize=1)
        self.q=queue.Queue();self.timeout=timeout
        def reader():
            for line in self.p.stdout:self.q.put(line)
            self.q.put(None)
        threading.Thread(target=reader,daemon=True).start()
        try:
            r=self.recv()
            if not r.get('ready') or r.get('lean')!='4.19.0' or r.get('scope_version')!=SCOPE:raise ValueError('unsafe or incompatible server '+str(r))
        except BaseException:self.close();raise
    def recv(self):
        try:line=self.q.get(timeout=self.timeout)
        except queue.Empty:self.close();raise TimeoutError('native request timeout')
        if line is None:raise RuntimeError('native process ended')
        return json.loads(line)
    def request(self,**kw):
        self.p.stdin.write(json.dumps(kw,ensure_ascii=False)+'\n');self.p.stdin.flush();return self.recv()
    def close(self):
        if self.p.poll() is None:
            try:self.p.stdin.close();self.p.wait(timeout=3)
            except Exception:self.p.kill();self.p.wait()
    def __enter__(self):return self
    def __exit__(self,*e):self.close()

class Exhausted(Exception):pass
class Meter:
    """Each native RPC is charged, as are attempted steps. Per-root open/replay
    recorded separately. In-flight call may exceed cap; no credit beyond cap."""
    def __init__(self,L,steps=STEP_CAP,rpcs=RPC_CAP,hb=HB_CAP):
        self.L=L;self.limits=(steps,rpcs,hb);self.ops=Counter();self.hb=0;self.ms=0;self.seconds=0.;self.feature_seconds=0.;self.inference_seconds=0.;self.journal=[]
    def exhausted(self):return self.ops['step']>=self.limits[0] or sum(self.ops.values())>=self.limits[1] or self.hb>=self.limits[2]
    def call(self,op,**kw):
        if sum(self.ops.values())>=self.limits[1] or self.hb>=self.limits[2] or (op=='step' and self.ops['step']>=self.limits[0]):raise Exhausted()
        t=time.perf_counter();r=self.L.request(op=op,**kw);secs=time.perf_counter()-t
        if not isinstance(r.get('request_hb_raw'),int) or not isinstance(r.get('request_ms'),int):raise ValueError('native metering missing')
        self.ops[op]+=1;self.hb+=r['request_hb_raw'];self.ms+=r['request_ms'];self.seconds+=secs
        self.journal.append({'op':op,'hb':r['request_hb_raw'],'server_ms':r['request_ms'],'wall_seconds':secs,'ok':r.get('ok')})
        if self.hb>self.limits[2]:raise Exhausted()
        return r
    def report(self):return {'ops':dict(self.ops),'steps':self.ops['step'],'rpcs':sum(self.ops.values()),'hb_raw':self.hb,'server_ms':self.ms,'native_seconds':self.seconds,'feature_seconds':self.feature_seconds,'inference_seconds':self.inference_seconds,'limits':self.limits,'rpc_journal':self.journal}

def open_root(L,source,row):
    t=time.perf_counter();r=L.request(op='open',path=str((source/row['file']).resolve()),positions=[[row['line'],row['col']]])
    cost={'wall_seconds':time.perf_counter()-t,'hb_raw':r.get('request_hb_raw'),'server_ms':r.get('request_ms')}
    if r.get('ok') is not True or r.get('errors'):raise ValueError('source elaboration failed '+str(r)[:300])
    states=r.get('states',[])
    if len(states)!=1:return None,'unreached',cost
    s=states[0]
    if s.get('env')!='before':
        if 'sid' in s:raise ValueError('unsafe later-environment state allocated')
        return None,s.get('reason','scope_unavailable'),cost
    if s.get('scope_version')!=SCOPE or s.get('decl_in_env') is not False:raise ValueError('unsafe scope receipt')
    if s.get('decl')!=row['decl'] or s.get('goals')!=[row['goal']] or s.get('source_goals')!=1:return None,'source_state_mismatch',cost
    if not(s.get('main_prop') and s.get('main_closed')):return None,'uncertifiable',cost
    return s,None,cost

def sentinel(exe,out):
    p=out/'scope_sentinel.lean';out.mkdir(parents=True,exist_ok=True)
    p.write_text('import Lean\nnamespace ScopeSentinel\ntheorem earlier (n : Nat) : n + 0 = n := Nat.add_zero n\n@[simp] theorem current (n : Nat) : n + 0 = n := by\n  exact earlier n\ntheorem later (n : Nat) : n + 0 = n := earlier n\nend ScopeSentinel\n',encoding='utf-8')
    with Lean(exe) as L:
        trace=out/'trace.jsonl';r=L.request(op='trace',path=str(p.resolve()),out=str(trace.resolve()),cand_every=0)
        if not r.get('ok') or r.get('errors'):raise ValueError('sentinel trace failed')
        row=next(r for r in probe.read_lines(trace) if r['decl']=='ScopeSentinel.current' and r['tactic'].strip()=='exact earlier n')
        row={'file':p.name,'line':row['line'],'col':row['col'],'decl':row['decl'],'goal':row['goals_before'][0]}
        root,why,_=open_root(L,out,row)
        if root is None:raise ValueError('sentinel inaccessible '+str(why))
        m=Meter(L);s=root['sid'];res=[]
        for tac in ('exact ScopeSentinel.earlier n','exact ScopeSentinel.current n','exact ScopeSentinel.later n','simp only [ScopeSentinel.current]'):
            r=m.call('step',sid=s,tactic=tac,hb=500);res.append(r)
        def accepted(r):
            if r.get('ok') is not True:return False
            if r.get('ngoals')!=0:return False
            cert=m.call('cert',sid=r['sid']).get('cert',{})
            r['sentinel_certificate']=cert
            return close_cert(cert)
        verdicts=[accepted(r) for r in res]
        if verdicts!=[True,False,False,False] or res[2].get('ok') is True:raise ValueError('native sentinel leakage')
        for r in (res[1],res[3]):
            if r.get('ok') is True and r.get('ngoals')==0 and r.get('sentinel_certificate',{}).get('kernel') is not False:raise ValueError('uncertified self closure lacks explicit negative kernel verdict')
        graph=m.call('graph',sid=s);prem=m.call('premises',sid=s,max_rw=40,max_apply=10)
        if graph.get('ok') is not True or 'nodes' not in graph or not all(k in prem for k in ('rw','apply')):raise ValueError('native graph/premise interface failed')
        # Ensure legacy post-command candidate endpoint is disabled.
        legacy=L.request(op='trace',path=str(p.resolve()),out=str((out/'forbidden.jsonl').resolve()),cand_every=1,candidates=['exact?'])
        if legacy.get('ok') is True:raise ValueError('legacy endpoint still enabled')
        write(out/'PASS.json',{'scope':SCOPE,'results':res,'meter':m.report()})

def freeze(source,exe,out,profile,workers):
    train=read(HOME/'train_registry.json');spent=read(HOME/'spent.json');seen=set(spent['goals_sha256']);decls=set(spent['decls'])
    # Source selection is deterministic and outcome independent. Re-certify every selected row.
    train=sorted(train,key=lambda r:digest([SEED,r['id']]))[:(8 if profile=='smoke' else 192)]
    roster=[(s,f,f'src/Init/Data/{f}/{n}.lean') for s,fams in FRESH.items() for f,names in fams.items() for n in names]
    if profile=='smoke':roster=[next(x for x in roster if x[0]==s) for s in FRESH]
    if any(p in spent['files'] or fam=='SInt' for _,fam,p in roster):raise ValueError('spent or reserved file in roster')
    def work(item):
        role,fam,rel=item;d=out/'trace'/probe.key(fam,rel);d.mkdir(parents=True,exist_ok=True);path=source/rel
        with Lean(exe) as L:
            meta=L.request(op='trace',path=str(path.resolve()),out=str((d/'trace.jsonl').resolve()),cand_every=0)
        if not meta.get('ok') or meta.get('errors'):raise ValueError('trace error '+rel)
        write(d/'report.json',meta);return item,d
    rows=list(train);cap=1 if profile=='smoke' else 12;trace_census=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (role,fam,rel),d in pool.map(work,roster):
            rep=read(d/'report.json');valid={r['decl'] for r in rep['decls'] if r['in_env'] and set(r['axioms'])<=probe.AXIOMS and r['decl'] not in ('','_anonymous','[anonymous]')}
            allrows=probe.rows_of(probe.read_lines(d/'trace.jsonl'));by=defaultdict(list)
            for r in allrows:by[r['decl']].append(r)
            candidates=[]
            for decl,rs in by.items():
                if decl not in valid or decl in decls:continue
                first=min(rs,key=lambda r:(r['line'],r['col']))
                for r in rs:
                    text=r.get('tactic','');gs=r.get('goals_before',[])
                    if len(gs)!=1 or not text or '…' in text or '…' in gs[0] or probe.FORBIDDEN.search(text) or re.match(r'\s*(decreasing_by|termination_by|where)\b',text):continue
                    kind='T' if r.get('goals_after')==[] else ('R' if r is first else None)
                    if not kind:continue
                    g=gs[0];ident=digest([SEED,rel,decl,r['line'],r['col'],g]);candidates.append({'id':ident,'set':kind,'role':role,'family':fam,'file':rel,'decl':decl,'line':r['line'],'col':r['col'],'goal':g,'human':probe.human_text(r),'decl_rows':len(rs)})
            selected=[]
            # Prefer a balanced registered T/R roster, without selecting by menu success.
            for kind in ('T','R'):
                for r in sorted((v for v in candidates if v['set']==kind),key=lambda r:r['id']):
                    if sum(v['set']==kind for v in selected)>=(cap if kind=='T' else max(1,cap//2)):break
                    h=hashlib.sha256(r['goal'].encode()).hexdigest()
                    if h in seen or r['decl'] in decls:continue
                    seen.add(h);decls.add(r['decl']);selected.append(r)
            rows.extend(selected);trace_census.append({'file':rel,'selected':len(selected),'candidates':len(candidates)})
    for r in rows:r['file_sha256']=sha(source/r['file'])
    write(out/'registry.json',rows);write(out/'registry_census.json',trace_census);return rows

def menu_once(exe,source,row):
    with Lean(exe) as L:
        s,why,opening=open_root(L,source,row)
        if not s:return {'id':row['id'],'scored':False,'reason':why,'open':opening}
        m=Meter(L,steps=20,rpcs=50,hb=HB_CAP);receipts=[];human=None
        for tac in (*probe.MENU,*([row['human']] if row['set']=='T' else [])):
            a=m.call('step',sid=s['sid'],tactic=tac,hb=20000 if len(receipts)==len(probe.MENU) else 500)
            c=close_cert(m.call('cert',sid=a['sid']).get('cert',{})) if a.get('ok') is True and a.get('ngoals')==0 else False
            r={'tactic':tac,'closed':c,'ok':a.get('ok'),'ngoals':a.get('ngoals'),'error':a.get('error')}
            if len(receipts)<len(probe.MENU):receipts.append(r)
            else:human=r
        return {'id':row['id'],'scored':True,'human':human,'menu':receipts,'any':any(r['closed'] for r in receipts),'open':opening,'meter':m.report()}

def menu(exe,source,row):
    a=menu_once(exe,source,row);b=menu_once(exe,source,row)
    def sig(r):
        if not r['scored']:return (False,r['reason'])
        def receipt(v):
            if v is None:return None
            x={'ok':v['ok'],'goals_after':v['ngoals'],'error':v.get('error') or ''}
            return (v['closed'], 'closed' if v['closed'] else probe.category(x))
        return (True,[receipt(v) for v in r['menu']],receipt(r['human']))
    if sig(a)!=sig(b):raise ValueError('native menu replay drift '+row['id'])
    a['replay']=b;return a

def enumerate_menu(m,s):
    p=m.call('premises',sid=s['sid'],max_rw=40,max_apply=10)
    if p.get('error') or not all(k in p for k in ('rw','apply')):raise ValueError('premise enumeration failed: '+str(p)[:300])
    # De-duplicate literal tactic strings only, never proof states.
    seen=set();out=[]
    for a in actions(s['info'],p):
        if a[-1] not in seen:seen.add(a[-1]);out.append(a)
    return out

def get_packet(m,s,acts,depth,prior):
    import reader
    g=m.call('graph',sid=s['sid']);names=sorted({arg for _,k,arg,_ in acts if k.endswith('_lemma') or k=='rw_lemma_rev' or k in ('simp_def','unfold')})
    l=m.call('lemma_graph',sid=s['sid'],names=names)
    if g.get('ok') is not True or l.get('ok') is not True or not isinstance(l.get('lemmas'),list) or not all(k in g for k in ('nodes','edges','obligations')):return None
    t=time.perf_counter();p=reader.packet(g,l['lemmas'],acts,depth,[a[0] for a in prior]);m.feature_seconds+=time.perf_counter()-t
    return p

class Search:
    """A branch is a live Lean snapshot. No printed or lossy graph dedup."""
    def __init__(self,m,root,arm,prior=None,model=None,alpha=0.):
        self.m=m;self.arm=arm;self.prior=prior;self.model=model;self.alpha=alpha;self.heap=[];self.serial=0;self.done=False;self.path=None;self.events=[];self.fallbacks=0;self.expand(root,0,0.,(),())
    def expand(self,s,depth,g,path,event_ids):
        acts=enumerate_menu(self.m,s)
        costs=reprice(acts,self.prior,depth) if self.prior is not None else acts
        if self.arm=='s1':costs=[(1.,*a[1:]) for a in acts]
        if self.model is not None and self.alpha:
            import reader
            p=get_packet(self.m,s,acts,depth,costs)
            if p is None:self.fallbacks+=1
            else:
                t=time.perf_counter();cs=reader.costs(self.model,p,self.alpha);self.m.inference_seconds+=time.perf_counter()-t
                costs=[(c,*a[1:]) for c,a in zip(cs,acts)]
        ranks=Counter()
        for c,k,arg,tac in costs:
            rank=ranks[k];ranks[k]+=1;extra=0.
            if self.arm=='s3c':extra=math.log2(max(1.,self.prior['hb'].get(tac if k in FREE else k,self.prior['hb_mean'])))
            self.serial+=1;heapq.heappush(self.heap,(g+c+extra,self.serial,s['sid'],depth,g+c,path,event_ids,k,arg,tac,rank))
    def step(self):
        if self.done:return
        if not self.heap:self.done=True;return
        _,_,sid,d,g,path,ids,k,arg,tac,rank=heapq.heappop(self.heap)
        a=self.m.call('step',sid=sid,tactic=tac,hb=500 if tac=='exact?' else 200)
        i=len(self.events);event={'kind':k,'argument':arg,'tactic':tac,'rank':rank,'depth':d,'ok':a.get('ok') is True,'ngoals':a.get('ngoals'),'hb_raw':a['request_hb_raw'],'parent_events':list(ids)};self.events.append(event)
        if not event['ok']:return
        path=(*path,tac);ids=(*ids,i)
        if a['ngoals']==0:
            c=self.m.call('cert',sid=a['sid']).get('cert',{})
            if close_cert(c):self.done=True;self.path=list(path);self.on_path=list(ids);self.cert=c
        elif d+1<DEPTH:self.expand(a,d+1,g,path,ids)

def replay(exe,source,row,path,prior=None,frames=False):
    with Lean(exe) as L:
        s,why,opening=open_root(L,source,row)
        if s is None:raise ValueError('fresh replay root missing '+str(why))
        m=Meter(L,steps=DEPTH+1,rpcs=100,hb=HB_CAP);data=[];warm=[]
        for d,tac in enumerate(path):
            if frames:
                acts=enumerate_menu(m,s);priced=reprice(acts,prior,d)
                if tac not in [x[-1] for x in acts]:raise ValueError('replay action missing from enumerator')
                p=get_packet(m,s,acts,d,priced)
                if p is not None:data.append((p,[x[-1] for x in acts].index(tac)))
            if tac=='exact?':
                # Infrastructure repair: exact? builds Lean's library-search index as a process-wide cache,
                # paid in search by earlier charged attempts. Warm it on this state, uncharged and recorded,
                # so the replayed step runs under the same 500 cap it met in search.
                w=L.request(op='step',sid=s['sid'],tactic=tac,hb=REPLAY_WARM_HB)
                warm.append({'depth':d,'ok':w.get('ok'),'hb_raw':w.get('request_hb_raw'),'server_ms':w.get('request_ms')})
            s=m.call('step',sid=s['sid'],tactic=tac,hb=500 if tac=='exact?' else 200)
            if s.get('ok') is not True:raise ValueError('fresh native path replay failed')
        if s.get('ngoals')!=0 or not close_cert(m.call('cert',sid=s['sid']).get('cert',{})):raise ValueError('fresh native certification failed')
        return data,{'open':opening,'meter':m.report(),'library_search_warmup':warm}

def search(exe,source,row,arm,prior=None,model=None,alpha=0.,limit=STEP_CAP):
    start=time.perf_counter()
    with Lean(exe) as L:
        s,why,opening=open_root(L,source,row)
        if s is None:raise ValueError('previous scored root lost '+str(why))
        m=Meter(L,steps=limit,rpcs=RPC_CAP*(4 if limit>STEP_CAP else 1),hb=HB_CAP*(4 if limit>STEP_CAP else 1));qs=[];winner=None
        try:
            # Matched total budget: alternate one verifier action between the
            # exact counted queue and a learned queue. Both use same native root.
            if model is not None and alpha:
                qs.append(Search(m,s,'s3',prior));qs.append(Search(m,s,'learned',prior,model,alpha))
            else:qs.append(Search(m,s,arm,prior))
            n=0
            while not all(q.done for q in qs) and not m.exhausted():
                q=qs[n%len(qs)];n+=1
                if q.done:continue
                q.step()
                if q.path is not None:winner=q;break
        except Exhausted:pass
        work=m.report();result={'id':row['id'],'role':row['role'],'family':row['family'],'file':row['file'],'arm':arm,'solved':winner is not None,'path':winner.path if winner else [],'on_path':winner.on_path if winner else [],'events':winner.events if winner else (qs[0].events if qs else []),'queues':[{'arm':q.arm,'events':q.events,'fallbacks':q.fallbacks} for q in qs],'open':opening,'meter':work,'steps':work['steps'],'elapsed':time.perf_counter()-start}
    if winner:
        _,audit=replay(exe,source,row,winner.path);result['replay']=audit
    return result

def fit_prior(rows):
    """Counts exact event ids on certified source paths. No fuzzy tactic matching."""
    n=Counter();yes=Counter();rk=Counter();ry=Counter();hb=Counter();allhb=[]
    for r in rows:
        if r['role']!='train':raise ValueError('non-source prior')
        on=set(r['on_path'])
        for j,e in enumerate(r['events']):
            k=e['kind'];t=e['tactic'] if k in FREE else k;key=f"{min(e['depth'],1)}|{t}"
            n[key]+=1;yes[key]+=j in on;hb[t]+=e['hb_raw'];allhb.append(e['hb_raw'])
            if k in RANKED:b=min(e['rank']//4,5);rk[b]+=1;ry[b]+=j in on
    tn=Counter(e['tactic'] if e['kind'] in FREE else e['kind'] for r in rows for e in r['events'])
    return {'weight':{k:(yes[k]+.5)/(v+1) for k,v in n.items()},'rank':[-math.log2((ry[b]+1)/(rk[b]+2)) for b in range(6)],'hb':{k:max(1.,v/tn[k]) for k,v in hb.items()},'hb_mean':max(1.,sum(allhb)/max(1,len(allhb))),'source_roots':len(rows)}

def stage(out,name,rows,fn,workers=1):
    d=out/name;d.mkdir(parents=True,exist_ok=True)
    def work(row):
        p=d/(row['id']+'.json');seal=p.with_suffix('.sha256')
        if p.exists() and seal.exists():
            if seal.read_text().strip()!=sha(p):raise ValueError('checkpoint hash mismatch '+str(p))
            v=read(p)
            if v['id']!=row['id']:raise ValueError('checkpoint root mismatch')
            return v
        v=fn(row);write(p,v);seal.write_text(sha(p)+'\n');return v
    with ThreadPoolExecutor(max_workers=workers) as pool:
        result=[]
        for i,v in enumerate(pool.map(work,rows)):
            result.append(v)
            if (i+1)%8==0 or i+1==len(rows):print(name,i+1,len(rows),flush=True)
    return result

def rate(rows):return {'roots':len(rows),'solved':sum(r['solved'] for r in rows),'rate':sum(r['solved'] for r in rows)/max(1,len(rows)),'mean_steps':sum(r['steps'] for r in rows)/max(1,len(rows)),'mean_hb_raw':sum(r['meter']['hb_raw'] for r in rows)/max(1,len(rows)),'mean_elapsed':sum(r['elapsed'] for r in rows)/max(1,len(rows))}

def compare(a,b):
    ai={r['id']:r for r in a};bi={r['id']:r for r in b}
    if ai.keys()!=bi.keys():raise ValueError('unpaired rows')
    diffs=defaultdict(list)
    for k,r in ai.items():diffs[r['file']].append(int(r['solved'])-int(bi[k]['solved']))
    fs=sorted(diffs);rng=random.Random(SEED);v=[]
    if not fs:return None
    for _ in range(2000):
        x=[d for f in rng.choices(fs,k=len(fs)) for d in diffs[f]];v.append(sum(x)/len(x))
    v.sort();return {'diff':(sum(r['solved'] for r in a)-sum(r['solved'] for r in b))/len(a),'ci95':[v[50],v[1949]],'files':len(fs)}

def main(args):
    import reader,torch
    torch.set_num_threads(args.threads);torch.backends.mkldnn.enabled=False;torch.use_deterministic_algorithms(True)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True);source=args.source.resolve();exe=args.exe.resolve()
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=COMMIT or subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('pinned clean Lean source checkout required')
    # Exact shared-tracer reissue prerequisite, supplied manifests from both CI jobs.
    prereq={}
    if args.profile=='full':
        for k,p in [('07a',args.reissue_a),('07b',args.reissue_b)]:
            if p is None:raise ValueError('full profile requires both exact-registry reissue manifests')
            r=read(p)
            if r.get('scope_version')!='pre_command_v1' or r.get('kind')!=k or not r.get('old_receipts_superseded') or not r.get('files'):raise ValueError('invalid reissue manifest '+k)
            
            expected={'07a':207,'07b':1441}[k]
            if sum(v['selected'] for v in r.get('census',{}).values())!=expected:raise ValueError('reissue census mismatch '+k)
            for n,h in r['files'].items():
                f=p.parent/n
                if not f.is_file() or sha(f)!=h:raise ValueError('reissue artifact hash mismatch '+n)
            prereq[k]=sha(p)
    contract={'schema':'olr08a.cpu.v1','profile':args.profile,'source':COMMIT,'exe_sha256':sha(exe),'torch':torch.__version__,'threads':args.threads,'code':{p.name:sha(p) for p in HOME.glob('*.py')},'data':{n:sha(HOME/n) for n in ('train_registry.json','spent.json')},'reissues':prereq,'caps':[STEP_CAP,RPC_CAP,HB_CAP,DEPTH]}
    p=out/'contract.json'
    if p.exists() and read(p)!=contract:raise ValueError('resume contract differs')
    write(p,contract);sentinel(exe,out/'sentinel')
    reg=out/'registry.json';seal=out/'registry.sha256'
    if reg.exists() and seal.exists():
        if seal.read_text().strip()!=sha(reg):raise ValueError('registry hash mismatch')
        rows=read(reg)
    else:
        rows=freeze(source,exe,out,args.profile,args.workers);seal.write_text(sha(reg)+'\n')
    for r in rows:
        if sha(source/r['file'])!=r['file_sha256']:raise ValueError('source changed')
    # Train/calibration receipts before ANY evaluation menu/search scoring.
    sc=[r for r in rows if r['role']!='evaluation'];ev=[r for r in rows if r['role']=='evaluation']
    receipts=stage(out,'menu_source',sc,lambda r:menu(exe,source,r),args.workers);idx={r['id']:r for r in receipts}
    def eligible(rs,idx):return [r for r in rs if idx[r['id']]['scored'] and not idx[r['id']]['any']]
    def validity(rs,ix):
        x=[ix[r['id']] for r in rs if ix[r['id']]['scored']];human=[v['human'] for v in x if v['human'] is not None]
        return {'selected':len(rs),'scored':len(x),'T_human':len(human),'human_closed':sum(v['closed'] for v in human),'human_rate':sum(v['closed'] for v in human)/max(1,len(human))}
    tr=eligible([r for r in sc if r['role']=='train'],idx);cal=eligible([r for r in sc if r['role']=='calibration'],idx)
    tv=validity([r for r in sc if r['role']=='train'],idx);cv=validity([r for r in sc if r['role']=='calibration'],idx)
    if args.profile=='full' and (tv['human_rate']<.95 or cv['human_rate']<.95 or tv['T_human']<20 or cv['T_human']<10):
        result={'schema':'olr08a.cpu.result.v1','profile':args.profile,'source_validity':tv,'calibration_validity':cv,'status':'source_validity_failed','architecture_promotion_eligible':False};write(out/'summary.json',result);return result
    teacher=stage(out,'source_s2',tr,lambda r:search(exe,source,r,'s2'),args.workers);prior=fit_prior(teacher);write(out/'prior.json',prior)
    cal_runs={a:stage(out,'cal_'+a,cal,lambda r,a=a:search(exe,source,r,a,prior if a in ('s3','s3c') else None),args.workers) for a in ('s1','s2','s3','s3c')}
    # Longer same prior search is a privileged measured headroom reference,
    # never used as a deployed 300-action policy.
    long=stage(out,'cal_s3_long',cal,lambda r:search(exe,source,r,'s3',prior,limit=1200),args.workers) if args.profile=='full' else cal_runs['s3']
    best=max((sum(x['solved'] for x in v) for v in cal_runs.values()),default=0);longcount=sum(x['solved'] for x in long)
    gate=args.profile=='full' and tv['human_rate']>=.95 and cv['human_rate']>=.95 and len(tr)>=40 and len(cal)>=24 and best<=.8*len(cal) and longcount-best>=.1*len(cal)
    frozen={'source_validity':tv,'calibration_validity':cv,'train_hard':len(tr),'cal_hard':len(cal),'calibration':{k:rate(v) for k,v in cal_runs.items()},'cal_long':rate(long),'headroom_gate':gate,'source_prior_sha256':sha(out/'prior.json'),'models':{}}
    # Gate 2: conditional source-only learner; never train on calibration/evaluation outcomes.
    models={};frame_meter=[]
    if gate:
        tridx={r['id']:r for r in tr};frames=[]
        fp=out/'source_frames.pt';fm=out/'source_frames_manifest.json'
        if fp.exists() and fm.exists():
            meta=read(fm)
            if sha(fp)!=meta['sha256'] or meta['prior_sha256']!=sha(out/'prior.json') or meta['teacher_sha256']!=digest(teacher):raise ValueError('source frame checkpoint changed')
            frames=torch.load(fp,map_location='cpu',weights_only=True)
        else:
            for r in teacher:
                if r['solved']:
                    v,m=replay(exe,source,tridx[r['id']],r['path'],prior,frames=True);frames.extend(v);frame_meter.append({'id':r['id'],'path':r['path'],'frames':len(v),**m})
            write(out/'frame_meter.json',frame_meter);tmp=fp.with_suffix('.part');torch.save(frames,tmp);os.replace(tmp,fp)
            write(fm,{'sha256':sha(fp),'prior_sha256':sha(out/'prior.json'),'teacher_sha256':digest(teacher),'frames':len(frames)})
        frozen['source_frames']=len(frames)
        if len(frames)>=64:
            fd=out/'fits';fd.mkdir(exist_ok=True)
            for seed in reader.SEEDS:
                signatures=[]
                for arm in reader.ARMS:
                    key=f'{seed}_{arm}';p=fd/(key+'.pt');meta=fd/(key+'.json')
                    if p.exists() and meta.exists():
                        stats=read(meta);m=reader.Reader(arm);m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True));m.eval()
                        if reader.hash_weights(m)!=stats['weights_sha256']:raise ValueError('fit weight hash changed')
                    else:
                        m,stats=reader.fit(frames,arm,seed);torch.save(m.state_dict(),p);write(meta,stats)
                    if arm!='linear':signatures.append((stats['initial_sha256'],stats['sampler_sha256']))
                    # Compare alternating learned and exact queues, then freeze
                    # alpha=0 or 1 on calibration (ties prefer prior).
                    rr=stage(out,'cal_model_'+key,cal,lambda row:search(exe,source,row,'s3',prior,m,1.),1)
                    base=cal_runs['s3'];alpha=float(sum(r['solved'] for r in rr)>sum(r['solved'] for r in base))
                    models[key]=(m,alpha);frozen['models'][key]={'fit':stats,'alpha':alpha,'calibration':rate(rr)}
                if len(set(signatures))!=1:raise ValueError('matched initialization or sample drift')
    f=out/'frozen_policy.json'
    if f.exists() and read(f)!=frozen:raise ValueError('frozen policy identity changed')
    write(f,frozen)
    # Fresh outcome collection starts here, after all source fits/routing freeze.
    er=stage(out,'menu_fresh',ev,lambda r:menu(exe,source,r),args.workers);ei={r['id']:r for r in er};hard=eligible(ev,ei)
    evalruns={a:stage(out,'eval_'+a,hard,lambda r,a=a:search(exe,source,r,a,prior if a in ('s3','s3c') else None),args.workers) for a in ('s1','s2','s3','s3c')}
    result={'schema':'olr08a.cpu.result.v1','profile':args.profile,'frozen':frozen,'evaluation_validity':validity(ev,ei),'fresh_hard':len(hard),'evaluation':{a:rate(v) for a,v in evalruns.items()},'models':{},'architecture_promotion_eligible':False,'sealed_SInt_opened':False}
    for key,(m,alpha) in models.items():
        raw=stage(out,'eval_model_'+key,hard,lambda row:search(exe,source,row,'s3',prior,m,1.),1)
        chosen=raw if alpha else evalruns['s3'];result['models'][key]={'alpha':alpha,'raw':rate(raw),'routed':rate(chosen),'paired_vs_s3':compare(chosen,evalruns['s3'])}
    result['fresh_native_valid']=result['evaluation_validity']['human_rate']>=.95 and result['evaluation_validity']['T_human']>=20 and args.profile=='full'
    result['training_updates']=len(models)*400
    result['development_gate']='not_evaluated_no_fit' if not models else 'requires_independent_paired_audit'
    result['fresh_by_family']={fam:{a:rate([x for x in rs if x['family']==fam]) for a,rs in evalruns.items()} for fam in sorted({r['family'] for r in hard})}
    write(out/'summary.json',result);print(json.dumps(result,indent=2),flush=True);return result

def cli():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--exe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',choices=('smoke','full'),default='smoke');p.add_argument('--threads',type=int,default=8);p.add_argument('--workers',type=int,default=2);p.add_argument('--reissue-a',type=Path);p.add_argument('--reissue-b',type=Path);return p.parse_args()
if __name__=='__main__':main(cli())
