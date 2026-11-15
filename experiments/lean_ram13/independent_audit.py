"""RAM13 third fresh native Lean4.35 source compiler audit, after scores frozen."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,'/mnt/data/ram13_local')
import harness
from native_replay import render_source,axiom_inventory

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--start',type=int,default=0)
 ap.add_argument('--stop',type=int,default=156)
 args=ap.parse_args()
 assert 0<=args.start<args.stop<=156
 phases=[('primary',harness.POLICIES),('secondary',harness.SECOND)]
 originals=harness.module_sources()
 cases=[(m,t) for m,(_,ts) in harness.LOCKED.items() for t in ts]
 all_rows=[]
 for ph,policies in phases:
  for i,(m,theorem) in enumerate(cases):
   for policy in policies:
    path=harness.OUT/ph/f'{i:02d}'/policy
    if not (path/'case.json').is_file():raise AssertionError('missing first scored case '+str(path))
    c=json.loads((path/'case.json').read_text())
    budget=harness.PRIMARY if ph=='primary' else harness.SECONDARY
    assert c['budget']==budget
    assert c['logical_evaluations']<=budget['evaluations']
    assert c['original_source_sha256']==harness.sha(originals[m].encode())
    if c['accepted']:
     assert c['reason']=='independently_certified' and c['actions']
     assert all(harness.grammar.valid_script(s) for s in c['actions'])
     assert len(c['actions'])<=budget['transitions']
     first,second=c['certificate'],c['replay']
     assert first['accepted'] and second['accepted'] and first['source_sha256']==second['source_sha256']
     raw=(path/'certificate.lean').read_bytes()
     assert raw==(path/'independent_replay.lean').read_bytes()
     assert harness.sha(raw)==first['source_sha256']
     exact=render_source(originals[m],theorem,c['actions']).encode()
     assert raw==exact
     for script in c['actions']:
      assert not any(w in script for w in ('sorry','admit','run_tac','unsafe'))
    all_rows.append(dict(phase=ph,case_index=i,module=m,theorem=theorem,policy=policy,source=path,accepted=c['accepted'],first=c))
 assert len(all_rows)==156
 got=[r for r in all_rows if r['accepted']]
 assert len(got)==114
 target=Path('/mnt/data/ram13_local/third_audit');target.mkdir(parents=True,exist_ok=True)
 for n,r in enumerate(all_rows[args.start:args.stop],start=args.start):
  if not r['accepted']:continue
  out=target/r['phase']/f"{r['case_index']:02d}"/r['policy']
  receipt=out/'third_audit.json'
  if receipt.exists():
   z=json.loads(receipt.read_text());assert z['accepted'] and z['compiler_sha256']==r['first']['certificate']['source_sha256']
   continue
  if out.exists():raise AssertionError('preexisting incomplete third compile '+str(out))
  out.mkdir(parents=True)
  raw=(r['source']/'certificate.lean').read_bytes()
  source=out/'third.lean';source.write_bytes(raw)
  t0=time.perf_counter()
  try:
   p=subprocess.run([str(harness.LEAN),str(source.resolve())],cwd=harness.SRC,
         env=harness.make_env(),capture_output=True,text=True,timeout=30)
   log=p.stdout+p.stderr
   (out/'third.log').write_text(log,encoding='utf8')
   gate,axioms,detail=axiom_inventory(log)
   assert p.returncode==0 and gate and 'error:' not in log and 'sorryAx' not in log,detail+log[-1000:]
   assert axioms==r['first']['certificate']['axioms']==r['first']['replay']['axioms']
   record=dict(accepted=True,phase=r['phase'],case_index=r['case_index'],module=r['module'],
      theorem=r['theorem'],policy=r['policy'],original_source_sha256=r['first']['original_source_sha256'],
      compiler_sha256=harness.sha(raw),axioms=axioms,third_log_sha256=harness.sha(log.encode()),
      elapsed_seconds=time.perf_counter()-t0,score_unchanged=True)
   receipt.write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
   print('RAM13_AUDIT',n,r['phase'],r['case_index'],r['policy'],'pass',round(record['elapsed_seconds'],2),flush=True)
  except Exception as e:
   (out/'AUDIT_FAILURE.txt').write_text(str(e));raise
 print('RAM13_AUDIT_CHUNK_OK',args.start,args.stop,'accepted cases audited',sum(x['accepted'] for x in all_rows[args.start:args.stop]),flush=True)

if __name__=='__main__':main()
