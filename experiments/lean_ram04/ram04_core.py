"""RAM-04 registered grammar, visible-state signatures and source isolation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

FAMILIES=('omega','rfl','simp_all','unfold_simp','all_simp','arith_norm','assumption',
 'exact_local','simpa_local','intro','constructor','rw_local','apply_local',
 'rcases_local','cases_local','induction_list','dsimp_visible')
ARITH='simp_all [Int.neg_mul, Int.mul_neg, Int.mul_comm, Int.mul_left_comm, Int.mul_assoc]'
STANDARD={'omega','rfl','simp_all','assumption','intro','constructor',ARITH,'all_goals simp_all'}
LEAN_IMPORTS='\n'.join('import '+s for s in ('Lean.Elab.Tactic.Basic','Lean.Elab.Tactic.BuiltinTactic','Lean.Elab.Tactic.RCases','Lean.Elab.Tactic.Induction','Lean.Elab.Tactic.Simp','Lean.Elab.Tactic.Omega','Lean.Elab.Deriving.FromToJson'))+'\n'
NAME=r"(?:[^\W\d]|_)[\w']*"
CONSTANT=r'MathCan(?:\.[A-Za-z_][A-Za-z0-9_]*)+'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sha(value):return hashlib.sha256(value.encode()).hexdigest()
def read_rows(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n');tmp.replace(path)
def append_json(path,value):
    with Path(path).open('a',encoding='utf8',newline='\n') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')

def canonical(goals):
    mapping={}
    def variable(m):return mapping.setdefault(m.group(),f'?m{len(mapping)}')
    out=[]
    for goal in goals:
        if goal.count('⊢')!=1:raise ValueError('one turnstile required per goal')
        lines=[s.strip() for s in goal.splitlines() if not re.match(r'^\s*case\s+',s)]
        text='\n'.join(' '.join(s.split()) for s in lines if s)
        out.append(re.sub(r'\?m(?:\.[A-Za-z_]+)*\.\d+',variable,text))
    return out
def signature(goals):return sha(json.dumps(canonical(goals),ensure_ascii=False,separators=(',',':')))

def valid_script(script):
    if script in STANDARD:return True
    patterns=(rf'exact {NAME}',rf'exact {NAME} {NAME} {NAME}',rf'exact \({NAME} {NAME} {NAME}\)\.symm',
      rf'exact {NAME} {NAME}',rf'exact {NAME}\.(?:symm|1|2)',rf'intro {NAME}',
      rf'apply {NAME}',rf'simpa using {NAME}',rf'rw \[(?:← )?{NAME}\]',
      rf'rcases {NAME} with ⟨fc12_left, fc12_right⟩',rf'cases {NAME}',
      rf'rcases {NAME} with ⟨r4a\d+, r4b\d+⟩',rf'rcases {NAME} with r4a\d+ \| r4b\d+',
      rf'rw \[(?:← )?{NAME}\] at \*',
      rf'induction {NAME}(?: generalizing {NAME}(?: {NAME})?)?',
      rf'(?:all_goals )?simp_all \[{CONSTANT}(?:, {CONSTANT}){{0,7}}\]',
      rf'dsimp \[{CONSTANT}(?:, {CONSTANT}){{0,7}}\] at \*')
    return any(re.fullmatch(p,script) for p in patterns)

def visible_locals(state):
    return [v for v in state.get('locals',[]) if re.fullmatch(NAME,v['name']) and v['name']!='_']

def fresh_suffix(state):
    names={v['name'] for v in visible_locals(state)}
    return next(i for i in range(100) if not names & {f'r4a{i}',f'r4b{i}',f'r4intro{i}'})

def candidates(state):
    from research.refutation_bench.fc12 import core
    if not state['goals']:return []
    old=core.generate(state['goals'][0]);pool={f:list(old.get(f,[])) for f in FAMILIES}
    local=visible_locals(state);props=[v for v in local if v['prop']];data=[v for v in local if not v['prop']]
    exact=[f"exact {v['name']}" for v in (props+data)[:4]]
    applications=[];symmetry=[];projections=[]
    for v in props:
        n=v['name']
        if v['head']=='Eq':symmetry.append(f'exact {n}.symm')
        if v['head']=='And':projections.extend((f'exact {n}.1',f'exact {n}.2'))
        if v['function']:
            arguments=[x for x in local if x['name']!=n and ' '.join(x['type'].split())==' '.join(v['domain'].split())]
            for x in arguments[:2]:
                applications.append(f"exact {n} {x['name']}")
                for y in [p for p in props if p['name'] not in (n,x['name'])][:3]:
                    applications.append(f"exact {n} {x['name']} {y['name']}")
    pool['exact_local']=exact+applications[:4]+symmetry[:2]+projections[:2]
    pool['apply_local']=[f"apply {v['name']}" for v in props[:4]]
    pool['simpa_local']=[f"simpa using {v['name']}" for v in props[:4]]
    pool['rw_local']=[]
    for v in [v for v in props if v['rewritable']][:2]:
        n=v['name'];pool['rw_local'] += [f'rw [{n}]',f'rw [← {n}]',f'rw [{n}] at *',f'rw [← {n}] at *']
    i=fresh_suffix(state);pool['rcases_local']=[]
    for v in [v for v in props if v['head'] in ('And','Exists','Or')][:6]:
        n=v['name'];pattern=f'r4a{i} | r4b{i}' if v['head']=='Or' else f'⟨r4a{i}, r4b{i}⟩'
        pool['rcases_local'].append(f'rcases {n} with {pattern}')
    pool['intro']=[f'intro r4intro{i}']
    definitions=sorted(d for d in state.get('definitions',[]) if re.fullmatch(CONSTANT,d))[:8]
    pool['all_simp']=['all_goals simp_all'];pool['arith_norm']=[ARITH]
    if definitions:
        bulk=', '.join(definitions)
        pool['unfold_simp']=[f'simp_all [{bulk}]']+[f'simp_all [{d}]' for d in definitions[:3]]
        pool['all_simp'].append(f'all_goals simp_all [{bulk}]');pool['dsimp_visible']=[f'dsimp [{bulk}] at *']
    scalar=[v['name'] for v in data if v['type'] in ('Nat','Int') or re.fullmatch(NAME,v['type']) and v['type'] not in ('Prop','Type','Bool')]
    pool['induction_list']=[]
    for v in [v for v in data if 'List ' in v['type']]:
        pool['induction_list'] += [f"induction {v['name']} generalizing {n}" for n in scalar[:2] if n!=v['name']]+[f"induction {v['name']}"]
    pool['cases_local']=[f"cases {v['name']}" for v in local if v['head'] in ('And','Exists','Or','Bool') or v['type']=='Op'][:4]
    caps={'exact_local':12,'rw_local':8,'rcases_local':6};result=[]
    for family in FAMILIES:
        for script in dict.fromkeys(pool[family]):
            if sum(f==family for f,_ in result)>=caps.get(family,4):break
            if not valid_script(script):raise ValueError('unregistered generated script: '+script)
            result.append((family,script))
    return result

def make_source(original,theorem,actions,*,server=False):
    from research.refutation_bench.fc11.source_core import theorem_bounds
    if any(not valid_script(s) for s in actions):raise ValueError('outside registered grammar')
    raw=original.splitlines();start,begin,end=theorem_bounds(raw,theorem)
    header=raw[begin].split(':= by',1)[0]+':= by'
    endings=[line for line in raw[end:] if re.fullmatch(r'end\s+[A-Za-z_0-9.]+',line.strip())]
    # Keep declaration documentation and attributes adjacent to the theorem.
    option_at=start
    while option_at>0:
        previous=raw[option_at-1].strip()
        if not previous or previous.startswith('@['):option_at-=1;continue
        if previous.endswith('-/'):
            opener=next((i for i in range(option_at-1,-1,-1) if '/--' in raw[i]),None)
            if opener is not None:option_at=opener;continue
        break
    limit=0 if server else 100000
    prefix=raw[:option_at]+[f'set_option maxHeartbeats {limit}']+raw[option_at:begin]
    body=[header]+(['  ram04_probe_server'] if server else ['  '+a for a in actions])
    if not server:body+=['#eval IO.println "RAM04_AXIOMS_BEGIN"','#print axioms '+theorem,'#eval IO.println "RAM04_AXIOMS_END"']
    return ('import ProbeServer\n' if server else LEAN_IMPORTS)+'\n'.join(prefix+body+([endings[-1]] if endings else []))+'\n'

def label(before,result):
    if not result['accepted'] or signature(before['goals'])==signature(result.get('goals',[])):return 0
    if not result['goals']:return 3
    return 1 if len(result['goals'])<len(before['goals']) else 2

def axiom_audit(log):
    if log.count('RAM04_AXIOMS_BEGIN')!=1 or log.count('RAM04_AXIOMS_END')!=1:return False,[]
    part=log.split('RAM04_AXIOMS_BEGIN')[1].split('RAM04_AXIOMS_END')[0]
    records=re.findall(r"'([^']+)'\s+(does not depend on any axioms|depends on axioms:\s*\[[^\]]*\])",part,re.S)
    if len(records)!=1:return False,[]
    if records[0][1]=='does not depend on any axioms':return True,[]
    axioms=[s.strip() for s in records[0][1].split('[')[1].split(']')[0].split(',') if s.strip()]
    return set(axioms)<={'propext','Classical.choice','Quot.sound'},axioms
