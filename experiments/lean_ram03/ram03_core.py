"""RAM-03 registered grammar, visible-state signatures and source isolation."""
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
NAME=r"[A-Za-z_][A-Za-z0-9_']*"
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
      rf'apply {NAME}',rf'simpa using {NAME}',rf'rw \[(?:← )?{NAME}\]',
      rf'rcases {NAME} with ⟨fc12_left, fc12_right⟩',rf'cases {NAME}',
      rf'induction {NAME}(?: generalizing {NAME}(?: {NAME})?)?',
      rf'(?:all_goals )?simp_all \[{CONSTANT}(?:, {CONSTANT}){{0,7}}\]',
      rf'dsimp \[{CONSTANT}(?:, {CONSTANT}){{0,7}}\] at \*')
    return any(re.fullmatch(p,script) for p in patterns)

def candidates(state):
    from research.refutation_bench.fc12 import core
    if not state['goals']:return []
    focus=state['goals'][0]
    old=core.generate(focus)
    pool={f:list(old.get(f,[])) for f in FAMILIES}
    definitions=sorted(d for d in state.get('definitions',[]) if re.fullmatch(CONSTANT,d))[:8]
    pool['all_simp']=['all_goals simp_all']
    pool['arith_norm']=[ARITH]
    if definitions:
        bulk=', '.join(definitions)
        pool['unfold_simp']=[f'simp_all [{bulk}]']+[f'simp_all [{d}]' for d in definitions[:3]]
        pool['all_simp'].append(f'all_goals simp_all [{bulk}]')
        pool['dsimp_visible']=[f'dsimp [{bulk}] at *']
    local=core.contextual_names(focus)
    scalar=[n for n,t in local if (t in ('Nat','Int') or re.fullmatch(NAME,t)) and t not in ('Prop','Type','Bool')]
    lists=[n for n,t in local if 'List ' in t or t.startswith('List(')]
    generated=[]
    for word in lists:
        for name in scalar[:2]:generated.append(f'induction {word} generalizing {name}')
        generated.append(f'induction {word}')
    pool['induction_list']=generated[:4]
    pool['cases_local']=[f'cases {n}' for n,t in local if t in ('Bool','Op') or any(s in t for s in (' ∨ ',' ∧ ','∃','Exists','And '))][:4]
    result=[]
    for family in FAMILIES:
        seen=set()
        for script in pool[family][:4]:
            if script not in seen:
                if not valid_script(script):raise ValueError('unregistered generated script: '+script)
                result.append((family,script));seen.add(script)
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
    body=[header]+(['  ram03_probe_server'] if server else ['  '+a for a in actions])
    if not server:body+=['#eval IO.println "RAM03_AXIOMS_BEGIN"','#print axioms '+theorem,'#eval IO.println "RAM03_AXIOMS_END"']
    return ('import ProbeServer\n' if server else 'import Lean\n')+'\n'.join(prefix+body+([endings[-1]] if endings else []))+'\n'

def label(before,result):
    if not result['accepted'] or signature(before['goals'])==signature(result.get('goals',[])):return 0
    if not result['goals']:return 3
    return 1 if len(result['goals'])<len(before['goals']) else 2

def axiom_audit(log):
    if log.count('RAM03_AXIOMS_BEGIN')!=1 or log.count('RAM03_AXIOMS_END')!=1:return False,[]
    part=log.split('RAM03_AXIOMS_BEGIN')[1].split('RAM03_AXIOMS_END')[0]
    records=re.findall(r"'([^']+)'\s+(does not depend on any axioms|depends on axioms:\s*\[[^\]]*\])",part,re.S)
    if len(records)!=1:return False,[]
    if records[0][1]=='does not depend on any axioms':return True,[]
    axioms=[s.strip() for s in records[0][1].split('[')[1].split(']')[0].split(',') if s.strip()]
    return set(axioms)<={'propext','Classical.choice','Quot.sound'},axioms
