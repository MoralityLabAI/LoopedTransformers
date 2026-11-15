"""RAM-13 exact RAM-08 registered 20-family source-only candidates.

Derived from immutable RAM-08 ram08_core.py blob d7295447 and FC-12
source-core blob 1b405ab. Verified against old 20,930 compiler rows.
"""
from __future__ import annotations
import hashlib,json,re
from collections import Counter

FAMILIES=('omega','rfl','decide','simp_all','all_simp','assumption',
 'exact_local','exact_typed','simpa_local','intro','intro_named','constructor','rw_local','apply_local',
 'rcases_local','cases_local','induction_list','simp_only_visible','simp_all_only_visible','dsimp_only_visible')
ARITH='simp_all [Int.neg_mul, Int.mul_neg, Int.mul_comm, Int.mul_left_comm, Int.mul_assoc]'
STANDARD={'omega','rfl','decide','simp_all','assumption','intro','constructor',ARITH,'all_goals simp_all'}
NAME=r"[A-Za-z_][A-Za-z0-9_']*"
CONSTANT=r'MathCan(?:\.[A-Za-z_][A-Za-z0-9_]*)+'
FC12_FAMILIES=('omega','rfl','simp_all','assumption','intro','constructor',
               'exact_local','apply_local','rw_local','rcases_local','induction_list','simpa_local')

def contextual_names(goal):
    if '⊢' not in goal:raise AssertionError('Lean actual goal required')
    prefix=goal.split('⊢',1)[0];rows=[]
    for line in prefix.splitlines():
        line=line.strip()
        m=re.fullmatch(r"([A-Za-z_][A-Za-z0-9_']*(?:\s+[A-Za-z_][A-Za-z0-9_']*)*)\s*:\s*(.+)",line)
        if not m:continue
        for name in m.group(1).split():
            if name not in ('case',"case'"):rows.append((name,m.group(2).strip()))
    return rows

def fc12_generate(goal):
    local=contextual_names(goal)
    out={f:[] for f in FC12_FAMILIES}
    for f in FC12_FAMILIES[:6]:out[f]=[f]
    def is_hypothesis(typ):
        return bool(any(x in typ for x in ('∀','→','=','≠','≤','<','∧','∨','∃','obsEqOn','Prop')) and
                    typ not in ('Prop','Type','Type u','Type v','Type w'))
    priority=[n for n,t in local if is_hypothesis(t)]
    rest=[n for n,t in local if not is_hypothesis(t)]
    ordered=list(dict.fromkeys(priority+rest))
    list_names=[n for n,t in local if 'List ' in t]
    validity=[n for n,t in local if '∀' in t and '→' in t]
    if list_names and validity:
        word=list_names[-1];cert=validity[-1]
        for h,t in local:
            if h in (word,cert) or not ('obsEqOn' in t or t.startswith('∀')):continue
            out['exact_local'].append(f'exact ({h} {word} {cert}).symm')
            out['exact_local'].append(f'exact {h} {word} {cert}')
    for name in ordered:
        out['exact_local'].append('exact '+name)
        out['apply_local'].append('apply '+name)
        out['simpa_local'].append('simpa using '+name)
    for name,typ in local:
        if re.search(r'(?:[^<>!])=(?:[^=>])|↔|≃',typ):
            out['rw_local'].extend(('rw ['+name+']','rw [← '+name+']'))
        if any(s in typ for s in ('∃','∧','Exists','And ')):
            out['rcases_local'].append('rcases '+name+' with ⟨fc12_left, fc12_right⟩')
        if 'List ' in typ or 'List(' in typ or typ.startswith('List '):
            out['induction_list'].append('induction '+name)
    for fam in FC12_FAMILIES:
        out[fam]=list(dict.fromkeys(out[fam]))[:4]
    return out

def valid_script(script):
    if script in STANDARD:return True
    patterns=(rf'exact {NAME}',rf'exact {NAME} {NAME} {NAME}',rf'exact \({NAME} {NAME} {NAME}\)\.symm',
      rf'apply {NAME}',rf'simpa using {NAME}',rf'rw \[(?:← )?{NAME}\]',
      rf'rcases {NAME} with ⟨fc12_left, fc12_right⟩',rf'cases {NAME}',
      rf'induction {NAME}(?: generalizing {NAME}(?: {NAME})?)?',
      rf'(?:all_goals )?simp_all \[{CONSTANT}(?:, {CONSTANT}){{0,7}}\]',
      rf'dsimp \[{CONSTANT}(?:, {CONSTANT}){{0,7}}\] at \*',
      rf'exact {NAME} {NAME}',rf'exact {NAME}\.(?:symm|left|right)',
      rf'exact {NAME}\.(?:mp|mpr) {NAME}',
      rf'exact \({NAME} {NAME} {NAME}\)\.trans \({NAME} {NAME} {NAME}\)',
      r'intro ram07_arg[0-4]',
      rf'simp only \[{CONSTANT}\]',rf'simp_all only \[{CONSTANT}\]',
      rf'dsimp only \[{CONSTANT}\] at \*')
    return any(re.fullmatch(p,script) for p in patterns)

def canonical(goals):
    mapping={}
    def variable(m):return mapping.setdefault(m.group(),f'?m{len(mapping)}')
    out=[]
    for goal in goals:
        if goal.count('⊢')!=1:raise ValueError('one turnstile per goal')
        lines=[s.strip() for s in goal.splitlines() if not re.match(r'^\s*case\s+',s)]
        text='\n'.join(' '.join(s.split()) for s in lines if s)
        out.append(re.sub(r'\?m(?:\.[A-Za-z_]+)*\.\d+',variable,text))
    return out

def signature(goals):return hashlib.sha256(json.dumps(canonical(goals),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def candidates(state):
    if not state['goals']:return []
    focus=state['goals'][0]
    old=fc12_generate(focus)
    pool={f:list(old.get(f,[])) for f in FAMILIES}
    pool['decide']=['decide'];pool['all_simp']=['all_goals simp_all']
    definitions=sorted(d for d in state.get('definitions',[]) if re.fullmatch(CONSTANT,d))[:3]
    pool['simp_only_visible']=[f'simp only [{d}]' for d in definitions]
    pool['simp_all_only_visible']=[f'simp_all only [{d}]' for d in definitions]
    pool['dsimp_only_visible']=[f'dsimp only [{d}] at *' for d in definitions]
    prop_names={n for n,t in contextual_names(focus) if t=='Prop'}
    names_and_types=contextual_names(focus)
    def is_hypothesis(name,typ):
        return typ in prop_names or any(mark in typ for mark in
            ('→','↔','=','≤','<','≥','∧','∨','∃','∀','obsEqOn')) or (
                typ not in ('Prop','Nat','Int','Bool','Type') and
                not typ.startswith('List ') and name.startswith(('h','p','proof','bridge','bound','pair','both','compound','impl','map','equality')))
    hyps=[(n,t) for n,t in names_and_types if is_hypothesis(n,t)]
    existing={n for n,_ in names_and_types}
    intro_name=next((f'ram07_arg{k}' for k in range(5) if f'ram07_arg{k}' not in existing),None)
    pool['intro_named']=[f'intro {intro_name}'] if intro_name else []
    typed=[]
    def propose(action):
        if action not in typed:typed.append(action)
    for name,typ in hyps[:12]:propose(f'exact {name}')
    for name,typ in hyps:
        if '=' in typ and '≠' not in typ and '≤' not in typ and '≥' not in typ:
            propose(f'exact {name}.symm')
        if '∧' in typ:
            propose(f'exact {name}.left');propose(f'exact {name}.right')
    for name,typ in hyps:
        parts=[t.strip() for t in typ.split('→')]
        if len(parts)>=2:
            arguments=[]
            for p in parts[:-1][:2]:
                arg=next((n for n,t in names_and_types if n!=name and t.strip()==p),None)
                if arg is None:break
                arguments.append(arg)
                propose(f'exact {name} {" ".join(arguments)}')
        if '↔' in typ:
            halves=[p.strip() for p in typ.split('↔',1)]
            if len(halves)==2:
                for side,argtype in (('mp',halves[0]),('mpr',halves[1])):
                    arg=next((n for n,t in names_and_types if n!=name and t.strip()==argtype),None)
                    if arg:propose(f'exact {name}.{side} {arg}')
    contextual=[]
    words=[n for n,t in names_and_types if 'List ' in t]
    validators=[n for n,t in names_and_types if ('∀' in t and '→' in t) and re.fullmatch(NAME,n)]
    for h,t in hyps:
        if 'obsEqOn' in t and re.fullmatch(NAME,h):
            for word in words[:2]:
                for valid in validators:
                    if word in dict(names_and_types)[valid] and re.fullmatch(NAME,word):
                        for script in (f'exact ({h} {word} {valid}).symm',f'exact {h} {word} {valid}'):
                            if script not in contextual:contextual.append(script)
    observed_equalities=[h for h,t in hyps if 'obsEqOn' in t and re.fullmatch(NAME,h)]
    for word in words[:2]:
        for valid in validators:
            if word not in dict(names_and_types)[valid] or not re.fullmatch(NAME,word):continue
            for h1 in observed_equalities[:3]:
                for h2 in observed_equalities[:3]:
                    if h1==h2:continue
                    script=f'exact ({h1} {word} {valid}).trans ({h2} {word} {valid})'
                    if script not in contextual:contextual.insert(0,script)
    pool['exact_typed']=(contextual+[x for x in typed if x not in contextual])[:24]
    local=contextual_names(focus)
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
        seen=set();limit=24 if family=='exact_typed' else 4
        for script in pool[family][:limit]:
            if script in seen:continue
            if not valid_script(script):raise ValueError('outside RAM08 grammar: '+script)
            result.append((family,script));seen.add(script)
    return result
