"""RAM-10 source-isolated Lean action contract. Importing this module reads no TEST source.

Reuse RAM-08's original, registered 20 action families, then apply a frozen
family-fair 80-action cap to all policies. No gold tactic or proof content is
used to propose a candidate.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RAM08 = REPO/'experiments'/'lean_ram08'
DECL = re.compile(r"^(theorem|lemma)\s+([A-Za-z_][A-Za-z_0-9']*)\b",re.M)
BOUNDARY = re.compile(r'^(?:theorem|lemma|def|abbrev|opaque|instance|axiom|namespace|section|end|#print|#eval)\b')
FAMILY_TARGETS = ('exact_typed','exact_local','rw_local','rcases_local','induction_list')


def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def sha_file(path):return sha_bytes(Path(path).read_bytes())
def stable_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def append_json(path,obj):
    with Path(path).open('a',encoding='utf8') as f:f.write(json.dumps(obj,sort_keys=True,ensure_ascii=False)+'\n')
def rows(path):return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def inventory(source,maximum):
    """Top-level declarations by source order; deliberately ignores proof bodies."""
    declarations=[(kind,name) for kind,name in DECL.findall(source)]
    names=[name for _,name in declarations]
    if len(set(names))!=len(names):raise AssertionError('duplicate original top-level names')
    if not declarations:raise AssertionError('no source-order declaration found')
    return declarations[:maximum]


def fair_candidates(candidates,families,limit=80,first=4):
    """Reserve first four source-visible scripts/family, then fill old order."""
    if len(families)!=20 or len(set(families))!=20:
        raise AssertionError('changed original family registry')
    if limit<first*len(families):raise AssertionError('family reservation exceeds cap')
    if any(x[0] not in families for x in candidates):
        raise AssertionError('unregistered tactic family')
    pick=[];counts=Counter();selected=set()
    for i,item in enumerate(candidates):
        if counts[item[0]]<first:
            counts[item[0]]+=1;pick.append(item);selected.add(i)
    for i,item in enumerate(candidates):
        if len(pick)>=limit:break
        if i not in selected:pick.append(item)
    if len(pick)>limit:pick=pick[:limit]
    if len(set(pick))!=len(pick):
        # Registered pool can contain duplicate script across different families,
        # but duplicate exact (family,script) within one state is forbidden.
        raise AssertionError('identical candidate duplicated')
    return pick


def source_order_roots(mathcan,config,partition):
    if partition not in ('train','dev','test'):raise ValueError(partition)
    count=config['root_selector']['first_test' if partition=='test' else 'first_train_dev']
    roots=[]
    for module,expected_blob in config['modules'][partition].items():
        path=Path(mathcan)/'MathCan'/f'{module}.lean'
        if not path.is_file():raise FileNotFoundError(str(path))
        blob=subprocess.check_output(['git','-C',str(mathcan),'hash-object',str(path.resolve())],text=True).strip()
        if blob!=expected_blob:raise AssertionError('original module Git blob changed: '+module)
        source=path.read_text(encoding='utf8')
        for kind,name in inventory(source,count):
            roots.append(dict(partition=partition,source_file=f'MathCan/{module}.lean',
                theorem=name,declaration_kind=kind,git_blob_sha=blob,source_sha256=sha_bytes(source.encode('utf8'))))
    return roots


def verify_checkout(mathcan,config,mode):
    pinned=subprocess.check_output(['git','-C',str(mathcan),'rev-parse','HEAD'],text=True).strip()
    if pinned!=config['source_mathcan']:raise AssertionError('MathCan source commit mismatch')
    if (Path(mathcan)/'lean-toolchain').read_text(encoding='utf8').strip()!=config['toolchain']:
        raise AssertionError('Lean toolchain mismatch')
    if mode=='train':
        for module in config['modules']['test']:
            if (Path(mathcan)/f'MathCan/{module}.lean').exists():
                raise AssertionError('TRAIN checkout contains unmasked heldout module: '+module)
    return pinned


def _lemma_bounds(lines,name,original):
    if any(re.match(r'^theorem\s+'+re.escape(name)+r'\b',x) for x in lines):
        return original(lines,name)
    starts=[i for i,x in enumerate(lines)
            if re.match(r'^lemma\s+'+re.escape(name)+r'\b',x)]
    if len(starts)!=1:raise AssertionError('expected one original theorem/lemma declaration')
    start=starts[0];stop=len(lines)
    for i in range(start+1,len(lines)):
        if BOUNDARY.match(lines[i]) or lines[i].startswith(('/--','/-!')):
            stop=i;break
    markers=[i for i in range(start,stop) if ':=' in lines[i]]
    if not markers:raise AssertionError('source lemma has no proof body assignment')
    return start,markers[0],stop


def registered_ram08_core(config):
    """Install strictly local RAM08 compatibility hooks; no added tactic scripts."""
    if str(RAM08) not in sys.path:sys.path.insert(0,str(RAM08))
    import ram08_core as old
    if tuple(config['candidate_families'])!=tuple(old.FAMILIES):
        raise AssertionError('candidate family order differs from original RAM08')
    if not hasattr(old,'_ram10_original_candidates'):
        old._ram10_original_candidates=old.candidates
        old._ram10_original_bounds=old.original_theorem_bounds
        old.candidates=lambda state:fair_candidates(old._ram10_original_candidates(state),old.FAMILIES,
                  config['collector']['max_actions_per_state'],config['collector']['each_family_first'])
        old.original_theorem_bounds=lambda lines,name:_lemma_bounds(lines,name,old._ram10_original_bounds)
    return old


def gate(rows,config):
    families=defaultdict(set)
    unique=defaultdict(set)
    for row in rows:
        if row['partition']!='train' or row['label']<=0:continue
        name=row['family'];root=(row['source_file'],row['theorem'])
        families[name].add(root)
        unique[name].add((root,row['state_sha256'],row['script']))
    exact_roots=families['exact_typed']|families['exact_local']
    rewrite=families['rw_local']
    case_or_induction=families['rcases_local']|families['induction_list']
    pass_gate=(len(exact_roots)>=config['gate']['min_distinct_train_exact_roots'] and
          len(rewrite)>=config['gate']['min_distinct_train_rw_roots'] and
          len(case_or_induction)>=config['gate']['min_distinct_train_rcases_or_induction_roots'])
    return dict(admitted=pass_gate,source_only_original=True,
                exact_roots=len(exact_roots),rewrite_roots=len(rewrite),
                cases_or_induction_roots=len(case_or_induction),
                per_family={name:dict(positive_distinct_roots=len(families[name]),
                                      positive_unique_state_scripts=len(unique[name]))
                            for name in FAMILY_TARGETS},
                synthesized_theorems=0,reference_proofs_as_labels=0,
                status='admitted' if pass_gate else 'insufficient_genuine_tactic_coverage')

