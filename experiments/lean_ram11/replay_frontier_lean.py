"""Lean4.35 original MathCan TRAIN-only replay of RAM-11F frontier candidates.

This is an optional LOCAL executable diagnostic, never a substitute for the
registered RAM08/10 held-out proof evaluations. Each proposed complete proof
is independently kernel compiled twice with permitted-axiom checks.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'lean_ram08'))
import ram08_core as core
from runtime import Session, ResourceGuard, ProbeTimeout, certify, compile_source
sys.path.insert(0,str(HERE))
from prefix_audit import load_archive,source_splits
from frontier_worklist import worklist

SOURCE_COMMIT='c84e2025334e9a967c1297425a490f98d5b3d189'
MAX_CASE_SECONDS=30.0
REQUEST_SECONDS=4.0
CERTIFY_SECONDS=10.0
REPLAY_SECONDS=10.0


def verified_inputs(archive:Path,mathcan:Path,planned:Path):
    actual=subprocess.check_output(['git','-C',str(mathcan),'rev-parse','HEAD'],text=True).strip()
    if actual!=SOURCE_COMMIT:raise ValueError('old original MathCan source checkout SHA changed')
    changed=subprocess.check_output(['git','-C',str(mathcan),'diff','--name-only','HEAD','--','MathCan','lean-toolchain','lakefile.toml'],text=True).strip()
    if changed:raise ValueError('tracked old MathCan source is modified; use a clean pinned checkout')
    original,audit=load_archive(archive)
    train,_=source_splits(original)
    expected,receipt=worklist(train)
    text=''.join(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n' for x in expected)
    file_bytes=planned.read_bytes()
    if file_bytes!=text.encode('utf8'):
        raise ValueError('proposed original TRAIN prefix worklist bytes differ from original pinned source')
    return expected,receipt


def replay_original_sources(archive:Path,mathcan:Path,planned:Path,lean:str,out:Path):
    if out.exists():raise FileExistsError('Use a new output directory for each independent local Lean run')
    jobs,receipt=verified_inputs(archive,mathcan,planned)
    version=subprocess.check_output([lean,'--version'],text=True).strip()
    if '4.35.0' not in version:
        raise ValueError('RAM11F original proof source replay requires Lean4.35.0: '+version)
    out.mkdir(parents=True)
    aggregate=dict(campaign='RAM11F-local-old-source-syntactic-prefix-candidate-replay',
        status='running',original_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        original_mathcan_source_commit=SOURCE_COMMIT,
        candidate_worklist_sha256=hashlib.sha256(planned.read_bytes()).hexdigest(),
        expected_candidate_scripts=len(jobs),attempted=0,
        complete_original_source_certificates=0,
        PEF_heldout_cases_scored=0,RAM10_heldout_cases_scored=0,
        candidate_results=[])
    grouped=defaultdict(list)
    for i,job in enumerate(jobs):grouped[(job['source_file'],job['theorem'])].append((i,job))
    try:
        with ResourceGuard(out/'resources.jsonl') as guard:
            for (filename,theorem),cases in grouped.items():
                original=(mathcan/filename).read_text(encoding='utf8')
                sha=core.sha(original)
                if sha!=cases[0][1]['original_source_sha256']:
                    raise ValueError('original source text SHA mismatch: '+filename)
                for _,job in cases:
                    if sha!=job['original_source_sha256']:
                        raise ValueError('inconsistent original source SHA in worklist')
                root_dir=out/'source_roots'/str(len(aggregate['candidate_results']))
                with Session(lean,original,theorem,root_dir,guard,10.) as server:
                    for case_index,job in cases:
                        started=time.perf_counter()
                        actions=job['proposed_new_actions']
                        if not actions or any(not core.valid_script(action) for action in actions):
                            raise ValueError('worklist includes script outside original RAM08 registered grammar')
                        trial=dict(case_index=case_index,filename=filename,theorem=theorem,
                            actions=actions,source_sha256=sha,classification='unknown',
                            independently_compiled_original_Lean=False)
                        case_dir=out/'evidence'/f'{case_index:03d}'
                        case_dir.mkdir(parents=True)
                        try:
                            guard.check()
                            candidate=server.query(actions,REQUEST_SECONDS)
                            trial['lean_compiler_accepted_full_script']=candidate.get('accepted')
                            trial['remaining_goals']=len(candidate['goals']) if candidate.get('accepted') else None
                            if not candidate.get('accepted'):
                                trial['classification']='lean_rejected'
                            elif candidate['goals']:
                                trial['classification']='lean_accepted_open_goals'
                            elif time.perf_counter()-started>MAX_CASE_SECONDS-20:
                                trial['classification']='in_budget_certificate_time_unavailable'
                            else:
                                # The source renderer replaces target theorem's old proof.
                                # Two independent fresh Lean compiles (not two probe responses)
                                # must accept the exact same source and axiom inventory.
                                first=certify(lean,original,theorem,actions,case_dir/'complete_source',guard,
                                    min(CERTIFY_SECONDS,max(.01,MAX_CASE_SECONDS-(time.perf_counter()-started))))
                                trial['first_full_source_compile']=first
                                if first['accepted'] and time.perf_counter()-started<MAX_CASE_SECONDS:
                                    full_text=Path(first['source_file']).read_text(encoding='utf8')
                                    second=compile_source(lean,full_text,case_dir/'independent_complete_source',guard,
                                        min(REPLAY_SECONDS,max(.01,MAX_CASE_SECONDS-(time.perf_counter()-started))))
                                    trial['second_full_source_replay']=second
                                    valid=second['accepted'] and second['source_sha256']==first['source_sha256'] and set(second['axioms'])==set(first['axioms']) and time.perf_counter()-started<=MAX_CASE_SECONDS
                                    if valid:
                                        trial['classification']='independently_kernel_certified_original_train_theorem'
                                        trial['independently_compiled_original_Lean']=True
                                        trial['source_proof_sha256']=first['source_sha256']
                                        aggregate['complete_original_source_certificates']+=1
                                    else:
                                        trial['classification']='independent_full_source_rejected_or_budget'
                                else:
                                    trial['classification']='first_full_source_rejected_or_budget'
                        except ProbeTimeout as error:
                            trial['classification']='compiler_timeout_unknown'
                            trial['error']=str(error)[:400]
                            trial['elapsed_seconds']=round(time.perf_counter()-started,3)
                            aggregate['candidate_results'].append(trial)
                            aggregate['attempted']+=1
                            raise RuntimeError('original compiler outcome unknown: stop before assigning label') from error
                        trial['elapsed_seconds']=round(time.perf_counter()-started,3)
                        aggregate['candidate_results'].append(trial)
                        aggregate['attempted']+=1
                        core.write_json(out/'partial_receipt.json',aggregate)
                        print('RAM11F_REPLAY',case_index,trial['classification'],flush=True)
        aggregate['status']='complete'
    except BaseException as e:
        aggregate.update(status='aborted',error=repr(e))
        core.write_json(out/'final_receipt.json',aggregate)
        raise
    core.write_json(out/'final_receipt.json',aggregate)
    return aggregate

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--mathcan',type=Path,required=True)
    p.add_argument('--worklist',type=Path,required=True)
    p.add_argument('--lean',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    replay_original_sources(a.archive,a.mathcan,a.worklist,a.lean,a.out)
