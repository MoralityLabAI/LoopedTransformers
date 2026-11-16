"""Execute one complete frozen RAM-15 theorem across matched six-policy pools.

Source pin and model SHA are checked before any public TEST theorem source is
opened. Every original result is a separate immutable per-policy directory.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
import harness

def run(index:int,*,phases:tuple[str,...]=('primary','secondary','ablation')):
    if subprocess.check_output(['git','-C',str(harness.SRC),'rev-parse','HEAD'],text=True).strip()!=harness.MANIFEST['source_commit']:
        raise ValueError("unregistered MathCan source")
    if (harness.SRC/'lean-toolchain').read_text().strip()!='leanprover/lean4:v4.35.0':
        raise ValueError("incorrect Lean compiler source")
    if not harness.LEAN.is_file():raise FileNotFoundError("compiled Lean4.35 not available")
    # Freeze/check all original learned weights and train-only control data
    # before opening any prospective theorem source.
    ranker=harness.Rankers()
    source=harness.module_sources()
    cases=[(module,t) for module,(_,names) in harness.LOCKED.items() for t in names]
    if len(cases)!=14:raise ValueError("RAM15 registered theorem count drifted")
    if not 0<=index<14:raise ValueError("case index out of registered range")
    module,theorem=cases[index]
    for phase in phases:
        if phase=='primary':policies=harness.POLICIES;budget=harness.PRIMARY;ablation=False
        elif phase=='secondary':policies=harness.SECOND;budget=harness.SECONDARY;ablation=False
        elif phase=='ablation':policies=['native','loop64'];budget=harness.PRIMARY;ablation=True
        else:raise ValueError("unregistered phase")
        ordered=policies[index%len(policies):]+policies[:index%len(policies)]
        for policy in ordered:
            out=harness.OUT/phase/f'{index:02d}'/policy
            if out.exists():
                if (out/'case.json').is_file():
                    raise FileExistsError("original scored case already exists; preserve it: "+str(out))
                raise FileExistsError("partial case must be quarantined without replacement: "+str(out))
            r=harness.search(source[module],theorem,ranker,policy,budget,out,ablation=ablation)
            if r['budget']!=budget:raise AssertionError("per-case scorer budget altered")
            if r['logical_evaluations']>budget['evaluations']:
                raise AssertionError("logical proof attempts exceeded budget")
            print('RAM15_RESULT',json.dumps(dict(phase=phase,index=index,module=module,theorem=theorem,
                policy=policy,accepted=r['accepted'],reason=r['reason'],
                calls=r['logical_evaluations'],wall=round(r['elapsed_seconds'],4))),
                flush=True)
    case_receipt=dict(index=index,module=module,theorem=theorem,
        source_commit=harness.MANIFEST['source_commit'],
        policies_by_phase={k:v for k,v in [('primary',harness.POLICIES),
                                           ('secondary',harness.SECOND),
                                           ('ablation',['native','loop64'])]},
        status='complete_first_attempt')
    root=harness.OUT/'case_manifests';root.mkdir(parents=True,exist_ok=True)
    (root/f'{index:02d}.json').write_text(json.dumps(case_receipt,indent=2)+'\n')
    print('RAM15_CASE_COMPLETE',index,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=int,required=True)
    p.add_argument('--phase',choices=['all','primary','secondary','ablation'],default='all')
    a=p.parse_args()
    run(a.case,phases=('primary','secondary','ablation') if a.phase=='all' else (a.phase,))
