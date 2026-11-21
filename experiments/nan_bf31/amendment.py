"""Explicit pre-inference identity amendment; old scientific method is exact.

No prior model reply is replayed. The original source freeze and full-path
engineering receipt remain historical evidence. Only worker runtime identity,
earlier enforcement of the existing admission rule, source-freeze bookkeeping
and this amendment's tests/documentation may change.
"""
import ast
import copy
from pathlib import Path


def assert_worker_equivalence(candidate,original):
    new=ast.parse(Path(candidate).read_text(encoding='utf-8'));old=ast.parse(Path(original).read_text(encoding='utf-8'))
    main=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name=='main')
    def runtime(tree):
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='write' and node.args:
                path=node.args[0]
                if isinstance(path,ast.BinOp) and isinstance(path.right,ast.Constant) and path.right.value=='bf16_runtime.json':
                    return next(k for k in node.args[1].keywords if k.arg=='backend')
        raise AssertionError('Missing original runtime identity write')
    current=runtime(new);prior=runtime(old)
    assert isinstance(current.value,ast.Constant) and current.value.value=='bf16'
    assert isinstance(prior.value,ast.Constant) and prior.value.value=='nf4'
    current.value.value='nf4'
    positions=[i for i,x in enumerate(main.body) if isinstance(x,ast.ImportFrom) and x.level==1 and x.module=='admission'
        and [(a.name,a.asname) for a in x.names]==[('verify_worker',None)]]
    assert len(positions)==1;position=positions[0]
    call=main.body[position+1]
    assert isinstance(call,ast.Expr) and isinstance(call.value,ast.Call) and isinstance(call.value.func,ast.Name)
    assert call.value.func.id=='verify_worker' and not call.value.args and not call.value.keywords
    imports=[i for i,x in enumerate(main.body) if isinstance(x,ast.Import) and any(a.name=='experiments.nan_bpd01.backend' for a in x.names)]
    assert len(imports)==1 and position<imports[0],'Existing admission must precede Torch backend import'
    del main.body[position:position+2]
    assert ast.dump(new,include_attributes=False)==ast.dump(old,include_attributes=False),'Unexpected scientific/numerical/worker change'
    return dict(runtime_backend_identity_corrected=True,existing_admission_checked_before_Torch_backend=True,
        all_other_worker_AST_exact=True,decoder_generation_scoring_budgets_unchanged=True)


def verify_historical_engineering():
    from .common import ROOT,read,sha
    proof=read(ROOT/'ENGINEERING.json');change=read(ROOT/'RUNTIME_ADMISSION_AMENDMENT.json')
    old=read(ROOT/'SOURCE_MANIFEST_before_runtime_admission.json')
    assert sha(ROOT/'SOURCE_MANIFEST_before_runtime_admission.json')==change['original_source_manifest_sha256']
    assert sha(ROOT/'ENGINEERING.json')==change['original_ENGINEERING_sha256']
    archives={'worker.py':'worker_before_runtime_admission.py.txt','freeze.py':'freeze_before_runtime_admission.py.txt'}
    for name,digest in proof['source_sha256'].items():
        if name in archives:
            assert sha(ROOT/archives[name])==digest
            assert sha(ROOT/name)==change['amended_source_sha256'][name]
        else:assert sha(ROOT/name)==digest,name
    for name,digest in old['critical_files_sha256'].items():
        if name not in ('worker.py','freeze.py','PROTOCOL.md'):assert sha(ROOT/name)==digest,name
    result=assert_worker_equivalence(ROOT/'worker.py',ROOT/archives['worker.py'])
    for name,digest in change['new_admission_contract_source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert change['new_admission_contracts_passed']==5 and change['pretrained_forwards_before_amendment']==0
    assert change['historical_full_host_fixture_reexecuted'] is False
    return result
