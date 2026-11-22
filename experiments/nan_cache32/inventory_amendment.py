"""Preserve the original source and proof; permit only inventory bookkeeping."""
import ast
import copy
from .common import ROOT,read,sha

ORIGINAL_SEAL='4993a010890cc4a6d5b257bcdea61cb63fd69282603f423a1f9c2762589aeddd'
HISTORICAL={'predecessors.py':'predecessors_before_inventory.py.txt','freeze.py':'freeze_before_inventory.py.txt'}


def assert_predecessor_equivalence(candidate,original):
    new=ast.parse(candidate.read_text(encoding='utf-8'));old=ast.parse(original.read_text(encoding='utf-8'))
    current=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='ensure_inventory_clear')
    before=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='ensure_inventory_clear')
    expected=ast.parse('def ensure_inventory_clear(exclude_pid=None):\n from .inventory import ensure_clear\n return ensure_clear(exclude_pid)\n').body[0]
    assert ast.dump(current)==ast.dump(expected),'Only the explicit owned-launcher inventory delegate is allowed'
    normalized=copy.deepcopy(new);index=next(i for i,n in enumerate(normalized.body) if isinstance(n,ast.FunctionDef) and n.name=='ensure_inventory_clear')
    normalized.body[index]=copy.deepcopy(before)
    assert ast.dump(normalized)==ast.dump(old),'Predecessor waits or other admission behavior changed'


def verify():
    archived=ROOT/'SOURCE_MANIFEST_before_inventory.json';assert sha(archived)==ORIGINAL_SEAL
    old=read(archived);amendment=read(ROOT/'INVENTORY_AMENDMENT.json')
    assert amendment['original_source_manifest_sha256']==ORIGINAL_SEAL
    assert amendment['pretrained_forwards_before_amendment']==0 and amendment['numeric_method_and_plan_unchanged']
    assert amendment['stdlib_contracts_passed']==17 and amendment['actual_Windows_fixture_passed']
    assert amendment['original_ENGINEERING_sha256']==sha(ROOT/'ENGINEERING.json')==old['critical_files_sha256']['ENGINEERING.json']
    paths={name:ROOT/backup for name,backup in HISTORICAL.items()}
    for name,value in old['critical_files_sha256'].items():assert sha(paths.get(name,ROOT/name))==value,name
    for name,value in amendment['new_source_sha256'].items():assert sha(ROOT/name)==value,name
    assert_predecessor_equivalence(ROOT/'predecessors.py',paths['predecessors.py'])
    assert sha(ROOT/'windows_inventory_fixture.py.txt')==amendment['actual_Windows_fixture_source_sha256']
    assert amendment['retirement']['cached_models_started']==0 and amendment['retirement']['worker_permits_written']==0
    return paths
