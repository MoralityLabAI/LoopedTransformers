"""Explicitly derive future-cache probes from XREF19, preserving its arithmetic."""
from pathlib import Path
from .common import ROOT,read,sha,write


def transform(text,rules):
    for old,new,count in rules:
        assert text.count(old)==count,(old,text.count(old),count)
        text=text.replace(old,new)
    return text


WORKER_RULES=[
    ('from .common import ROOT, RUNS, qualified, read, sha, verify, write','from .common import ROOT, RUNS, qualified, read, rows, sha, verify, write',1),
    ('from .cache import correlations, groups, load, nullable, root_mean','from .cache import correlations, coverage, groups, load, nullable, root_mean',1),
    ("read(ROOT / 'CACHE_INVENTORY.json')","read(ROOT / 'DICTIONARY_INVENTORY.json')",1),
    ("inventory = read(ROOT / 'DICTIONARY_INVENTORY.json')['backends'][a.backend]","inventory = read(ROOT / 'DICTIONARY_INVENTORY.json')['backends'][a.backend]\n    admission = read(RUNS / 'admission' / (a.backend + '.json'))",1),
    ("        folder = Path('runs/nan_fit16_io18')", "        if admission['status'] != 'admitted':\n            cells.append(dict(passes=passes, status='unrun_no_passed_terminal_context_audit')); continue\n        if admission['available_contexts'] == 0:\n            cells.append(dict(passes=passes, status='unrun_no_audited_vectors')); continue\n        folder = Path('runs/nan_fit16_io18')",1),
    ('assert len(meta) == expected',"assert len(meta) == 2 * admission['available_contexts'] <= expected",1),
    ("r.get('arm', 'focused') == 'focused'","r.get('arm') == 'main'",1),
    ("results.append(dict(dataset=dataset, vectors=len(meta),", "results.append(dict(dataset=dataset, coverage=coverage(rows('experiments/nan_ctx21/fixtures/public.jsonl'), meta), vectors=len(meta),",1),
    ("elapsed_seconds=time.monotonic()-started, pretrained_model_loaded=False", "admission_sha256=sha(RUNS / 'admission' / (a.backend + '.json')), elapsed_seconds=time.monotonic()-started, pretrained_model_loaded=False",1),
]

AUDIT_RULES=[
    ('from .common import ROOT, RUNS, read, sha, verify, write','from .common import ROOT, RUNS, read, rows, sha, verify, write',1),
    ("read(ROOT / 'CACHE_INVENTORY.json')","read(ROOT / 'DICTIONARY_INVENTORY.json')",1),
    ("    results = []\n", "    admission = read(RUNS / 'admission' / (backend + '.json'))\n    assert receipt['admission_sha256'] == sha(RUNS / 'admission' / (backend + '.json'))\n    results = []\n",1),
    ("haar = load_npz(haarfile)\n", "haar = load_npz(haarfile)\n        np.testing.assert_allclose(np.linalg.norm(haar['V'], axis=0), np.linalg.norm(final['V'], axis=0), rtol=1e-10, atol=1e-10)\n        np.testing.assert_allclose(np.linalg.norm(haar['U'], axis=1), np.linalg.norm(final['U'], axis=1), rtol=1e-10, atol=1e-10)\n",1),
    ("            keys = STRATA[dataset]\n", "            keys = STRATA[dataset]\n            check_coverage(data, meta, rows('experiments/nan_ctx21/fixtures/public.jsonl'), keys)\n",1),
    ('assert checked_vectors == 1706 and len(cell[\'geometry\']) == 18',"assert checked_vectors == 2 * admission['available_contexts'] and len(cell['geometry']) == 6",1),
    ('physical_geometry_cells=18','physical_geometry_cells=6',1),
]


def check_coverage_source():
    return '''\n\ndef check_coverage(data, meta, public, keys):
    expected = {tuple(r[k] for k in keys[:-1]) + (t,) for r in public for t in (1, 2)}
    slots = data['coverage']; assert len(slots) == len(expected) == 48
    assert {tuple(g[k] for k in keys) for g in slots} == expected
    observed = set()
    for g in slots:
        planned = [r for r in public if all(r[k] == g[k] for k in keys[:-1])]
        actual = [r for r in meta if all(r[k] == g[k] for k in keys)]
        assert g['registered_vectors'] == len(planned) and g['available_vectors'] == len(actual)
        assert g['missing_vectors'] == len(planned) - len(actual)
        assert g['registered_roots'] == len({r['root'] for r in planned}) and g['available_roots'] == len({r['root'] for r in actual})
        if actual:
            observed.add(tuple(g[k] for k in keys))
    assert sum(g['registered_vectors'] for g in slots) == 1248 and sum(g['available_vectors'] for g in slots) == len(meta)
    for bank in [data['learned_groups']] + [v['groups'] for v in data['towers']]:
        assert len(bank) == len(observed) and {tuple(g[k] for k in keys) for g in bank} == observed
'''


def main():
    inventory=read('experiments/nan_xref19/CACHE_INVENTORY.json')
    write(ROOT/'DICTIONARY_INVENTORY.json',dict(parent_inventory_sha256=sha('experiments/nan_xref19/CACHE_INVENTORY.json'),
        backends={b:dict(dictionaries=v['dictionaries']) for b,v in inventory['backends'].items()},
        future_cache_binding='Terminal CTX21 audit+receipt and exact contiguous public prefix; exclusive admission at runtime; no future cache vector read before freeze.'))
    derivation=[]
    for name,rules in [('worker.py',WORKER_RULES),('audit.py',AUDIT_RULES)]:
        old=Path('experiments/nan_xref19')/name;new=ROOT/name
        assert not new.exists()
        text=old.read_text(encoding='utf-8');result=transform(text,rules)
        if name=='audit.py':
            result=result.replace('\n\nif __name__',check_coverage_source()+'\n\nif __name__')
        with new.open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(result)
        derivation.append(dict(parent=str(old).replace('\\','/'),parent_sha256=sha(old),derived=name,derived_sha256=sha(new),
            explicit_substitutions=[dict(old=a,new=b,occurrences=n) for a,b,n in rules],independent_coverage_function_added=name=='audit.py'))
    write(ROOT/'DERIVATION.json',dict(files=derivation,source_only=True,numerical_cache_analysis_started=False,new_model_calls=0,sealed_test_accessed=False))
    print(dict(derived_files=2,dictionary_backends=2,numerical_cache_vectors_read=0))


if __name__=='__main__':
    main()
