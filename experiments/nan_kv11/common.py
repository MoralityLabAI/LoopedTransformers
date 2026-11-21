"""Public-only complete suite roster; no routing corpus or host labels."""
from pathlib import Path
from experiments.nan_fit07.io import read,rows,sha,write


def public_rows():
    root=Path('experiments/nan_gym05');seal=read(root/'SOURCE_MANIFEST.json');file=root/'fixtures/public.jsonl'
    assert sha(root/'SOURCE_MANIFEST.json')=='96294903b1be52021725f2b839a7ad79fa635fd75aa4d5752cb3d53af0d132b0'
    assert sha(file)==seal['public_repo_fixture_sha256'];allrows=rows(file);dev=[r for r in allrows if r['split']=='development']
    assert len(allrows)==290 and len(dev)==130 and len({r['root'] for r in dev})==41
    assert all(r['suite']!='routing' for r in allrows)
    return allrows,dev


def verify(root):
    seal=read(root/'SOURCE_MANIFEST.json')
    for name,h in seal['worker_files_sha256'].items():assert sha(root/name)==h,name
    for name,h in seal['model_dependencies_sha256'].items():assert sha(name)==h,name
    return seal
