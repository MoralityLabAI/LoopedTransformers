"""Rank/site registration helpers; queues never import model libraries."""
from pathlib import Path
from experiments.nan_fit07.io import read,rows,sha,write


def key(condition):
    return f"t{condition['passes']}_layer21_{condition['module']}_rank{condition['rank']}"


def condition(index):
    root=Path(__file__).parent;base=read(root/'config.json');extra=read(root/'run_config.json')
    assert 0<=index<len(base['conditions'])==6
    return {**base,**extra,**base['conditions'][index]}


def verify():
    root=Path(__file__).parent;seal=read(root/'SOURCE_MANIFEST.json')
    for name,h in seal['worker_files_sha256'].items():assert sha(root/name)==h,name
    for name,h in seal['model_dependencies_sha256'].items():assert sha(name)==h,name
    return seal
