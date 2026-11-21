"""Preserve the prior independent checks/probe definitions under the new seal."""
import ast
from pathlib import Path


def main():
    root=Path(__file__).parent
    if (root/'cell_audit.py').exists() or (root/'probe.py').exists():raise FileExistsError('Preserve the derived analysis files')
    text=Path('mechinterp/nanbeige_bpd_gym/audit_fit.py').read_text();tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cell_audit')
    function='\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n'
    header='''"""Original FIT07 independent cell checks with the RECOMP09 source root.

No check is relaxed; resident-reference/recomputation assertions are additional.
"""
import math
from pathlib import Path
import numpy as np
from experiments.nan_fit07.io import read,sha
from mechinterp.nanbeige_bpd_gym.audit_fit import checkpoint,check_steps,diagnostics_audit
from experiments.nan_fit07.pointwise import masks
ROOT=Path(__file__).parent


'''
    (root/'cell_audit.py').write_text(header+function,encoding='utf-8',newline='\n')
    probe=Path('experiments/nan_fit07/probe.py').read_text()
    for old,new in [('from .io import read,sha,write','from experiments.nan_fit07.io import read,sha,write'),
                    ('from .pointwise import masks','from experiments.nan_fit07.pointwise import masks'),
                    ("default='runs/nan_fit07'","default='runs/nan_recomp09'")]:
        if probe.count(old)!=1:raise ValueError('Probe derivation anchor changed')
        probe=probe.replace(old,new,1)
    (root/'probe.py').write_text(probe,encoding='utf-8',newline='\n')


if __name__=='__main__':main()
