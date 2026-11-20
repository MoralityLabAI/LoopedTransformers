"""Atomic local evidence I/O; no model or gold imports."""
from pathlib import Path
import json
from experiments.nan_gym05.common import read,rows,sha


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    temporary.replace(path)


def write_rows(path,values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in values),encoding='utf-8',newline='\n')
