"""Small I/O contracts; no model, source oracle, dataset or network imports."""
import hashlib
import json
from pathlib import Path

REVISION = '451ed48c3273ecef7ea8faaa43c31ce529763bb1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8', newline='\n')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s]

def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v, separators=(',',':'))+'\n' for v in values), encoding='utf-8', newline='\n')

def prompt_ids(tok, snapshot, text):
    cfg = read(Path(snapshot)/'tokenizer_config.json')
    t = tok.tok
    t.chat_template = cfg['chat_template']
    t.bos_token, t.eos_token = cfg['bos_token'], cfg['eos_token']
    rendered = t.apply_chat_template([{'role':'user','content':text}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
    return t.encode(rendered+'\nAnswer:', add_special_tokens=False)
