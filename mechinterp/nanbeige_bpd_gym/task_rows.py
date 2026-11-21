"""Prime/other suite interchange for only the frozen fresh development sources."""
import hashlib
import json
from pathlib import Path

REGISTRATION='10936e4caac06115e4a53b600cbab658295cc246'
SOURCE_SEAL_SHA256='37eba00f942ccd2153e749f1689d1eed7cfa397e72cfbd6b77ee26afd6b573a2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def load_rows(experiment_root=Path('experiments/nan_diag04'),*,split='development',component='free'):
    root=Path(experiment_root)
    if split not in ('calibration','development'):raise ValueError('Only fresh CAL/DEV sources are supported')
    allowed={'free','scaffold','copy','select','table','coded','labelcopy','codecopy','erased','erased_repeat'}
    if component not in allowed:raise ValueError('Unknown registered component')
    if sha(root/'SOURCE_MANIFEST.json')!=SOURCE_SEAL_SHA256:raise ValueError('Unrecognized source registration; refuse before reading any data')
    seal=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    for name,h in seal['model_input_files_sha256'].items():
        if sha(root/'fixtures'/name)!=h:raise ValueError('Public source seal mismatch')
    if sha(root/'fixtures/host_labels.jsonl')!=seal['host_labels_sha256']:raise ValueError('Host source seal mismatch')
    public={r['root']:r for r in map(json.loads,(root/'fixtures/public.jsonl').read_text().splitlines())}
    host={r['root']:r for r in map(json.loads,(root/'fixtures/host_labels.jsonl').read_text().splitlines())};rows=[]
    for key,row in public.items():
        if row['split']!=split:continue
        c=host[key]
        for arm,prompt in row['prompts'].items():
            prefix,n=arm.split('/') if '/' in arm else (arm,None)
            if prefix!=component:continue
            if prefix in ('free','scaffold','copy'):answer=c['refs']['R'][0][int(n[1:])]
            elif n is not None:
                op=c['worlds'][int(n)];answer=c['codes'][op] if prefix in ('coded','codecopy') else op
            else:answer=list(c['worlds'])
            rows.append({'id':f'nan-diag04:{key}:{arm}','root_id':str(key),'arm':arm,'split':split,'kind':prompt['kind'],
                'prompt':prompt['text'],'answer':answer,'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),
                'endpoint':'complete_final_reply_grammar; distinct from local first-complete-prefix decoding',
                'registration_commit':REGISTRATION,'gold_assistance':prefix in ('copy','table','coded','labelcopy','codecopy')})
    return rows

def score_reply(text,kind,answer):
    # A complete reply, not a scan for letters/numbers in explanatory prose.
    from experiments.nan_diag04.parser import parse
    result=parse(text,kind,eos=True)
    if result['status']!='accepted':return 0.
    value=result['value']
    return sum(value==x for x in answer)/len(answer) if isinstance(answer,list) else float(value==answer)
