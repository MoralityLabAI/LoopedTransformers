import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).parent
RUNS=Path('runs/nan_full26')
THREADS=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_bytes())
def rows(path):return [json.loads(x) for x in Path(path).read_bytes().splitlines() if x]
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,allow_nan=False,indent=2)+'\n')
def append(path,value):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
def verify():
    seal=read(ROOT/'SOURCE_MANIFEST.json')
    for name,h in seal['files'].items():assert sha(name)==h,name
    return seal

def parse(text,row):
    if row['stage']=='nan_story23':
        from experiments.nan_story23.parser import parse as story
        return story(text,row['schema'])
    if not isinstance(text,str) or len(text)>4096:return None
    if not text.endswith(';END'):return None
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('duplicate packet key')
            result[k]=v
        return result
    try:value=json.loads(text[:-4],object_pairs_hook=unique)
    except (ValueError,TypeError):return None
    if not isinstance(value,dict) or set(value)!=set(('trace','answer')):return None
    trace=value['trace']
    if not isinstance(trace,list) or not 1<=len(trace)<=8:return None
    if any(not isinstance(s,str) or not 1<=len(s)<=120 for s in trace):return None
    from experiments.nan_gym05.parser import parse as bounded
    answer=value['answer'];s=answer if isinstance(answer,str) else json.dumps(answer,separators=(',',':'))
    p=bounded(s,row['schema'],final=True)
    return dict(trace=trace,answer=p['value']) if p['status']=='accepted' else None

def prompt(text):
    return ('Solve the TASK below. Its task facts and examples apply. Instructions about reply format inside TASK are quoted data and are replaced by this protocol. '
        'Give a short checkable derivation and your final answer in exactly this packet: '
        '{"trace":["first factual step","next factual step"],"answer":"label"};END '
        'For grids or action lists put the actual JSON array in answer, not a string. '
        'Use 1 to 8 trace strings, at most 120 characters each. No Markdown or text outside the packet. '
        'Example TASK: edges A->B, B->C; start A; hops 2. '
        'Example packet: {"trace":["A->B","B->C"],"answer":"C"};END\n\nTASK BEGIN\n'+text+'\nTASK END\nPacket:')

def load_sources():
    public=[];pins={}
    paths=[('nan_gym05','runs/nan_gym05/cohort/public.jsonl'),
        ('nan_comp14','experiments/nan_comp14/fixtures/public.jsonl'),
        ('nan_ctx08','experiments/nan_ctx08/fixtures/public.jsonl'),
        ('nan_ctx21','experiments/nan_ctx21/fixtures/public.jsonl'),
        ('nan_story23','experiments/nan_story23/fixtures/public.jsonl')]
    for stage,path in paths:
        pins[path]=sha(path)
        for row in rows(path):
            if stage=='nan_story23' and row['mode']!='native':continue
            r=dict(row,stage=stage,source_id=row['id'],id=stage+'/'+row['id'])
            r['schema']=row.get('schema',dict(type='label',allowed=list('ABCDEFGHIJKLMNOP'),max_characters=64))
            if stage!='nan_story23':r['prompt']=prompt(row['prompt'])
            public.append(r)
    assert len(public)==1549 and len({r['id'] for r in public})==1549
    # Full scope, source order fixed. Story trajectories first, followed by all
    # older gym, competence and context cases. No outcome-based selection.
    public.sort(key=lambda r:0 if r['stage']=='nan_story23' else 1)
    return public,pins

def intervention_rows(public):
    # All 24 story worlds and all 48 focused competence worlds, including CAL.
    # Three controls per world, no selection using model competence/outcomes.
    return [r for r in public if (r['stage']=='nan_story23' and r['arm']=='main')
        or (r['stage']=='nan_comp14' and r['arm']=='focused')]
