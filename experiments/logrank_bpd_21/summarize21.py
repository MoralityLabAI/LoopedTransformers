"""Summarize audited development rows and verify exposed-root timing rerun."""
import json,statistics as s
from collections import Counter,defaultdict
from pathlib import Path
from base19 import sha
H=Path(__file__).resolve().parent
rs=[json.loads((H/f'audit{n}.json').read_text()) for n in (12,48,96)];counts=Counter();rows=[]
for r in rs:assert r['status']=='PASS';counts.update(r['counts']);rows+=r['rows']
g=defaultdict(list)
for r in rows:g[(r['n'],r['family'],r['regime'],r['kib'],r['policy'])].append(r)
aggregate=[]
for k,v in sorted(g.items()):
    n,family,regime,kib,policy=k;assert len(v)==6
    aggregate.append(dict(n=n,family=family,regime=regime,kib=kib,policy=policy,conditions=len(v),mean_work=s.mean(x['work'] for x in v),mean_ms=1000*s.mean(x['seconds'] for x in v),mean_plan_ms=1000*s.mean(x['plan_seconds'] for x in v),mean_peak_bytes=s.mean(x['peak_bytes'] for x in v),max_peak_bytes=max(x['peak_bytes'] for x in v),mean_warm_work=s.mean(x['warm_work'] for x in v) if v[0]['warm_work'] is not None else None,mean_uncached_work=s.mean(x['uncached_work'] for x in v) if v[0]['uncached_work'] is not None else None))
a=json.loads((H/'receipt96.json').read_text());b=json.loads((H/'timing_repeat.json').read_text())
def strip(x):
    if isinstance(x,dict):return {k:strip(v) for k,v in x.items() if k not in ('seconds','planning_seconds','peak_rss_kib')}
    if isinstance(x,list):return [strip(v) for v in x]
    return x
assert strip(a)==strip(b)
rg=defaultdict(list);ws={w['seed']:w for w in b['worlds']}
for x in b['results']:
    if x['status']=='refused':continue
    for reg in x['regimes']:
        for arm in reg['arms']:rg[(ws[x['seed']]['family'],reg['regime'],arm['kib'],arm['policy'])].append(arm['seconds']*1000)
repeat=[dict(family=k[0],regime=k[1],kib=k[2],policy=k[3],mean_ms=s.mean(v)) for k,v in sorted(rg.items())]
files=['PROTOCOL.md','pilot21.py','audit21.py','base19.py','base20.py','reference19.py','summarize21.py','fixture.json','timing_repeat.json']+[f'{p}{n}.json' for n in (12,48,96) for p in ('receipt','audit')]
summary=dict(evidence='development only; 96D timing repeat on exposed worlds',counts=dict(counts),fixtures=json.loads((H/'fixture.json').read_text()),non_timing_repeat_matches=18,aggregates=aggregate,repeat96=repeat,source_and_raw_sha256={k:sha(H/k) for k in files},root_rows=rows)
(H/'summary_receipt.json').write_text(json.dumps(summary,indent=2)+'\n');print('aggregates',len(aggregate),'roots',len(rows),'audit',counts,'fixture',summary['fixtures']);print('source and raw hashes',summary['source_and_raw_sha256'])
