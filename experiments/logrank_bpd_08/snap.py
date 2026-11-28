"""Adaptive compression audit: signed coordinate projection of staged learned bank."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
from audit import unpack,eval_program,equivalent,component_certificate
rows=[]
for path in sys.argv[1:]:
    r=json.load(open(path)); k=r['k']
    for t in r['results']:
        a=next(a for a in t['arms'] if a['arm']=='teacher_warmup'); v=np.array(a['V']); z=unpack(t['heldout_ids'],k); idx=np.abs(v).argmax(0); ss=np.where(v[idx,np.arange(k)]>=0,1,-1); programs=[]; cert=[]
        for route,p in zip(a['query_routes'],t['truth']):
            ix=route['indices']; pr=(route['op'],idx[ix].tolist(),(ss[ix]*route['signs']).tolist()); programs.append(pr); cert.append(equivalent(pr,p))
        pred=np.stack([eval_program(z,p) for p in programs],1); saved=''.join('1' if b else '0' for b in (pred>0).flat); correct=sum(x==y for x,y in zip(saved,t['labels'])); disagreements=sum(x!=y for x,y in zip(saved,a['predictions'])); active=sorted({i for route in a['query_routes'] for i in route['indices']}); rows.append(dict(k=k,root=t['root'],correct=correct,total=len(saved),prediction_disagreements=disagreements,certified_queries=sum(cert),used_components=len(active),distinct_selected_coordinates=len(set(idx[active])),snapped_indices=idx.tolist(),snapped_signs=ss.tolist(),programs=programs,certificate=cert,bank_payload_bits=k*k*32,signed_coordinate_payload_bits=k*k.bit_length(),used_coordinate_payload_bits=len(active)*k.bit_length()))
print(json.dumps(dict(evidence='adaptive projection; no claim of global equivalence to original dense bank',rows=rows,source_receipt_sha256={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sys.argv[1:]}),indent=2))
