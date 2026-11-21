"""Original story packet endpoints plus exact immutable trajectory/state bindings."""
import hashlib
import json
import math
from pathlib import Path
import re
from .common import read,write,sha
from experiments.nan_full26.common import parse

WIDTHS=dict(residual_after_layer=3072,down_proj_input=10752,down_proj_output=3072,o_proj_input=6144,o_proj_output=3072)


def digest(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def grade(text,r,g):
    parsed=parse(text,r);valid=parsed is not None
    final=bool(valid and parsed['final_room']==g['final_room'])
    checkpoints=bool(valid and parsed['intermediate_rooms']==g['intermediate_rooms'])
    path=bool(valid and parsed['path']==g['path'])
    return dict(format_valid=valid,final_correct=final,checkpoints_correct=checkpoints,path_correct=path,joint_correct=final and checkpoints and path)


def check_episode(base,entry,r,g,tok,initial,bound):
    import numpy as np
    from experiments.nan_mlp27.host import decode
    base=Path(base);closed=base/entry['file'];assert sha(closed)==entry['sha256']
    m=re.fullmatch(r'baseline_(\d{4})_r([01])_closed.json',entry['file']);assert m
    ordinal,row=int(m[1]),int(m[2]);assert ordinal+row==entry['slot']
    stem=f'baseline_{ordinal:04d}_r{row}'
    record=read(closed)
    assert all(record[k]==r[k] for k in ('root','stage','split'))
    assert record['source_id']==r['id'] and record['mode']=='native' and record['lane']=='baseline'
    assert record['initial_ids']==initial and len(initial)==bound['tokens'] and digest(initial)==bound['ids_sha256']
    generated=record['generated_ids'];n=len(generated)
    assert 1<=n<=64 and len(record['fragments'])==record['prefix_count']==n
    assert all(isinstance(v,int) and 0<=v<166144 for v in generated)
    expected={(phase,layer,site) for phase in (1,2) for layer in (5,11,17,21) for site in WIDTHS}
    batch_shapes=[]
    for step,fragment in enumerate(record['fragments']):
        assert fragment['file']==f'{stem}_p{step:03d}.json'
        file=base/fragment['file'];assert sha(file)==fragment['sha256'];p=read(file)
        assert p['step']==step and p['source_id']==r['id'] and p['mode']=='native' and p['token']==generated[step]
        assert p['history_sha256']==digest(generated[:step]) and p['prefix_ids_sha256']==digest(initial+generated[:step])
        assert p['actual_length']==len(initial)+step and p['padded_length']>=p['actual_length']
        assert p['batch_size'] in (1,2) and 0<=p['batch_row']<p['batch_size']
        assert p['batch_row']==(row if p['batch_size']==2 else 0)
        assert p['padding_tokens']>=0 and (p['padding_tokens']==0 if p['batch_size']==1 else True)
        assert p['full_vocabulary_size']==166144
        assert math.isfinite(p['top_logit']) and math.isfinite(p['log_normalizer']) and p['log_normalizer']>=p['top_logit']-1e-10
        assert all(math.isfinite(p[k]) and p[k]>=0 for k in ('cuda_elapsed_ms_for_entire_batch','wall_seconds_for_entire_batch'))
        assert p['state_file']==f'{stem}_p{step:03d}.npz'
        state=base/p['state_file'];assert sha(state)==p['state_sha256']
        index=p['state_index'];sites={(i['loop'],i['layer'],i['site']) for i in index}
        assert len(index)==40 and sites==expected and len({i['key'] for i in index})==40
        with np.load(state,allow_pickle=False) as z:
            assert set(z.files)=={i['key'] for i in index}|{'boundaries','native_at_reached_input'}
            for i in index:
                a=z[i['key']];assert a.dtype==np.uint16 and a.shape==(WIDTHS[i['site']],) and np.isfinite(decode(a)).all()
                if i['layer']==21 and i['site']=='down_proj_output':
                    assert np.array_equal(a,z['native_at_reached_input'][i['loop']-1]),'native local-write identity'
            for key in ('boundaries','native_at_reached_input'):
                a=z[key];assert a.dtype==np.uint16 and a.shape==(2,3072) and np.isfinite(decode(a)).all()
        eos=generated[step]==tok.eos_token_id
        text=tok.decode(generated[:step] if eos else generated[:step+1],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        stop='eos' if eos else 'complete_packet' if parse(text,r) is not None else 'character_cap' if len(text)>512 else 'token_cap' if step+1>=64 else None
        if step<n-1:assert stop is None,'generation continued after registered stop'
        else:assert stop==record['stop_reason'] and text==record['text']
        batch_shapes.append((p['batch_size'],p['batch_row'],p['actual_length'],p['padded_length'],p['padding_tokens']))
    return dict(slot=entry['slot'],source_id=r['id'],root=r['root'],world=r['world'],split=r['split'],arm=r['arm'],query_type=r['query_type'],
        event_count=r['event_count'],grade=grade(record['text'],r,g),prefixes=n,physical_vectors=40*n,stop_reason=record['stop_reason'],
        closed_file=entry['file'],closed_sha256=entry['sha256'],batch_shapes=batch_shapes)


def check_pair_shapes(records):
    byslot={v['slot']:v for v in records};assert len(byslot)==len(records)
    for slot,r in byslot.items():
        other=byslot.get(slot^1)
        if other is None:continue
        for step,shape in enumerate(r['batch_shapes']):
            if shape[0]==2:
                assert step<len(other['batch_shapes'])
                peer=other['batch_shapes'][step]
                assert peer[0]==2 and peer[1]!=shape[1]
                assert shape[3]==peer[3]==max(shape[2],peer[2])
                assert shape[4]==peer[4]==2*shape[3]-shape[2]-peer[2]
            else:assert step>=other['prefixes'],'unexpected serial substitute while both story rows active'
