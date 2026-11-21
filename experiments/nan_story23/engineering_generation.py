"""Exercise real multi-prefix writer/stops with synthetic CPU capture injection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from .common import ROOT,RUNS,read,rows,sha,write


def run():
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    from experiments.nan_ranksite13.resources import below_normal
    below_normal()
    import torch
    torch.set_num_threads(1)
    from .engine import Engine,ids_hash
    cfg=read(ROOT/'config.json');source=rows(ROOT/'fixtures/public.jsonl')[0];gold=rows(ROOT/'fixtures/host_labels.jsonl')[0]
    text=gold['canonical_trace'];half=len(text)//2;pieces={1:text[:half],2:text[half:],3:'unstructured response'}
    RUNS.mkdir(exist_ok=True)
    base=RUNS.resolve()
    with tempfile.TemporaryDirectory(prefix='generation_mock_',dir=base) as temp:
        out=Path(temp);assert out.resolve().is_relative_to(base)
        for label,tokens,expected in [('packet',[1,2],'first_complete_packet'),('eos',[3,0],'eos'),('capacity',[3,3],None)]:
            folder=out/label;folder.mkdir();engine=Engine.__new__(Engine)
            engine.cfg=dict(cfg);engine.torch=torch;engine.out=folder;engine.instrument=None
            engine.bytes=0;engine.ids=lambda row:[10,11,12];engine.budget=lambda started:None
            cache=SimpleNamespace(rows_gathered=0,reset=lambda:None);engine.embedding_cache=cache
            engine.tok=SimpleNamespace(tok=SimpleNamespace(eos_token_id=0,
                decode=lambda ids,**kw:''.join(pieces[i] for i in ids)))
            calls=[]
            def capture(ids):
                if label=='capacity' and calls:raise RuntimeError('MOCK_ONLY_injected_capacity_stop')
                step=len(calls);calls.append(ids);cache.rows_gathered=len(ids)
                states={('context',phase,layer,site):torch.full((10752 if site=='down_proj_input' else 3072,),float(phase+step))
                    for phase in (1,2) for layer in cfg['layers_zero_based']
                    for site in ('residual_after_layer','down_proj_input','down_proj_output','o_proj_input','o_proj_output')}
                return dict(states=states,boundaries=[torch.zeros((1,1,3072)).numpy() for _ in range(2)],
                    scores=[dict(free_top_token_id=tokens[step]) for _ in range(2)])
            engine.capture=capture
            if expected is None:
                try:engine.generate(source,0)
                except RuntimeError as error:assert str(error)=='MOCK_ONLY_injected_capacity_stop'
                else:raise AssertionError('Injected capacity failure disappeared')
                assert len(list(folder.glob('*prefix_*.json')))==1 and not (folder/'request_0000.json').exists()
            else:
                result=engine.generate(source,0);assert result['stop_reason']==expected and len(result['fragments'])==2
                assert result['generated_ids']==tokens and result['embedding_rows_gathered']==4
                fragments=[read(folder/v['file']) for v in result['fragments']]
                assert fragments[0]['prefix_ids_sha256']==ids_hash([10,11,12])
                assert fragments[1]['prefix_ids_sha256']==ids_hash([10,11,12,tokens[0]])
                assert result['text']==(text if label=='packet' else pieces[3])
    return dict(status='passed',evidence_class='real Engine.generate writer with synthetic CPU capture only',
        complete_packet_prefixes=2,EOS_prefixes=2,preserved_partial_capacity_prefixes=1,
        model_forwards=0,CUDA_allocations=0,source_sha256={name:sha(ROOT/name) for name in ('engine.py','engineering_generation.py')},sealed_test_accessed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();value=run();write(a.out,value);print(value)
