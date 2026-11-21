"""Exercise real B2 generation/writers/EOS/tail with injected CPU captures."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from .common import ROOT,THREADS,write,sha,read,rows

def run():
    for key in THREADS:os.environ[key]='1'
    from experiments.nan_ranksite13.resources import below_normal
    below_normal()
    import torch
    torch.set_num_threads(1)
    from .worker import Engine
    packet='{"trace":["A->B"],"answer":"B"};END';half=len(packet)//2
    pieces={1:packet[:half],2:packet[half:]}
    base=Path('runs/nan_full26_engineering').resolve();assert base.is_relative_to(Path.cwd().resolve());base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mock_',dir=base) as temp:
        assert Path(temp).resolve().is_relative_to(base)
        engine=Engine.__new__(Engine);engine.t=torch;engine.out=Path(temp);engine.bytes=0
        engine.embedding=SimpleNamespace(reset=lambda:None);engine.budget=lambda start:None;engine.admitted=lambda b,n:True
        engine.ids=lambda r:[10,11,12] if r['id']=='A' else [20,21]
        engine.tok=SimpleNamespace(tok=SimpleNamespace(eos_token_id=0,decode=lambda ids,**kw:''.join(pieces[i] for i in ids)))
        calls=[]
        def capture(sequences,modes):
            step=len(calls);calls.append(dict(lengths=list(map(len,sequences)),modes=list(modes)))
            scores=torch.full((len(sequences),4),-5.)
            for b in range(len(sequences)):scores[b,0 if b==1 else 1 if step==0 else 2]=10
            return dict(states={(b,p,l,s):torch.full((4,),float(p+b),dtype=torch.bfloat16)
                for b in range(len(sequences)) for p in (1,2) for l in (5,11,17,21)
                for s in ('residual_after_layer','down_proj_input','down_proj_output','o_proj_input','o_proj_output')},
                boundaries=[torch.ones(len(sequences),4,dtype=torch.bfloat16)]*2,logits=[scores,scores],
                native=[[torch.zeros(4,dtype=torch.bfloat16) for b in sequences]]*2,masks=[torch.ones(8)]*2,
                cuda_elapsed_ms=1.,wall_seconds=.01,padding_tokens=1)
        engine.capture=capture
        public=[dict(id=name,root=name,split='development',stage='nan_comp14',schema=dict(type='label',allowed=['A','B'])) for name in ('A','B')]
        engine.generate(public,['native','component_only'],0,'AB')
        a=read(engine.out/'AB_0000_r0_closed.json');b=read(engine.out/'AB_0000_r1_closed.json')
        assert a['text']==packet and a['stop_reason']=='complete_packet' and a['prefix_count']==2
        assert b['text']=='' and b['stop_reason']=='eos' and b['prefix_count']==1
        assert calls==[dict(lengths=[3,2],modes=['native','component_only']),dict(lengths=[4],modes=['native'])]
        assert read(engine.out/a['fragments'][1]['file'])['batch_size']==1
        for entry in rows(engine.out/'closed.jsonl'):assert sha(engine.out/entry['file'])==entry['sha256']
        assert len(rows(engine.out/'closed.jsonl'))==2
        import numpy as np
        from .host import decode
        for episode in (a,b):
            for f in episode['fragments']:
                record=read(engine.out/f['file'])
                with np.load(engine.out/record['state_file'],allow_pickle=False) as z:
                    assert len(record['state_index'])==40
                    for index in record['state_index']:
                        assert np.all(decode(z[index['key']])==index['loop']+record['batch_row'])
        # Preserve incomplete prefix files after a later capture fails; never
        # publish or replay them as a completed model reply.
        engine.out=engine.out/'injected_failure';engine.out.mkdir();calls.clear()
        def failing(sequences,modes):
            if calls:raise RuntimeError('MOCK_ONLY_injected_capture_failure')
            return capture(sequences,modes)
        engine.capture=failing
        try:engine.generate(public,['native','component_only'],0,'AB')
        except RuntimeError as error:assert str(error)=='MOCK_ONLY_injected_capture_failure'
        else:raise AssertionError('Injected failure disappeared')
        assert (engine.out/'AB_0000_r0_p000.json').exists()
        assert not (engine.out/'closed.jsonl').exists()
    write(ROOT/'generation_receipt.json',dict(status='passed',synthetic_only=True,pretrained_model_calls=0,
        actual_generate_writer_tested=True,B2_to_B1_after_first_EOS=True,packet_stop=True,EOS_stop=True,
        BF16_lossless_storage=True,closed_episode_hashes=True,incomplete_prefixes_not_completed_or_replayed=True,
        source_sha256={n:sha(ROOT/n) for n in ('worker.py','generation_fixture.py')},sealed_test_accessed=False))
    print('Real generation B2/EOS/B1-tail/packet/closed-writer CPU fixture passed.')

if __name__=='__main__':run()
