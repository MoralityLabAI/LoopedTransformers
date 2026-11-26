"""MMS engineering tests on the native Mathlib trainer, no Lean corpus or verifier."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lcgrt import tactic_lm as T, tactic_mms as M


def opts(**kw):
    return dict(mib=.5,mode='observe',sites='auto',schedule='round_robin',
        rank=4,topk=2,ratio=.2,refresh=8,max_age=32,visit=-1,
        map_manifest='',map_manifest_sha='') | kw


def model(arch):
    return T.TacticLM(T.TLMConfig(arch=arch,width=16,heads=2,slots=4,layers=1,
        fast_steps=3,slow_every=2,trm_n=2,trm_T=3,trm_sup=3,
        dec_width=16,dec_heads=2,dec_layers=1,mlp_ratio=2))


def batch():
    return dict(enc=torch.tensor([[65,66,67,68],[65,67,67,68]]),
        enc_mask=torch.ones(2,4,dtype=torch.bool),
        dec_in=torch.tensor([[T.BOS,65,66],[T.BOS,66,65]]),
        target=torch.tensor([[65,66,T.EOS],[66,65,T.EOS]]))


@pytest.mark.parametrize('arch',['loop','hrm','trm'])
def test_native_clocks_observe_parity(arch):
    torch.manual_seed(3);torch.set_num_threads(1)
    m=model(arch);base=copy.deepcopy(m)
    mem=M.TacticMemory(m,opts(),2,5)
    b=batch();draw=torch.arange(2);c=base_c=ids=None
    for step in range(12):
        m.zero_grad();base.zero_grad()
        if arch=='trm':
            ids,stage=M.active_rows(draw,c,ids)
            with mem.capture(step,ids,stage=stage):
                l,c,_=m.act_step(c,b,np.random.default_rng(step));l.backward()
            l2,base_c,_=base.act_step(base_c,b,np.random.default_rng(step));l2.backward()
            assert torch.equal(c['y'],base_c['y']) and not c['y'].requires_grad
            assert torch.equal(c['z'],base_c['z']) and not c['z'].requires_grad
        else:
            depth=(3,5)[step%2]
            with mem.capture(step,draw,depth=depth): T.nll(m,b,depth).backward()
            T.nll(base,b,depth).backward()
        for p,q in zip(m.parameters(),base.parameters(),strict=True):
            if p.grad is None: assert q.grad is None
            else: assert torch.equal(p.grad,q.grad)
    assert mem.memory.pool.used>0 and mem.memory.pool.used<=mem.memory.pool.budget_bytes
    assert sum(s.bank.hits for s in mem.memory.sites.values() if s.bank is not None)>0


def test_actual_carried_row_not_draw():
    ids,stage=M.active_rows(torch.tensor([9,8]),{'halted':torch.tensor([False,True]),
        'steps':torch.tensor([2,3])},torch.tensor([2,3]))
    assert ids.tolist()==[2,8] and stage.tolist()==[2,0]


def test_native_trainer_checkpoint_resume(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    data=tmp_path/'data';root=data/'fixture';packed=root/'packed_tlm';packed.mkdir(parents=True)
    (root/'manifest.json').write_text(json.dumps({'split_sha256':{'train':'synthetic','val':'synthetic-val'}}))
    for split in ('train','val'):
        enc=np.zeros((2,T.CTX),np.uint8);enc[:,:3]=[65,66,67]
        dec=np.zeros((2,T.TGT),np.uint8);dec[:,:2]=[65,66]
        np.savez(packed/f'{split}.npz',enc=enc,dec=dec,enc_len=np.array([3,3]),
            dec_len=np.array([2,2]),terminal=np.array([True,True]))
    monkeypatch.setattr(T.config,'DATA_ROOT',data)
    monkeypatch.setattr(T.hw,'check',lambda *args:(True,'ok',{}))
    torch.use_deterministic_algorithms(True)
    cli=['--data','fixture','--preset','tlm-trm-tiny','--cfg','width=16','heads=2',
        'dec_width=16','dec_heads=2','layers=1','mlp_ratio=2','trm_n=1','trm_T=2','trm_sup=2',
        '--device','cpu','--steps','4','--batch','2','--threads','1','--warmup','1',
        '--log-every','4','--eval-every','4','--val-items','2','--val-batch','2',
        '--save-every','2','--train-pool-rows','1','--mms-mib','1','--mms-mode','bpd','--mms-sites','trm0.ff',
        '--mms-rank','4','--mms-topk','2']
    whole,resumed=tmp_path/'whole',tmp_path/'resumed'
    assert T.main(cli+['--out',str(whole)])==0
    assert T.main(cli+['--out',str(resumed),'--stop-at','2'])==0
    assert T.main(cli+['--out',str(resumed)])==0
    _,a=T.load_verified(whole,'cpu');_,b=T.load_verified(resumed,'cpu')
    assert a['identity_hash']==b['identity_hash'] and a['fresh']==b['fresh']
    assert a['identity']['train_pool']['rows']==1 and a['act_row_ids'].unique().numel()==1
    for k in ('model','ema','act_carry'):
        for name in a[k]:assert torch.equal(a[k][name],b[k][name]),(k,name)
    assert torch.equal(a['act_row_ids'],b['act_row_ids'])
    assert a['mms']['pool']==b['mms']['pool']
    sa,sb=a['mms']['sites']['trm0.ff']['bank'],b['mms']['sites']['trm0.ff']['bank']
    assert sa['hits']==sb['hits'] and sa['hits']>0
    for page in sa['pages']:
        for x,y in zip(sa['pages'][page],sb['pages'][page],strict=True):assert torch.equal(x,y)
    assert json.loads((resumed/'mms_receipt.json').read_text())['step']==4
