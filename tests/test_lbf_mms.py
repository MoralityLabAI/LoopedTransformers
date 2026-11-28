import sys
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parents[1]/'research'/'lite_braille_fresco'
sys.path.insert(0,str(HERE))


def test_lbf_native_recurrent_hooks():
    import lbf20d_learning as d
    import lbf20f_learning as f
    from research_gym.interp.mms import MechanisticMemory
    torch.set_num_threads(1)
    torch.manual_seed(12)
    for model,inputs,visits in [(d.Net('recur3'),(torch.randn(4,d.DIM),),3),
                               (f.Net('recur4'),(torch.randn(4,8,f.NFEAT),torch.eye(8).expand(4,-1,-1)),4)]:
        memory=MechanisticMemory(model.blocks[0],mib=.1,rank=4,mode='observe')
        for step in range(2):
            model.zero_grad()
            with memory.capture(torch.arange(4),step,visits//2,visits):
                model(*inputs).square().mean().backward()
        assert memory.bank.hits==4


def test_lbf_transfer_shared_length_banks():
    import lbf19_learned_transfer as g
    from research_gym.interp.mms import MechanisticMemory, SharedPagePool
    torch.set_num_threads(1)
    model=g.make_arm('rec_prior_hs');pool=SharedPagePool(.1)
    memories={}
    for length in (8,9):
        bank=MechanisticMemory(model.blocks[0].cand,mib=.1,rank=4,mode='observe')
        bank.page_pool,bank.site_name=pool,str(length);memories[length]=bank
    for step in range(4):
        length=8+step%2
        word='a'*length
        rules=(('a','b'),)
        bt=g.to_batch([(0,rules,word,g.exact_support(rules,word,g.KTRAIN))])
        model.zero_grad()
        with memories[length].capture(torch.tensor([0]),step,1,g.KTRAIN):
            model(bt,g.KTRAIN).square().mean().backward()
    assert all(m.bank.hits==1 for m in memories.values())
    assert pool.used<=pool.budget_bytes


def test_lbf_exact_budget_preserves_answers():
    from lbf20f_csp import Case,FULL
    from lbf20m_ram import run_case
    # One independent residual component and cut vertex, exact witness checks.
    c=Case('mms_smoke','sparse',3,(FULL,)*3,((0,1,(1,2,4,8)),(1,2,(FULL,)*4)))
    qs=[{}, {1:1},{1:2},{1:1},{1:2}]
    low=run_case(c,(0,),qs,0)
    high=run_case(c,(0,),qs,.1)
    assert low['answers']==high['answers']
    assert low['charged_bytes']==0
    assert high['stats'].get('query_hits',0)>0
    assert high['peak_charged_bytes']<=high['cap_bytes']
