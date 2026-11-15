"""Fast, checkpoint-free CPU regression tests for local causal fiber gyms."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB
from mechinterp.dark_fiber_gyms import arg03_roster as R
from mechinterp.dark_fiber_gyms import core as C
from mechinterp.dark_fiber_gyms import loopscope04_trm as TR
from mechinterp.dark_fiber_gyms import sudoku_roster as SU
from mechinterp.dark_fiber_gyms import observability as OBS


def test_sphere_swap_preserves_all_head_logits_and_radius():
    torch.manual_seed(4)
    w = torch.randn(3,8)
    src = torch.randn(2,19,8)
    donor = torch.randn_like(src)
    out, cert = C.trm_manifold_swap(src,donor,w,board_start=2,atol=1e-4)
    assert cert["orthogonal_dim"] == 5
    assert torch.equal(out[:,:2],src[:,:2])
    assert (out[:,2:]@w.T-src[:,2:]@w.T).abs().max() < 1e-4
    assert torch.allclose(out[:,2:].norm(dim=-1),src[:,2:].norm(dim=-1),atol=1e-5)
    assert (out[:,2:]-src[:,2:]).norm().item()>1e-4


def test_float16_swap_refused():
    x=torch.randn(1,5,8,dtype=torch.float16)
    with pytest.raises(ValueError,match="float32"):
        C.trm_manifold_swap(x,x,torch.randn(3,8),board_start=1)


def test_nanbeige_context_preserves_last_three_exactly():
    a=torch.randn(16,18,6)
    b=torch.randn(16,18,6)
    out,meta=C.nanbeige_context_swap(a,b,width=6)
    assert meta["positions"] == list(range(9,15))
    assert torch.equal(a[:,-3:],out[:,-3:])
    assert torch.equal(out[:,9:15],b[:,9:15])
    assert torch.equal(out[:,:9],a[:,:9])


class ToyLoop:
    n=1
    def prepare(self,ids):
        ids=ids.float()
        emb=torch.stack([ids,ids.square()/100,ids.sin(),ids.cos()],dim=-1)
        return emb,{}
    def run(self,h,lo,hi,ctx):
        return h+0.17*h.roll(1,dims=1)
    def readout(self,h,ctx):
        w=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
        return torch.log_softmax(h[:,-3:,:]@w.T,dim=-1)


class ToyEngine:
    op_ids=dict(zip("INSR",range(4)))
    loop=ToyLoop()
    def rows(self,row):
        ids=torch.stack([torch.tensor(list(range(10))+[i,j],dtype=torch.long)
                         for i in range(4) for j in range(4)])
        return ids,10


def test_nanbeige_mock_full_readout_equality_and_later_causality():
    r={"case_id":"toy","public_sha256":"0"*64}
    out=NB.evaluate_case(ToyEngine(),r,sigma=0.1,width=5,seed=27,depths=(1,2,4))
    assert out["max_precontinuation_logprob_error"]==0.0
    assert out["baseline"]["1"]==out["context_swap"]["1"]
    assert out["postloop_readout_distance"]["2"]>0.0
    assert len(out["baseline"]["4"])==64
    assert set(out["temperature_controls"]) == {"T0.5","T0.75","T1.25","T2"}
    assert "4" in out["open_loop"]


def test_program_score_layout():
    logp=torch.log_softmax(torch.randn(16,3,10),dim=-1)
    values=C.arg02_program_scores(logp,dict(zip("INSR",range(4))))
    assert len(values)==64
    assert abs(values[0]-float(logp[0,0,0]+logp[0,1,0]+logp[0,2,0]))<1e-6


def test_ridge_diagnostic_and_mcnemar():
    rng=np.random.default_rng(4)
    x=rng.standard_normal((80,4))
    w=rng.standard_normal((5,7))
    y=np.column_stack([np.ones(80),x])@w
    fitted=TR.ridge_fit(x,y,alpha=1e-5)
    assert np.abs(TR.ridge_predict(x,fitted)-y).max()<1e-4
    res=R.exact_mcnemar([True,True,True],[False,False,False])
    assert res["gain"]==3 and res["loss"]==0
    assert res["exact_two_sided_p"]==0.25


def test_sudoku_generator_and_uniqueness():
    import random
    rng=random.Random(9)
    solved=SU.generate_solution(rng)
    assert len(solved)==81 and SU.count_solutions(solved)==1
    puzzle=SU.delete_clues(solved,78,rng)
    assert puzzle is not None and puzzle.count(".")==3
    assert SU.count_solutions(puzzle)==1
    relabel="".join("2" if c=="1" else "1" if c=="2" else c for c in puzzle)
    assert TR.canonical_digit_relabel(puzzle)==TR.canonical_digit_relabel(relabel)


def _write_boards(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=("question","answer"))
        w.writeheader()
        for row in rows:w.writerow(row)


def test_dataset_overlap_rejected(tmp_path):
    old=tmp_path/"old.csv"
    q="."*81
    a=SU.generate_solution(__import__("random").Random(5))
    _write_boards(old,[{"question":q,"answer":a}])
    test=[{"question":q,"answer":a,"id":"test"}]
    with pytest.raises(ValueError,match="overlaps excluded"):
        TR.audit_boards([],test,[old])


def test_nanbeige_dry_run_rejects_hidden_fields(tmp_path,capsys):
    public=tmp_path/"public.jsonl"
    base={"case_id":"toy","public_sha256":"a"*64,"topology":"fork","observed_degree":12,
          "query":{},"demos":[{},{}]}
    public.write_text(json.dumps(base)+"\n")
    assert NB.main(["--public-jsonl",str(public),"--compactification-root",str(tmp_path),
                    "--out-dir",str(tmp_path/"out"),"--dry-run"])==0
    base["future_secret"]="blocked"
    public.write_text(json.dumps(base)+"\n")
    with pytest.raises(ValueError,match="hidden"):
        NB.main(["--public-jsonl",str(public),"--compactification-root",str(tmp_path),
                 "--out-dir",str(tmp_path/"out"),"--dry-run"])


class TinyRecurrentMap:
    def __init__(self):
        self.lm_head=torch.nn.Linear(6,2,bias=False)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            self.lm_head.weight[0,0]=1.0
            self.lm_head.weight[1,1]=1.0

    def act_step(self, zh,zl,x):
        y=zh.clone()
        y[:,:,0]=zh[:,:,0]+0.2*zh[:,:,2]
        y[:,:,1]=zh[:,:,1]+0.1*zh[:,:,3]
        return y,zl


def test_sampled_gramian_local_tangents_exact_and_delayed():
    model=TinyRecurrentMap()
    torch.manual_seed(5)
    zh=torch.randn(1,7,6)
    zl=torch.zeros_like(zh)
    x=torch.zeros_like(zh)
    pairs,certificate=OBS.sphere_tangent_pairs(
        zh,model.lm_head.weight,board_start=2,probes=5,angle=0.03)
    assert pairs.shape==(10,7,6)
    assert certificate["max_readout_error"] < 1e-5
    assert certificate["max_l2_error"] < 1e-5
    result=OBS.sampled_trm_gramian(
        model,zh,zl,x,board_start=2,probes=5,angle=0.03,horizon=3)
    assert result["sampled_rank_relative_1e-3"] >= 1
    assert result["eigenvalues_descending"][0]>0
    assert len(result["per_lag_mean_response"])==3
