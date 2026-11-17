"""Small eight-pass shared-weight GRU verifier for NW-01.

The architecture mirrors the recurrent state-access idea in
research_gym.neural.trm.TRMProposer. It is an explicit, testable analogue,
not the pretrained TRM or an actual BPD decomposition.
"""
from __future__ import annotations

import torch
from torch import nn,Tensor
from dataclasses import dataclass

LATENT=48
STEPS=8


@dataclass
class Trace:
    logits: Tensor      # [batch, steps], before sigmoid
    states: Tensor      # [batch, steps, latent]


class RecurrentWitnessVerifier(nn.Module):
    def __init__(self,public_dim:int,latent_dim:int=LATENT,steps:int=STEPS):
        super().__init__()
        self.public_dim=public_dim
        self.latent_dim=latent_dim
        self.steps=steps
        self.public_encoder=nn.Sequential(
            nn.Linear(public_dim,latent_dim),nn.Tanh())
        # Present flag, signed true-output discrepancy and its absolute value.
        self.witness_encoder=nn.Linear(3,latent_dim)
        self.core=nn.GRUCell(latent_dim,latent_dim)
        self.classifier=nn.Linear(latent_dim,1)

    def forward(self,public:Tensor,witness:Tensor,*,patch_after:int|None=None,
                donor_state:Tensor|None=None)->Trace:
        if public.ndim!=2 or public.shape[1]!=self.public_dim:
            raise ValueError("expected [batch, public_dim] public features")
        if witness.shape!=(public.shape[0],3):
            raise ValueError("expected [batch,3] witness features")
        if patch_after is not None and (patch_after<0 or patch_after>=self.steps):
            raise ValueError("invalid recurrent patch pass")
        if (patch_after is None)!=(donor_state is None):
            raise ValueError("donor state and patch pass are paired")
        encoded=self.public_encoder(public)
        message=self.witness_encoder(witness)
        state=torch.zeros_like(encoded)
        traces=[]
        predictions=[]
        for t in range(self.steps):
            # The oracle can influence the recurrent state only at t=0.
            inp=encoded+message if t==0 else encoded
            state=self.core(inp,state)
            if t==patch_after:
                assert donor_state is not None
                if donor_state.shape!=state.shape:
                    raise ValueError("donor has wrong shape")
                state=donor_state
            traces.append(state)
            predictions.append(self.classifier(state).squeeze(-1))
        return Trace(torch.stack(predictions,dim=1),
                     torch.stack(traces,dim=1))


def witness_vectors(betas:Tensor,mode:str):
    """Two labels per root: clean 0; hidden 1. Public pair stays unchanged."""
    if betas.ndim!=1:raise ValueError("expected 1D beta vector")
    if mode not in ("public","authentic","sham"):
        raise ValueError(mode)
    if not torch.all((betas.abs()>=1)&(betas.abs()<=2)):
        raise ValueError("hidden beta must be signed ±1 or ±2")
    n=betas.shape[0]
    out=torch.zeros(n*2,3,dtype=torch.float32,device=betas.device)
    if mode!="public":
        out[:,0]=1.0
        out[1::2,1]=betas/2.0
        out[1::2,2]=betas.abs()/2.0
    return out


def root_paired_targets(n:int,device=None):
    return torch.tensor([0.0,1.0],device=device).repeat(n)


def root_paired_public(features:Tensor):
    if features.ndim!=2:raise ValueError("public features must be [roots,dim]")
    return features.repeat_interleave(2,dim=0)


def matched_world_donor(states:Tensor):
    """Opposite clean/hidden donor for the SAME root, SAME recurrent pass."""
    if states.ndim!=2 or states.shape[0]%2:
        raise ValueError("expected [2*roots, latent]")
    return states.reshape(-1,2,states.shape[1]).flip(1).reshape_as(states)
