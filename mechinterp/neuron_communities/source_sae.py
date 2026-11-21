"""NEC-03D source-only SAE on native 64-D GELU MLP activations.

This is a development instrument, not a general-purpose sparse feature
discovery claim. Fit on PEF-07 train contexts 0:80, select top-attributed
features on the next 80 train contexts. Neither phase opens the 48
heldout context groups, and PEF-07 weights remain frozen.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from mechinterp.neuron_communities.core import decode
from research.pef07.pef07 import D, L, as_tensor, model_digest

WIDTH = D * 2
FEATURES = 128
TRAIN_CONTEXTS = 80
DEV_CONTEXTS = 80
STEPS = 240
BATCH = 256
SEED_OFFSET = 30003


class SourceSAE(nn.Module):
    def __init__(self, features: int = FEATURES):
        super().__init__()
        self.encoder = nn.Linear(WIDTH, features)
        self.decoder = nn.Linear(features, WIDTH)

    def forward(self, normalized):
        z = F.relu(self.encoder(normalized))
        return self.decoder(z), z


def clean_activations(model, bits: torch.Tensor) -> torch.Tensor:
    """Return the true GELU input at mlp_down, memory token, per visit."""
    if bits.ndim != 2 or bits.shape[1] != 12:
        raise ValueError("expected twelve input bits")
    captured = []
    hook = model.block.mlp_down.register_forward_pre_hook(
        lambda module, args: captured.append(args[0][:, 1, :].detach().clone()))
    try:
        with torch.no_grad():
            model.logits(bits)
    finally:
        hook.remove()
    if len(captured) != L:
        raise AssertionError("incorrect shared-block visit count")
    result = torch.stack(captured, dim=1)
    if result.shape != (len(bits), L, WIDTH):
        raise AssertionError("not the physical GELU basis")
    return result


def collect_source(model, context_rows, *, context_batch: int = 8):
    """Stack full 16-state quartets, preserving source contexts."""
    out = []
    if len(context_rows) % 16:
        raise AssertionError("source contexts require complete quartets")
    for offset in range(0, len(context_rows), context_batch*16):
        chunk = context_rows[offset:offset+context_batch*16]
        out.append(clean_activations(model, as_tensor(chunk)))
    return torch.cat(out, dim=0)


def seal_array(t: torch.Tensor) -> str:
    h = hashlib.sha256()
    arr = t.detach().cpu().contiguous()
    h.update(str(tuple(arr.shape)).encode())
    h.update(str(arr.dtype).encode())
    h.update(bytes(arr.view(torch.uint8).flatten().tolist()))
    return h.hexdigest()


def fit_source_sae(model, train_rows, *, seed: int,
                   steps: int = STEPS) -> tuple[SourceSAE, dict, torch.Tensor, torch.Tensor]:
    """Fixed hyperparameters, CPU FP32 and source-train-only normalization."""
    if steps < 1 or len(train_rows) != 16*TRAIN_CONTEXTS:
        raise ValueError("SAE fit requires 80 complete training source contexts")
    before = model_digest(model)
    states = collect_source(model, train_rows).reshape(-1, WIDTH)
    if not torch.isfinite(states).all():
        raise AssertionError("nonfinite source activations")
    mean, scale = states.mean(0), states.std(0, unbiased=False).clamp_min(1e-3)
    data = (states-mean)/scale
    torch.manual_seed(seed+SEED_OFFSET)
    sae = SourceSAE()
    opt = torch.optim.AdamW(sae.parameters(), lr=1e-3, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed+SEED_OFFSET+29)
    first = last = None
    for step in range(steps):
        chosen = torch.randint(len(data), (BATCH,), generator=generator)
        pred, codes = sae(data[chosen])
        loss = (pred-data[chosen]).square().mean()+0.002*codes.abs().mean()
        if not torch.isfinite(loss):
            raise AssertionError("SAE loss nonfinite")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.detach())
        last = float(loss.detach())
    sae.eval()
    with torch.no_grad():
        pred, codes = sae(data)
        mse = float((pred-data).square().mean())
        rms = float(data.square().mean())
        active = float((codes>1e-7).float().sum(1).mean())
    if model_digest(model) != before:
        raise AssertionError("SAE fit mutated frozen trunk")
    receipt = {"status":"development-sae-source-only",
               "fitted_trunk_model_sha256":before,
               "n_contexts_train":TRAIN_CONTEXTS,
               "n_contexts_feature_selection":DEV_CONTEXTS,
               "n_holdout_contexts_seen_by_fit":0,
               "source_activation_count":len(states),
               "model_input_dimension":WIDTH,
               "sae_features":FEATURES,
               "sae_optimizer_steps":steps,
               "sparsity_lambda":0.002,
               "training_loss_initial":first,
               "training_loss_final":last,
               "training_reconstruction_mse_normalized":mse,
               "train_variance_normalized":rms,
               "mean_active_features":active,
               "normalization_mean_sha256":seal_array(mean),
               "normalization_scale_sha256":seal_array(scale)}
    return sae, receipt, mean, scale


def dev_feature_importance(model, sae, mean, scale, rows) -> torch.Tensor:
    """Source-only feature activation * downstream-gradient attribution."""
    if len(rows) != 16*DEV_CONTEXTS:
        raise ValueError("feature selection requires all separate train-dev contexts")
    before = model_digest(model)
    out = torch.zeros((L, FEATURES), dtype=torch.float64)
    physical_decoder = sae.decoder.weight.detach().T * scale[None, :]
    for offset in range(0, len(rows), 16*4):
        bits = as_tensor(rows[offset:offset+16*4])
        captured = []
        def capture(module, args):
            captured.append(args[0])
        handle = model.block.mlp_down.register_forward_pre_hook(capture)
        try:
            logits = model.logits(bits)
            choice = logits.argmax(dim=1)
            diff = logits.gather(1, choice[:, None]).squeeze(1) - (
                logits.gather(1,(1-choice)[:, None]).squeeze(1))
            if len(captured) != L:
                raise AssertionError("gradient visits missing")
            gradients = torch.autograd.grad(diff.sum(), captured)
        finally:
            handle.remove()
        for visit, (a,g) in enumerate(zip(captured, gradients)):
            z = F.relu(sae.encoder(
                (a[:,1,:].detach()-mean)/scale)).detach()
            influence = z * (g[:,1,:].detach() @ physical_decoder.T)
            out[visit] += influence.abs().double().sum(0)
    if model_digest(model) != before:
        raise AssertionError("feature selection mutated trunk")
    return out / len(rows)


def feature_rosters(scores, candidate_nodes: list[str], *, seed: int = 1907):
    """Same number of SAE features as native occurrences at each visit."""
    counts = {}
    for key in candidate_nodes:
        visit, idx = decode(key)
        if visit not in range(L) or idx not in range(WIDTH):
            raise ValueError("candidate occurrence outside frozen model")
        counts[visit] = counts.get(visit, 0)+1
    ranked = {}
    pseudo = {}
    for visit, n in sorted(counts.items()):
        order = sorted(range(FEATURES),
                       key=lambda j: (-float(scores[visit,j]), j))
        top = order[:n]
        # Deterministic cyclic feature control; distinct from top features.
        pool = order[n:]
        if len(pool) < n:
            raise ValueError("insufficient SAE features for sham")
        start = (seed+visit*103) % (len(pool)-n+1)
        ranked[visit] = top
        pseudo[visit] = pool[start:start+n]
    return ranked, pseudo


def physical_feature_delta(sae, mean, scale, states, selected: list[int]):
    """Additive feature contribution in physical GELU coordinates."""
    if states.ndim != 2 or states.shape[1] != WIDTH:
        raise ValueError("wrong physical GELU activation dimension")
    if any(j not in range(FEATURES) for j in selected):
        raise ValueError("SAE feature outside dictionary")
    if not selected:
        return torch.zeros_like(states)
    codes = F.relu(sae.encoder((states-mean)/scale))
    columns = sae.decoder.weight[:,selected].T * scale[None,:]
    return codes[:,selected] @ columns


def save_sae(path: Path, sae, receipt, mean, scale, *, sources: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    package={"state":sae.state_dict(),"mean":mean,"scale":scale,
             "receipt":receipt,"sources":sources}
    torch.save(package,path)
    raw=path.read_bytes()
    receipt=dict(receipt)
    receipt["sae_checkpoint_file"]=path.name
    receipt["sae_checkpoint_sha256"]=hashlib.sha256(raw).hexdigest()
    receipt["source_roster_sha256"]=hashlib.sha256(
        (json.dumps(sources,sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest()
    return receipt
