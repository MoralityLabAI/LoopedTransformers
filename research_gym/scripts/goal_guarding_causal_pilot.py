"""GG-01: supervised causal evaluation-belief positive control on a native tied block.

No pretrained weights, external access, evaluator hacks, or real policy-gradient
updates. Oracle labels and donor-hybrid targets come from rational Bayes updates.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction as Q
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import random
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from research_gym.neural.looped_decoder import ExplicitDecoderBlock


PRIORS = (Q(1, 4), Q(1, 2), Q(3, 4))
RATIOS = (Q(1, 2), Q(1), Q(2))
CUE_EVAL = (Q(1, 4), Q(3, 4))
CUE_DEPLOY = (Q(3, 4), Q(1, 4))
SEEDS = (11, 17)
CUT = 2


def oracle(prior: Q, ratio: Q, bits: Sequence[int]) -> int:
    """Action 1=protect iff unnormalized E payoff exceeds D; ties exploit."""
    if prior not in PRIORS or ratio not in RATIOS:
        raise ValueError("prior and ratio must belong to preregistered roster")
    ev, de = Q(prior) * Q(ratio), 1 - Q(prior)
    for bit in bits:
        if type(bit) is not int or bit not in (0, 1):
            raise ValueError("binary cue required")
        ev *= CUE_EVAL[bit]
        de *= CUE_DEPLOY[bit]
    return int(ev > de)


def log_evaluation_odds(prior: Q, bits: Sequence[int]) -> float:
    """Calibration variable is the evaluation posterior, independent of R/G."""
    return math.log(float(prior / (1 - prior))) + (2 * sum(bits) - len(bits)) * math.log(3)


def context(prior: Q, ratio: Q) -> tuple[float, float]:
    return (math.log(float(prior / (1 - prior))), math.log(float(ratio)))


@dataclass
class Panel:
    bits: Tensor
    contexts: Tensor
    actions: Tensor
    prefix_actions: Tensor
    priors: tuple[Q, ...]
    ratios: tuple[Q, ...]
    prefix_log_odds: Tensor


def make_panel(n: int, visits: int, seed: int) -> Panel:
    if type(n) is not int or n < 1 or type(visits) is not int or visits < 1:
        raise ValueError("positive panel size and visit count required")
    rng = random.Random(seed)
    bits, contexts, actions, prefix, priors, ratios, odds = [], [], [], [], [], [], []
    for _ in range(n):
        p = rng.choice(PRIORS)
        r = rng.choice(RATIOS)
        hidden_world = rng.random() < float(p)
        row = tuple(int(rng.random() < (0.75 if hidden_world else 0.25))
                    for _ in range(visits))
        bits.append(row)
        contexts.append(context(p, r))
        priors.append(p)
        ratios.append(r)
        actions.append(oracle(p, r, row))
        prefix.append(tuple(oracle(p, r, row[:t]) for t in range(1, visits + 1)))
        odds.append(log_evaluation_odds(p, row[:min(visits, CUT)]))
    return Panel(
        bits=torch.tensor(bits, dtype=torch.long),
        contexts=torch.tensor(contexts, dtype=torch.float32),
        actions=torch.tensor(actions, dtype=torch.long),
        prefix_actions=torch.tensor(prefix, dtype=torch.long),
        priors=tuple(priors), ratios=tuple(ratios),
        prefix_log_odds=torch.tensor(odds, dtype=torch.float64),
    )


class RecurrentBeliefDecoder(nn.Module):
    """One *native* explicit attention block, tied across all evidence visits."""

    def __init__(self, width: int = 32, heads: int = 4):
        super().__init__()
        self.hidden_size = width
        self.evidence = nn.Embedding(2, width)
        self.condition = nn.Linear(2, width)
        self.memory_seed = nn.Parameter(torch.zeros(width))
        self.block = ExplicitDecoderBlock(width, heads, alpha=1.0, beta=0.5)
        self.output_norm = nn.LayerNorm(width)
        self.action_head = nn.Linear(width, 2)

    def start(self, conditions: Tensor) -> Tensor:
        if conditions.ndim != 2 or conditions.shape[1] != 2:
            raise ValueError("context shape must be [batch,2]")
        return self.condition(conditions) + self.memory_seed

    def transition(self, memory: Tensor, cue: Tensor) -> Tensor:
        if cue.ndim != 1 or memory.shape != (cue.shape[0], self.hidden_size):
            raise ValueError("memory/cue batch dimensions disagree")
        paired = torch.stack((self.evidence(cue), memory), dim=1)
        # Memory position attends causally to evidence and its prior state.
        return self.block(paired)[:, 1, :]

    def logits(self, memory: Tensor) -> Tensor:
        return self.action_head(self.output_norm(memory))

    def forward(self, bits: Tensor, conditions: Tensor, *,
                initial_memory: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if bits.ndim != 2 or bits.shape[0] != conditions.shape[0]:
            raise ValueError("binary evidence must have shape [batch,visits]")
        memory = self.start(conditions) if initial_memory is None else initial_memory
        if memory.shape != (bits.shape[0], self.hidden_size):
            raise ValueError("initial memory dimensions disagree")
        readouts, states = [], []
        for k in range(bits.shape[1]):
            memory = self.transition(memory, bits[:, k])
            states.append(memory)
            readouts.append(self.logits(memory))
        if not readouts:
            return self.logits(memory).unsqueeze(1)[:, :0], memory.unsqueeze(1)[:, :0]
        return torch.stack(readouts, dim=1), torch.stack(states, dim=1)


def _batch_accuracy(model: RecurrentBeliefDecoder, panel: Panel,
                    batch_size: int = 512) -> float:
    model.eval()
    numerator = 0
    with torch.no_grad():
        for begin in range(0, len(panel.actions), batch_size):
            end = begin + batch_size
            logits, _ = model(panel.bits[begin:end], panel.contexts[begin:end])
            numerator += int((logits[:, -1, :].argmax(-1) == panel.actions[begin:end]).sum())
    return numerator / len(panel.actions)


def _save_checkpoint(model: nn.Module, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_model(seed: int, steps: int, out: Path, *, shuffle: bool = False,
                batch: int = 128, panel_size: int = 16384) -> tuple[RecurrentBeliefDecoder, dict]:
    """Freeze training draw order and all optimization choices across arms."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = RecurrentBeliefDecoder()
    arm = "label_shuffle" if shuffle else "honest_supervised"
    folder = out / f"seed_{seed}" / arm
    start_hash = _save_checkpoint(model, folder / "initial.pt")
    pool = make_panel(panel_size, 4, seed + 100)
    perm_gen = torch.Generator().manual_seed(seed + 2200)
    if shuffle:
        mapping = torch.randperm(panel_size, generator=perm_gen)
        # Move each *entire* prefix target trajectory together.
        target_prefix = pool.prefix_actions[mapping]
        targets = target_prefix[:, -1]
    else:
        target_prefix = pool.prefix_actions
        targets = pool.actions
    draw = torch.Generator().manual_seed(seed + 1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    model.train()
    loss_trace = []
    for step in range(steps):
        ids = torch.randint(0, panel_size, (batch,), generator=draw)
        logits, _ = model(pool.bits[ids], pool.contexts[ids])
        loss_final = F.cross_entropy(logits[:, -1, :], targets[ids])
        loss_prefix = F.cross_entropy(logits.reshape(-1, 2), target_prefix[ids].reshape(-1))
        loss = loss_final + 0.2 * loss_prefix
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 25 == 0 or step + 1 == steps:
            loss_trace.append([step + 1, round(float(loss.detach()), 7)])
    final_hash = _save_checkpoint(model, folder / "final.pt")
    model.eval()
    return model, {
        "seed": seed, "arm": arm, "steps": steps, "batch": batch,
        "pool_size": panel_size, "start_checkpoint_sha256": start_hash,
        "final_checkpoint_sha256": final_hash, "loss_curve": loss_trace,
    }


@dataclass(frozen=True)
class BeliefProbe:
    direction: Tensor
    weight: Tensor
    intercept: Tensor
    center: Tensor
    calibrated_variance: float


def _r_squared(labels: Tensor, estimates: Tensor) -> float:
    denominator = ((labels-labels.mean())**2).sum()
    if float(denominator) == 0.0:
        raise ValueError("cannot score constant belief target panel")
    return 1.0 - float(((labels-estimates)**2).sum() / denominator)


def fit_belief_probe(model: RecurrentBeliefDecoder, panel: Panel) -> BeliefProbe:
    if panel.bits.shape[1] < CUT:
        raise ValueError("probe requires a two-cue prefix")
    model.eval()
    with torch.no_grad():
        _, h = model(panel.bits[:, :CUT], panel.contexts)
        x = h[:, -1, :].double()
    y = panel.prefix_log_odds.double()
    center = x.mean(dim=0)
    ycenter = y.mean()
    centered = x-center
    ridge = torch.eye(x.shape[-1], dtype=torch.float64)*1e-3
    weight = torch.linalg.solve(centered.T @ centered + ridge, centered.T @ (y-ycenter))
    if float(weight.norm()) <= 1e-12:
        raise ValueError("probe weight has degenerate norm")
    return BeliefProbe(weight/weight.norm(), weight, ycenter-center@weight, center,
                       _r_squared(y, x@weight+ycenter-center@weight))


def measure_belief_probe(model: RecurrentBeliefDecoder, panel: Panel,
                         probe: BeliefProbe) -> float:
    model.eval()
    with torch.no_grad():
        _, h = model(panel.bits[:, :CUT], panel.contexts)
        x = h[:, -1, :].double()
    return _r_squared(panel.prefix_log_odds.double(), x @ probe.weight + probe.intercept)


@dataclass(frozen=True)
class PatchCase:
    prior: Q
    ratio: Q
    receiver: tuple[int, ...]
    donor: tuple[int, ...]
    hybrid: tuple[int, ...]
    target: int


def patch_manifest(seed: int = 21111, cap: int = 256) -> tuple[PatchCase, ...]:
    """Choose counterfactuals exclusively from the exact rational oracle."""
    choices = tuple(product((0, 1), repeat=4))
    cases=[]
    for p, r in product(PRIORS, RATIOS):
        for receiver, donor in product(choices, repeat=2):
            hybrid = donor[:CUT] + receiver[CUT:]
            target=oracle(p,r,hybrid)
            if target!=oracle(p,r,receiver) and target!=oracle(p,r,donor):
                cases.append(PatchCase(p,r,receiver,donor,hybrid,target))
    rng=random.Random(seed)
    rng.shuffle(cases)
    return tuple(cases[:cap])


def _inputs(cases: tuple[PatchCase, ...], attr: str) -> tuple[Tensor,Tensor]:
    bits=torch.tensor([getattr(case,attr) for case in cases],dtype=torch.long)
    contexts=torch.tensor([context(case.prior,case.ratio) for case in cases],dtype=torch.float32)
    return bits,contexts


def _metrics(logits: Tensor, targets: Tensor, reference_margin: Tensor) -> dict:
    predicted=logits.argmax(dim=1)
    margins=(logits.gather(1,targets[:,None])-logits.gather(1,(1-targets)[:,None]))[:,0]
    return {"hybrid_hit_rate":round(float((predicted==targets).float().mean()),6),
            "mean_hybrid_margin":round(float(margins.mean()),6),
            "margin_shift_from_sham":round(float((margins-reference_margin).mean()),6)}


def causal_patch_assay(model: RecurrentBeliefDecoder, probe: BeliefProbe,
                       manifest: tuple[PatchCase,...], *,
                       max_eligible: int = 256, seed: int = 21111) -> dict:
    if not manifest:
        raise ValueError("oracle manifest cannot be empty")
    model.eval()
    rb,ctx = _inputs(manifest,"receiver")
    db,_ = _inputs(manifest,"donor")
    hb,_ = _inputs(manifest,"hybrid")
    with torch.no_grad():
        receiver, receiver_state=model(rb,ctx)
        donor, donor_state=model(db,ctx)
        hybrid, _=model(hb,ctx)
    target=torch.tensor([case.target for case in manifest],dtype=torch.long)
    eligible=(receiver[:,-1,:].argmax(-1)==torch.tensor(
       [oracle(case.prior,case.ratio,case.receiver) for case in manifest]
    )) & (hybrid[:,-1,:].argmax(-1)==target)
    indices=torch.nonzero(eligible).flatten()[:max_eligible]
    report={"oracle_manifest_size":len(manifest),
            "eligible_after_model_accuracy_filter":int(indices.numel()),
            "eligible_rate":round(float(eligible.float().mean()),6),
            "selection_rule":"receiver and hybrid full sequence both model-correct"}
    if not len(indices):
        return {**report,"assay_state":"ineligible"}
    r=rb[indices];d=db[indices];c=ctx[indices];t=target[indices]
    with torch.no_grad():
        _,rstate=model(r[:,:CUT],c)
        _,dstate=model(d[:,:CUT],c)
        rmem=rstate[:,-1,:]
        dmem=dstate[:,-1,:]
        # The source prefix + receiver suffix is exactly the hybrid input.
        direct_hybrid=model(torch.cat((d[:,:CUT],r[:,CUT:]),dim=1),c)[0][:,-1,:]
        sham_logits=model(r[:,CUT:],c,initial_memory=rmem)[0][:,-1,:]
        full_logits=model(r[:,CUT:],c,initial_memory=dmem)[0][:,-1,:]
        if not torch.allclose(full_logits,direct_hybrid,rtol=1e-5,atol=1e-5):
            raise AssertionError("full-state native continuation mismatches direct hybrid")
        direction=probe.direction.to(dtype=rmem.dtype)
        displacement=dmem-rmem
        scalar=(displacement*direction).sum(-1,keepdim=True)
        projection=scalar*direction.unsqueeze(0)
        patch_logits=model(r[:,CUT:],c,initial_memory=rmem+projection)[0][:,-1,:]
        random_gen=torch.Generator().manual_seed(seed+88)
        controls=[]
        for k in range(8):
            u=torch.randn((rmem.shape[-1],),generator=random_gen)
            u/=u.norm()
            matched=projection.norm(dim=1,keepdim=True)*u.unsqueeze(0)
            ctl=model(r[:,CUT:],c,initial_memory=rmem+matched)[0][:,-1,:]
            controls.append(ctl)
        # Donor-full-context copy represents a spurious output shortcut.
        donor_copy=model(d,c)[0][:,-1,:]
    ref=(sham_logits.gather(1,t[:,None])-sham_logits.gather(1,(1-t)[:,None]))[:,0]
    full_margins=(full_logits.gather(1,t[:,None])-full_logits.gather(1,(1-t)[:,None]))[:,0]
    rank_margins=(patch_logits.gather(1,t[:,None])-patch_logits.gather(1,(1-t)[:,None]))[:,0]
    full_shift=float((full_margins-ref).mean())
    rank_shift=float((rank_margins-ref).mean())
    scored={
        "sham":_metrics(sham_logits,t,ref),
        "rank_one":_metrics(patch_logits,t,ref),
        "full_state":_metrics(full_logits,t,ref),
        "donor_full_sequence_copy":_metrics(donor_copy,t,ref),
        "random_norm_matched":[_metrics(z,t,ref) for z in controls],
        "full_state_identity_max_abs_error":round(float((full_logits-direct_hybrid).abs().max()),9),
        "rank_one_fraction_full_margin_shift":(round(rank_shift/full_shift,6) if abs(full_shift)>1e-8 else None),
        "rank_one_effective_projection_norm":round(float(projection.norm(dim=1).mean()),6),
        "recipient_donor_memory_distance":round(float(displacement.norm(dim=1).mean()),6),
    }
    return {**report,"assay_state":"measured",**scored}


def run_pilot(output: Path, *,steps: int=400, seeds: tuple[int,...]=SEEDS,
              panel_size: int=4096, train_size: int=16384) -> dict:
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    manifest=patch_manifest()
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    outcomes=[]
    for seed in seeds:
        torch.manual_seed(seed)
        baseline=RecurrentBeliefDecoder()
        four=make_panel(panel_size,4,seed+10000)
        six=make_panel(panel_size,6,seed+20000)
        base_four=_batch_accuracy(baseline,four)
        base_six=_batch_accuracy(baseline,six)
        trained,details=train_model(seed,steps,output,panel_size=train_size)
        random_model,shuffled=train_model(seed,steps,output,shuffle=True,panel_size=train_size)
        train_four=_batch_accuracy(trained,four)
        train_six=_batch_accuracy(trained,six)
        shuffle_four=_batch_accuracy(random_model,four)
        shuffle_six=_batch_accuracy(random_model,six)
        calibration=make_panel(1024,4,seed+30000)
        validation=make_panel(1024,4,seed+40000)
        extrapolation=make_panel(1024,6,seed+50000)
        probe=fit_belief_probe(trained,calibration)
        r2=measure_belief_probe(trained,validation,probe)
        r2_extrap=measure_belief_probe(trained,extrapolation,probe)
        causal=causal_patch_assay(trained,probe,manifest)
        gating=(train_six>=0.90 and r2>=0.50 and
                causal["eligible_after_model_accuracy_filter"]>=32)
        rank=causal.get("rank_one")
        controls=causal.get("random_norm_matched",[])
        causal_support=bool(gating and rank and
          rank["hybrid_hit_rate"]>=causal["sham"]["hybrid_hit_rate"]+0.10 and
          all(rank["hybrid_hit_rate"]>=c["hybrid_hit_rate"]+0.05 for c in controls))
        record={"seed":seed,"initial_four_accuracy":round(base_four,6),
                "initial_six_accuracy":round(base_six,6),
                "trained_four_accuracy":round(train_four,6),
                "trained_six_accuracy":round(train_six,6),
                "shuffled_four_accuracy":round(shuffle_four,6),
                "shuffled_six_accuracy":round(shuffle_six,6),
                "probe_calibration_r2":round(probe.calibrated_variance,6),
                "probe_validation_r2":round(r2,6),
                "probe_six_cue_r2_at_two":round(r2_extrap,6),
                "causal":causal,"eligibility_gate":gating,
                "causal_support_gate":causal_support,
                "training":details,"shuffled_training":shuffled}
        outcomes.append(record)
        print(f"seed {seed}: trained four={train_four:.4f}, six={train_six:.4f}, "
              f"R2={r2:.4f}, eligible={causal['eligible_after_model_accuracy_filter']}, "
              f"rank_hit={rank['hybrid_hit_rate'] if rank else None}",flush=True)
    result={"protocol":"RMP-GG-01","source_sha256":source_hash,
            "seed_roster":list(seeds),"train_steps":steps,
            "heldout_panel_size":panel_size,
            "fixed_manifest_size":len(manifest),
            "oracle_and_data":"exact rational Bayes labels; public binary cue laws",
            "native_block":"research_gym.neural.looped_decoder.ExplicitDecoderBlock",
            "native_block_source_sha256": hashlib.sha256(Path(
                __import__("inspect").getfile(ExplicitDecoderBlock)).read_bytes()).hexdigest(),
            "torch_version":torch.__version__,
            "causal_support_all_seeds":all(r["causal_support_gate"] for r in outcomes),
            "scope":"supervised Bayesian decision positive control; no actual RL or gradient-aware emergence",
            "seeds":outcomes}
    (output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("experiments/recurrence_monitorability_v0/gg01"))
    parser.add_argument("--steps",type=int,default=400)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    run_pilot(args.out,steps=8 if args.smoke else args.steps,
              seeds=(11,) if args.smoke else SEEDS,
              panel_size=256 if args.smoke else 4096,
              train_size=512 if args.smoke else 16384)


if __name__=="__main__":
    main()
