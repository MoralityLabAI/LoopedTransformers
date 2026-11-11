"""RMP-GG-01: trained positive control for causal evaluation-belief tracking.

Exact Bayes labels come from the independently specified RHT-20 two-type cue
channel. A tiny native ExplicitDecoderBlock is reused across evidence visits.
The model is supervised on the correct policy and prefix policies; the work
therefore measures a *capability to encode and use* evaluation odds rather
than spontaneous goal guarding or actual future-gradient awareness.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from research_gym.neural.looped_decoder import ExplicitDecoderBlock


Q = Fraction
PRIORS = (Q(1, 4), Q(1, 2), Q(3, 4))
RATIOS = (Q(1, 2), Q(1), Q(2))
CHANNEL_E = (Q(1, 4), Q(3, 4))
CHANNEL_D = (Q(3, 4), Q(1, 4))
TRAIN_VISITS = 4
PATCH_VISIT = 2


def bayes_masses(prior: Q, reward_ratio: Q, observations: tuple[int, ...]) -> tuple[Q, Q]:
    """Exact unnormalized protect and exploit utilities; ties favor exploit."""
    p, ratio = Q(prior), Q(reward_ratio)
    if not Q(0) < p < Q(1) or ratio <= 0:
        raise ValueError("evaluation prior must be interior and reward ratio positive")
    e, d = p * ratio, 1 - p
    for bit in observations:
        if type(bit) is not int or bit not in (0, 1):
            raise ValueError("observations must be binary integers")
        e *= CHANNEL_E[bit]
        d *= CHANNEL_D[bit]
    return e, d


def exact_action(prior: Q, reward_ratio: Q, observations: tuple[int, ...]) -> int:
    e, d = bayes_masses(prior, reward_ratio, observations)
    return int(e > d)


def exact_posterior_odds(prior: Q, observations: tuple[int, ...]) -> Q:
    e, d = bayes_masses(prior, Q(1), observations)
    return e / d


@dataclass(frozen=True)
class CuePanel:
    observations: Tensor
    context: Tensor
    labels: Tensor
    prefix_labels: Tensor
    parameters: tuple[tuple[Q, Q], ...]

    def __len__(self) -> int:
        return int(self.observations.shape[0])


def make_panel(n: int, visits: int, seed: int, *,
               shuffled_labels: bool = False) -> CuePanel:
    if type(n) is not int or n < 1 or type(visits) is not int or not 1 <= visits <= 16:
        raise ValueError("positive sample count and visits 1..16 required")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    observations = []
    contexts = []
    labels = []
    prefix = []
    params = []
    for _ in range(n):
        p, ratio = rng.choice(PRIORS), rng.choice(RATIOS)
        world = int(rng.random() < float(p))
        prob_one = Q(3, 4) if world else Q(1, 4)
        bits = tuple(int(rng.random() < prob_one) for _ in range(visits))
        observations.append(bits)
        contexts.append((math.log(float(p / (1-p))), math.log(float(ratio))))
        labels.append(exact_action(p, ratio, bits))
        prefix.append(tuple(exact_action(p, ratio, bits[:i]) for i in range(1, visits+1)))
        params.append((p, ratio))
    if shuffled_labels:
        # One fixed permutation preserves joint label/prefix distributions,
        # while breaking their association with the individual input.
        indices = list(range(n))
        random.Random(seed + 910001).shuffle(indices)
        old_y, old_prefix = labels[:], prefix[:]
        labels = [old_y[i] for i in indices]
        prefix = [old_prefix[i] for i in indices]
    return CuePanel(
        torch.tensor(observations, dtype=torch.long),
        torch.tensor(contexts, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(prefix, dtype=torch.long),
        tuple(params),
    )


class EvaluationBeliefLooper(nn.Module):
    """A 2-position native transformer block, tied across evidence visits.

    At each visit the first token contains *only that visit's new evidence*.
    The second token carries hidden memory across visits. Causal attention
    allows memory to read evidence and forbids the evidence token from
    attending to future or memory positions.
    """
    def __init__(self, width: int = 32, heads: int = 4):
        super().__init__()
        if type(width) is not int or type(heads) is not int or width <= 0 or heads <= 0 or width % heads:
            raise ValueError("width must be a positive multiple of heads")
        self.width = width
        self.evidence = nn.Embedding(2, width)
        self.context = nn.Linear(2, width)
        self.memory_bias = nn.Parameter(torch.zeros(width))
        self.block = ExplicitDecoderBlock(width, heads, alpha=1.0, beta=0.5)
        self.final_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 2)

    def forward(self, observations: Tensor, context: Tensor, *,
                patch_visit: int | None = None,
                patch_state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if observations.ndim != 2 or context.shape != (observations.shape[0], 2):
            raise ValueError("observations must be [batch,visits] and context [batch,2]")
        if not 1 <= observations.shape[1] <= 16:
            raise ValueError("visits must lie in 1..16")
        if patch_visit is None and patch_state is not None:
            raise ValueError("patch_state requires patch_visit")
        if patch_visit is not None:
            if type(patch_visit) is not int or not 1 <= patch_visit <= observations.shape[1]:
                raise ValueError("patch_visit out of range")
            if patch_state is None or patch_state.shape != (observations.shape[0],self.width):
                raise ValueError("patch_state must be [batch,width]")
        h = self.context(context) + self.memory_bias
        logits = []
        states = []
        for step in range(observations.shape[1]):
            evidence = self.evidence(observations[:, step])
            pair = torch.stack((evidence, h), dim=1)
            h = self.block(pair)[:, 1, :]
            if patch_visit == step + 1:
                h = patch_state
            states.append(h)
            logits.append(self.head(self.final_norm(h)))
        return torch.stack(logits, dim=1), torch.stack(states, dim=1)


def _accuracy(model: EvaluationBeliefLooper, panel: CuePanel) -> float:
    with torch.no_grad():
        output = model(panel.observations, panel.context)[0][:,-1,:]
        return float((output.argmax(-1) == panel.labels).float().mean())


def _weights_digest(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_cell(seed: int, *, steps: int = 400, batch: int = 128,
               shuffled_labels: bool = False,
               save_dir: Path | None = None) -> tuple[EvaluationBeliefLooper, dict[str, Any]]:
    if type(steps) is not int or steps < 0 or type(batch) is not int or batch < 1:
        raise ValueError("steps must be nonnegative and batch positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = EvaluationBeliefLooper()
    start = _weights_digest(model)
    panel = make_panel(16_384, TRAIN_VISITS, seed+100, shuffled_labels=shuffled_labels)
    generator = torch.Generator().manual_seed(seed+1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    losses = []
    for update in range(steps):
        index = torch.randint(len(panel), (batch,), generator=generator)
        logits, _ = model(panel.observations[index],panel.context[index])
        loss = (F.cross_entropy(logits[:,-1,:],panel.labels[index])
                + Q(1,5).numerator / Q(1,5).denominator
                * F.cross_entropy(logits.reshape(-1,2), panel.prefix_labels[index].reshape(-1)))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if (update+1) in (1, 100, 200, 300, 400) or update+1 == steps:
            losses.append((update+1,float(loss.detach())))
    model.eval()
    val4=make_panel(4096,4,901)
    val6=make_panel(4096,6,903)
    record = {
        "seed": seed, "updates": steps, "batch": batch,
        "shuffled_labels": shuffled_labels,
        "initial_state_digest": start,
        "final_state_digest": _weights_digest(model),
        "heldout_4_accuracy": _accuracy(model,val4),
        "heldout_6_accuracy": _accuracy(model,val6),
        "loss_checkpoints": losses,
        "supervision": "terminal CE + 0.2 mean per-visit CE",
        "checkpoint_selection": "final update only",
    }
    if save_dir is not None:
        save_dir.mkdir(parents=True,exist_ok=True)
        name=f"gg01_seed{seed}_{'shuffled' if shuffled_labels else 'trained'}"
        torch.save({
            "model":model.state_dict(), "record":record,
            "protocol":"RMP-GG-01","model_architecture":"native ExplicitDecoderBlock",
        },save_dir/f"{name}.pt")
        record["artifact_name"]=f"{name}.pt"
        record["artifact_sha256"]=hashlib.sha256((save_dir/f"{name}.pt").read_bytes()).hexdigest()
    return model,record


@dataclass(frozen=True)
class BeliefProbe:
    direction: Tensor
    weight: Tensor
    intercept: float
    calibration_r2: float
    validation_r2: float
    depth6_r2: float


def _prefix_log_odds(panel: CuePanel, cutoff: int) -> Tensor:
    return torch.tensor([
        math.log(float(p / (1-p)))+
        (2*sum(int(b) for b in panel.observations[i,:cutoff])-cutoff)*math.log(3)
        for i,(p,ratio) in enumerate(panel.parameters)
    ], dtype=torch.float64)


def _r2(actual: Tensor, estimate: Tensor) -> float:
    actual,estimate=actual.double(),estimate.double()
    denom=torch.sum((actual-actual.mean())**2)
    return float(1-torch.sum((actual-estimate)**2)/denom) if denom>0 else float("nan")


def fit_belief_probe(model: EvaluationBeliefLooper, cutoff: int = PATCH_VISIT) -> BeliefProbe:
    if type(cutoff) is not int or cutoff<1 or cutoff>TRAIN_VISITS:
        raise ValueError("invalid probe cutoff")
    calibration=make_panel(1024,4,5001)
    validation=make_panel(1024,4,5002)
    longer=make_panel(1024,6,5003)
    with torch.no_grad():
        features=model(calibration.observations,calibration.context)[1][:,cutoff-1,:].double()
    target=_prefix_log_odds(calibration,cutoff)
    mu_x=features.mean(0)
    mu_y=target.mean()
    centered=features-mu_x
    weight=torch.linalg.solve(
        centered.T@centered+1e-3*torch.eye(model.width,dtype=torch.float64),
        centered.T@(target-mu_y),
    )
    if float(weight.norm()) <= 1e-12:
        raise ValueError("zero-norm learned belief probe")
    intercept=float(mu_y-weight@mu_x)
    direction=(weight / weight.norm()).float()
    def evaluate(panel: CuePanel):
        with torch.no_grad():
            x=model(panel.observations,panel.context)[1][:,cutoff-1,:].double()
        return _r2(_prefix_log_odds(panel,cutoff),x@weight+intercept)
    return BeliefProbe(
        direction.detach(),weight.float().detach(),intercept,
        _r2(target,features@weight+intercept),
        evaluate(validation),evaluate(longer),
    )


@dataclass(frozen=True)
class SwapPanel:
    receiver: Tensor
    donor: Tensor
    hybrid: Tensor
    context: Tensor
    labels: Tensor
    attempted: int


def counterfactual_pairs(n: int = 256, visits: int = 4, cutoff: int = PATCH_VISIT,
                         seed: int = 21111) -> SwapPanel:
    if type(n) is not int or n < 1 or not 1 <= cutoff < visits <= 16:
        raise ValueError("invalid pair panel dimensions")
    rng=random.Random(seed)
    receiver=[]; donor=[]; hybrid=[]; contexts=[]; labels=[]
    attempts=0
    while len(receiver)<n and attempts<n*200:
        attempts+=1
        p,ratio=rng.choice(PRIORS),rng.choice(RATIOS)
        r=tuple(rng.randrange(2) for _ in range(visits))
        d=tuple(rng.randrange(2) for _ in range(visits))
        h=d[:cutoff]+r[cutoff:]
        y_r=exact_action(p,ratio,r)
        y_d=exact_action(p,ratio,d)
        y_h=exact_action(p,ratio,h)
        if y_h==y_r or y_h==y_d:
            continue
        receiver.append(r);donor.append(d);hybrid.append(h)
        contexts.append((math.log(float(p/(1-p))),math.log(float(ratio))))
        labels.append(y_h)
    if len(receiver)<n:
        raise ValueError("insufficient oracle-discriminating pairs")
    return SwapPanel(
        torch.tensor(receiver,dtype=torch.long),
        torch.tensor(donor,dtype=torch.long),
        torch.tensor(hybrid,dtype=torch.long),
        torch.tensor(contexts,dtype=torch.float32),
        torch.tensor(labels,dtype=torch.long),
        attempts,
    )


def _hybrid_margin(logits: Tensor, target: Tensor) -> Tensor:
    logits=logits[:,-1,:]
    chosen=logits.gather(1,target[:,None]).flatten()
    other=logits.gather(1,(1-target)[:,None]).flatten()
    return chosen-other


def causal_intervention(model: EvaluationBeliefLooper, probe: BeliefProbe,
                        *, visits: int = 4, cutoff: int = PATCH_VISIT,
                        pairs: int = 256, random_controls: int = 8) -> dict[str, Any]:
    if random_controls<1:
        raise ValueError("at least one random-direction control required")
    panel=counterfactual_pairs(pairs,visits,cutoff)
    with torch.no_grad():
        recv, recv_states=model(panel.receiver,panel.context)
        donor, donor_states=model(panel.donor,panel.context)
        direct_hybrid,_=model(panel.hybrid,panel.context)
        receiver_correct=(recv[:,-1,:].argmax(-1)==(1-panel.labels))
        hybrid_correct=(direct_hybrid[:,-1,:].argmax(-1)==panel.labels)
        mask=receiver_correct&hybrid_correct
        n=int(mask.sum())
        if n==0:
            return {"attempted":panel.attempted,"manifest":pairs,"eligible":0,
                    "status":"NO_ELIGIBLE_PAIRS"}
        receiver_state=recv_states[mask,cutoff-1,:]
        donor_state=donor_states[mask,cutoff-1,:]
        ctx=panel.context[mask]; bits=panel.receiver[mask]
        label=panel.labels[mask]
        w=probe.direction.to(receiver_state.device)
        displacement=donor_state-receiver_state
        component=(displacement@w)[:,None]*w[None,:]
        lifted=receiver_state+component
        full=model(bits,ctx,patch_visit=cutoff,patch_state=donor_state)[0]
        rank1=model(bits,ctx,patch_visit=cutoff,patch_state=lifted)[0]
        sham=model(bits,ctx,patch_visit=cutoff,patch_state=receiver_state)[0]
        target=direct_hybrid[mask]
        if not torch.allclose(full,target,rtol=3e-5,atol=3e-5):
            raise AssertionError("full hidden-state patch does not equal explicit hybrid replay")
        if not torch.allclose(sham,recv[mask],rtol=3e-5,atol=3e-5):
            raise AssertionError("same-state sham changed the deterministic continuation")
        clean=recv[mask]
        def metric(logits: Tensor):
            return {
                "hit":float((logits[:,-1,:].argmax(-1)==label).float().mean()),
                "mean_margin":float(_hybrid_margin(logits,label).mean()),
            }
        base=metric(clean); full_report=metric(full); rank_report=metric(rank1)
        donor_copy=metric(donor[mask])
        generator=torch.Generator(device=receiver_state.device).manual_seed(66713)
        controls=[]
        lengths=component.norm(dim=1,keepdim=True)
        for _ in range(random_controls):
            noise=torch.randn(receiver_state.shape,generator=generator,device=receiver_state.device)
            noise=noise / noise.norm(dim=1,keepdim=True).clamp_min(1e-12)
            noise=noise * lengths
            random_result=model(bits,ctx,patch_visit=cutoff,
                                patch_state=receiver_state+noise)[0]
            controls.append(metric(random_result))
        denominator=full_report["mean_margin"]-base["mean_margin"]
        return {
            "status":"EVALUATED", "attempted":panel.attempted,
            "manifest":pairs, "eligible":n,
            "eligible_fraction":n/pairs,
            "receiver":base, "full_patch":full_report,
            "rank_one":rank_report, "donor_copy":donor_copy,
            "random_controls":controls,
            "rank_one_margin_lift":rank_report["mean_margin"]-base["mean_margin"],
            "full_margin_lift":denominator,
            "rank_one_fraction_of_full_lift":(
                (rank_report["mean_margin"]-base["mean_margin"])/denominator
                if abs(denominator)>1e-9 else None
            ),
            "sham_equal_to_receiver":True,
            "full_equal_to_exact_hybrid":True,
            "donor_copy_label_disagrees_by_oracle":True,
        }


def run_pilot(*, steps: int = 400, seeds: tuple[int,...] = (11,17),
              save_dir: Path | None = None,
              include_shuffled_control: bool = True) -> dict[str, Any]:
    if not seeds or any(type(s) is not int for s in seeds):
        raise ValueError("seeds must be nonempty integer tuple")
    record={"protocol":"RMP-GG-01","training_steps":steps,
            "kind":"supervised neural causal positive control",
            "evalmath_basis":"RHT-20–25 exact finite type-cue decision",
            "trained_model_causal_claim":False,
            "results":[]}
    for seed in seeds:
        model,train=train_cell(seed,steps=steps,save_dir=save_dir)
        probe=fit_belief_probe(model)
        patch=causal_intervention(model,probe)
        passed=(
            train["heldout_6_accuracy"]>=0.90
            and probe.validation_r2>=0.50
            and patch.get("eligible",0)>=32
        )
        supported=(
            passed and patch["rank_one"]["hit"]-patch["receiver"]["hit"]>=0.10
            and all(patch["rank_one"]["hit"]-null["hit"]>=0.05
                    for null in patch["random_controls"])
        )
        record["results"].append({
            "training":train,
            "probe":{"calibration_r2":probe.calibration_r2,
                     "validation_r2":probe.validation_r2,
                     "depth6_r2":probe.depth6_r2},
            "causal_intervention":patch,
            "eligible_for_causal_interpretation":passed,
            "rank_one_causally_supported":supported,
        })
    if include_shuffled_control:
        null_model,null_train=train_cell(seeds[0]+9000,steps=steps,
                                          shuffled_labels=True,save_dir=save_dir)
        record["shuffled_label_control"]=null_train
    record["all_seeds_eligible"]=all(r["eligible_for_causal_interpretation"]
                                      for r in record["results"])
    record["all_seeds_rank_one_supported"]=all(r["rank_one_causally_supported"]
                                            for r in record["results"])
    return record


def save_receipt(receipt: dict[str,Any], path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n",
                    encoding="utf-8")
