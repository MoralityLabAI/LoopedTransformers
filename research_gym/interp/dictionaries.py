"""E4 dictionaries: TopK sparse autoencoders and transcoders trained on states generated on the fly."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterator

import torch
from torch import Tensor, nn

from .analysis import AnalysisModel


def rms(x: Tensor) -> Tensor:
    return x.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()


@dataclass(frozen=True)
class DictionaryConfig:
    kind: str  # "shared", "visit", "tok", "trans"
    width: int
    k: int
    seed: int = 0
    site: int = -1  # the visit for "visit", the layer for a per-layer "trans"; -1 when pooled
    steps: int = 10_000
    batch_examples: int = 32
    learning_rate: float = 3e-4
    dead_after: int = 2_000
    aux_weight: float = 1 / 32

    @property
    def name(self) -> str:
        site = "" if self.site < 0 else f"_t{self.site}"
        return f"{self.kind}{site}_w{self.width}_k{self.k}_s{self.seed}"


class TopKDictionary(nn.Module):
    """TopK autoencoder (``skip=False``) or transcoder with a linear skip term (``skip=True``).

    Decoder rows are kept at unit norm. Inputs are RMS-normalized states; outputs are in the same units.
    """

    def __init__(self, dim: int, width: int, k: int, *, skip: bool = False, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        decoder = torch.randn(width, dim, generator=generator)
        decoder = decoder / decoder.norm(dim=1, keepdim=True)
        self.k = k
        self.pre_bias = nn.Parameter(torch.zeros(dim))
        self.encoder = nn.Parameter(decoder.T.clone())
        self.encoder_bias = nn.Parameter(torch.zeros(width))
        self.decoder = nn.Parameter(decoder)
        self.skip = nn.Parameter(torch.zeros(dim, dim)) if skip else None
        self.output_bias = nn.Parameter(torch.zeros(dim)) if skip else None

    def preactivations(self, x: Tensor) -> Tensor:
        return (x - self.pre_bias) @ self.encoder + self.encoder_bias

    def encode(self, x: Tensor) -> Tensor:
        pre = self.preactivations(x)
        values, indices = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, indices, torch.relu(values))

    def decode(self, codes: Tensor, x: Tensor | None = None) -> Tensor:
        out = codes @ self.decoder
        if self.skip is None:
            return out + self.pre_bias
        return out + x @ self.skip + self.output_bias

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        codes = self.encode(x)
        return self.decode(codes, x), codes

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.data /= self.decoder.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def reconstruct_state(self, state: Tensor) -> Tensor:
        """Autoencoder reconstruction of raw states, scaled back by each state's RMS norm."""

        scale = rms(state)
        return scale * self(state / scale)[0]

    def predict_update(self, state: Tensor) -> Tensor:
        """Transcoder prediction of a raw update from a raw state."""

        scale = rms(state)
        return scale * self(state / scale)[0]


@torch.no_grad()
def model_states(model: AnalysisModel, tokens: Tensor) -> Tensor:
    """All residual sites for a batch: (B, visits + 1, positions, H), site 0 entering the first visit."""

    output = model.run(tokens)
    entering = model.model.embed(tokens)
    return torch.stack([entering, *output.visit_outputs], dim=1).float()


def site_vectors(states: Tensor, config: DictionaryConfig, *, layers: int) -> tuple[Tensor, Tensor]:
    """Normalized (input, target) vectors for one dictionary kind from a batch of states."""

    if config.kind == "tok":
        x = states[:, 0]
        y = x
    elif config.kind == "shared":
        x = states[:, 1:]
        y = x
    elif config.kind == "visit":
        x = states[:, config.site]
        y = x
    elif config.kind == "trans":
        visits = range(layers) if config.site < 0 else [config.site]
        x = states[:, list(visits)]
        y = states[:, [visit + 1 for visit in visits]] - x
    else:
        raise ValueError(config.kind)
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1, y.shape[-1])
    scale = rms(x)
    return x / scale, y / scale


def batch_source(
    model: AnalysisModel, tokens: Tensor, config: DictionaryConfig, *, generator: torch.Generator
) -> Iterator[tuple[Tensor, Tensor]]:
    device = next(model.model.parameters()).device
    layers = model.config.visits
    while True:
        index = torch.randint(0, tokens.shape[0], (config.batch_examples,), generator=generator)
        states = model_states(model, tokens[index].to(device))
        yield site_vectors(states, config, layers=layers)


def train_dictionary(
    source: Iterator[tuple[Tensor, Tensor]],
    config: DictionaryConfig,
    *,
    dim: int,
    device: torch.device,
    log: Callable[[dict], None] | None = None,
) -> tuple[TopKDictionary, dict]:
    """Train one dictionary; returns it with a summary of its last 500 steps."""

    torch.manual_seed(config.seed)
    dictionary = TopKDictionary(dim, config.width, config.k, skip=config.kind == "trans", seed=config.seed).to(device)
    first_x, first_y = next(source)
    with torch.no_grad():
        # Pre-encoder bias starts at the data mean, as is usual for these autoencoders.
        dictionary.pre_bias.copy_(first_x.mean(dim=0))
        if dictionary.output_bias is not None:
            dictionary.output_bias.copy_(first_y.mean(dim=0))
    optimizer = torch.optim.Adam(dictionary.parameters(), lr=config.learning_rate)
    last_fired = torch.zeros(config.width, dtype=torch.long, device=device)
    k_aux = min(256, config.width // 2)
    history: list[dict] = []
    batch = (first_x, first_y)
    for step in range(config.steps):
        if step:
            batch = next(source)
        x, y = batch
        decay = min(1.0, (config.steps - step) / (0.2 * config.steps))
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * decay
        pre = dictionary.preactivations(x)
        values, indices = pre.topk(config.k, dim=-1)
        codes = torch.zeros_like(pre).scatter(-1, indices, torch.relu(values))
        prediction = dictionary.decode(codes, x)
        error = y - prediction
        loss = error.pow(2).sum(dim=-1).mean()
        fired = (codes > 0).any(dim=0)
        last_fired = torch.where(fired, torch.full_like(last_fired, step), last_fired)
        dead = (step - last_fired) >= config.dead_after
        aux = torch.zeros((), device=device)
        if dead.any():
            dead_pre = pre.masked_fill(~dead, float("-inf"))
            count = min(k_aux, int(dead.sum()))
            aux_values, aux_indices = dead_pre.topk(count, dim=-1)
            aux_codes = torch.zeros_like(pre).scatter(-1, aux_indices, torch.relu(aux_values))
            aux = (error.detach() - aux_codes @ dictionary.decoder).pow(2).sum(dim=-1).mean()
        optimizer.zero_grad(set_to_none=True)
        (loss + config.aux_weight * aux).backward()
        optimizer.step()
        dictionary.normalize_decoder()
        if step >= config.steps - 500:
            variance = (y - y.mean(dim=0)).pow(2).sum(dim=-1).mean()
            history.append({"mse": float(loss.detach()), "fvu": float(loss.detach() / variance.clamp_min(1e-12)),
                            "dead": int(dead.sum())})
        if log is not None and step % 1000 == 0:
            log({"step": step, "loss": float(loss), "aux": float(aux), "dead": int(dead.sum())})
    summary = {
        "config": asdict(config),
        "train_mse": sum(item["mse"] for item in history) / len(history),
        "train_fvu": sum(item["fvu"] for item in history) / len(history),
        "dead_latents": history[-1]["dead"],
    }
    return dictionary.eval(), summary


def save_dictionary(dictionary: TopKDictionary, summary: dict, path) -> str:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": dictionary.state_dict(), "summary": summary,
                "shape": {"dim": dictionary.decoder.shape[1], "width": dictionary.decoder.shape[0],
                          "k": dictionary.k, "skip": dictionary.skip is not None}}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dictionary(path, *, device: str | torch.device = "cpu") -> tuple[TopKDictionary, dict]:
    saved = torch.load(path, map_location=device, weights_only=False)
    shape = saved["shape"]
    dictionary = TopKDictionary(shape["dim"], shape["width"], shape["k"], skip=shape["skip"]).to(device)
    dictionary.load_state_dict(saved["state"])
    return dictionary.eval(), saved["summary"]


def fraction_of_variance_unexplained(dictionary: TopKDictionary, x: Tensor, y: Tensor) -> float:
    with torch.no_grad():
        error = (y - dictionary(x)[0]).pow(2).sum(dim=-1).mean()
        variance = (y - y.mean(dim=0)).pow(2).sum(dim=-1).mean()
    return float(error / variance.clamp_min(1e-12))
