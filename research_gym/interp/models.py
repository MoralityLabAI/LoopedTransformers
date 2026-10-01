"""Model arms of the Recurrence Monitorability Program."""

from __future__ import annotations

from torch import nn

from research_gym.neural.looped_decoder import LoopedDecoderLM

from .tasks import VOCABULARY_SIZE


ARMS = {"LOOP-T": "fully_tied", "FF-U": "untied"}


def build_model(
    arm: str,
    *,
    hidden_size: int = 128,
    num_heads: int = 4,
    visits: int = 8,
    vocab_size: int = VOCABULARY_SIZE,
) -> LoopedDecoderLM:
    """Rung M0: one tied block (LOOP-T) or one block per visit (FF-U), same width and visits."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    tied = arm == "LOOP-T"
    model = LoopedDecoderLM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        physical_modules=1 if tied else visits,
        expanded_visits=visits,
        tying=ARMS[arm],
        parameter_visits=frozenset(range(visits)),
        retained_state_edges=frozenset(range(visits - 1)),
        supervision_points=(visits - 1,),
        alpha=1.0,
        beta=0.25,
        normalization="pre",
        carry_policy="reset",
        module_word=(0,) * visits if tied else tuple(range(visits)),
        use_sinusoidal_positions=True,
        attention="explicit",
    )
    # The unembedding is tied; unit-variance rows would start the loss near sqrt(hidden) nats per logit.
    nn.init.normal_(model.embedding.weight, std=hidden_size**-0.5)
    return model
