"""Public, bounded LRM-REV-02 sources and fixed recurrent input encoding.

The fixture namespace is available for source tests before the prospective
freeze. Training and development namespaces belong to the frozen runner.
Every random role has its own hash-derived generator. No confirmation phase
is implemented here.
"""
from __future__ import annotations

from fractions import Fraction as Q
from functools import lru_cache
from hashlib import sha256
import json
from math import comb
import random

from gym import Episode, MATCH, PRIORS


N = 12
J = 6
ROOT_COUNT = 4
TRAIN_EPISODES = 2048
DEVELOPMENT_EPISODES = 512
FIXTURE_EPISODES = 64
CONTROL_COUNT = 4
INPUT_DIM = 17
DOMAIN = "lrm-rev02/public-cpu-pilot/v1"
PHASE_LIMITS = {
    "fixture": FIXTURE_EPISODES,
    "train": TRAIN_EPISODES,
    "development": DEVELOPMENT_EPISODES,
}


def _root(root: int) -> None:
    if type(root) is not int or not 0 <= root < ROOT_COUNT:
        raise ValueError("model root must be an integer in 0..3")


def _digest(role: str, *parts: object) -> bytes:
    # JSON framing prevents ambiguous concatenation of role and identifier.
    text = json.dumps([DOMAIN, role, *parts], separators=(",", ":"),
                      ensure_ascii=True)
    return sha256(text.encode("ascii")).digest()


def _rng(role: str, *parts: object) -> random.Random:
    return random.Random(int.from_bytes(_digest(role, *parts), "big"))


def model_seed(root: int) -> int:
    """Weight-initialization seed from a role disjoint from episode sources."""
    _root(root)
    return int.from_bytes(_digest("model-initialization", root)[:8], "big") & ((1 << 63) - 1)


def make_pilot_episode(root: int, index: int, phase: str) -> Episode:
    """Generate one registered public source after validating all bounds."""
    _root(root)
    if type(phase) is not str or phase not in PHASE_LIMITS:
        raise ValueError("phase must be fixture, train, or development")
    if type(index) is not int or not 0 <= index < PHASE_LIMITS[phase]:
        raise ValueError("episode index outside the registered phase")
    source_id = _digest("episode-source", root, phase, index).hex()

    # Choose the schedule through a separate role before generating outcomes.
    slots = tuple(sorted(_rng("episode-schedule", source_id).sample(range(N), J)))
    draw = _rng("episode-regime", source_id).randrange(20)
    regime = 0 if draw < 9 else (1 if draw < 16 else 2)
    threshold = int(MATCH[regime] * 20)
    old_rng = _rng("episode-old-bits", source_id)
    replacement_rng = _rng("episode-replacement-bits", source_id)
    old = tuple(old_rng.randrange(2) if i in slots else
                int(old_rng.randrange(20) < threshold) for i in range(N))
    replacements = tuple(int(replacement_rng.randrange(20) < threshold)
                         for _ in slots)
    return Episode(source_id, regime, old, slots, replacements)


def _binomial(n: int, k: int, p: Q) -> Q:
    if not 0 <= k <= n:
        return Q(0)
    return Q(comb(n, k)) * p ** k * (1 - p) ** (n - k)


@lru_cache(maxsize=None)
def _initial_posterior(old_count: int, n: int, j: int) -> tuple[Q, ...]:
    weights = tuple(
        prior * sum((_binomial(n - j, u, t)
                     * _binomial(j, old_count - u, Q(1, 2))
                     for u in range(n - j + 1)), Q(0))
        for prior, t in zip(PRIORS, MATCH)
    )
    total = sum(weights, Q(0))
    return tuple(weight / total for weight in weights)


def initial_posterior(old_count: int, n: int = N, j: int = J) -> tuple[Q, ...]:
    """Exact Pr(K | named old stream, n, j) with uniformly hidden S.

    Exchangeability makes the old total sufficient at this initial stage.
    The boundary domains j=0 and j=n support independent finite audits.
    Validation precedes the private cache, including bool/int distinctions.
    """
    if not all(type(value) is int for value in (old_count, n, j)):
        raise ValueError("integer posterior metadata required")
    if not (1 <= n <= N and 0 <= j <= n and 0 <= old_count <= n):
        raise ValueError("invalid initial posterior domain")
    return _initial_posterior(old_count, n, j)


def _source(episode: Episode) -> str:
    if not isinstance(episode, Episode) or type(episode.source_id) is not str or not episode.source_id:
        raise ValueError("episode with a nonempty source identifier required")
    return episode.source_id


def _slots(slots: object) -> tuple[int, ...]:
    if not isinstance(slots, (tuple, list)):
        raise ValueError("six named positions required")
    if (len(slots) != J or any(type(i) is not int or not 0 <= i < N for i in slots)
            or len(set(slots)) != J):
        raise ValueError("six distinct positions in 0..11 required")
    return tuple(slots)


def control_slots(episode: Episode, control_index: int) -> tuple[int, ...]:
    """Uniform six-slot active reminder control, chosen without outcome access.

    Coincidences with the correction schedule are retained. Each of the four
    controls owns a separate role-index seed.
    """
    source_id = _source(episode)
    if type(control_index) is not int or not 0 <= control_index < CONTROL_COUNT:
        raise ValueError("control index must be an integer in 0..3")
    return tuple(sorted(_rng("control-reminder-subset", source_id, control_index)
                        .sample(range(N), J)))


def training_slots(episode: Episode) -> tuple[int, ...]:
    """Choose corrected slots or an independent uniform subset with equal odds."""
    source_id = _source(episode)
    target = tuple(sorted(_slots(episode.correction_slots)))
    if _rng("training-reminder-choice", source_id).randrange(2) == 0:
        return target
    return tuple(sorted(_rng("training-reminder-subset", source_id).sample(range(N), J)))


def _check_observations(episode: Episode) -> None:
    if not isinstance(episode, Episode):
        raise ValueError("Episode required")
    _slots(episode.correction_slots)
    if (len(episode.old) != N or len(episode.replacements) != J
            or any(type(bit) is not int or bit not in (0, 1)
                   for bit in (*episode.old, *episode.replacements))):
        raise ValueError("twelve old bits and six replacement bits required")


def _token(event_type: int, position: int, bit: int) -> list[float]:
    vector = [0.0] * INPUT_DIM
    vector[event_type] = 1.0
    vector[3 + position] = 1.0
    vector[15 + bit] = 1.0
    return vector


def encode_prefix(episode: Episode) -> list[list[float]]:
    """Encode twelve OLD events, then six REPLACE events; no private fields.

    Feature blocks are event type (OLD=0, REPLACE=1, REMINDER=2), twelve
    named positions, and two bit values. Every token has squared norm three.
    The initial twelve tokens depend only on the named old observations.
    """
    _check_observations(episode)
    stream = [_token(0, i, bit) for i, bit in enumerate(episode.old)]
    stream.extend(_token(1, i, bit) for i, bit in
                  zip(episode.correction_slots, episode.replacements))
    return stream


def encode_stream(episode: Episode, reminder_slots: tuple[int, ...]) -> list[list[float]]:
    """Append six truthful OLD-REMINDER events in fixed ascending order."""
    slots = tuple(sorted(_slots(reminder_slots)))
    stream = encode_prefix(episode)
    stream.extend(_token(2, i, episode.old[i]) for i in slots)
    return stream
