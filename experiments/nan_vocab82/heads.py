"""Reduce the native B2 vocabulary heads to exact, answer-blind class maxima."""

from hashlib import sha256
from math import isfinite

from experiments.nan_comp14.parser import label
from experiments.nan_relabel68.pipeline import audited_score


LABELS = "ABCDEFGHIJKLMNOP"
INVALID = len(LABELS)


def partition(tokenizer, vocab, check):
    """Map every vocabulary ID to its first-token grammar class."""
    classes = bytearray()
    for start in range(0, vocab, 4096):
        check()
        tokens = tokenizer.decode_batch([[i] for i in range(start, min(start + 4096, vocab))],
                                        skip_special_tokens=False)
        classes.extend(INVALID if name is None else LABELS.index(name) for name in map(label, tokens))
    if len(classes) != vocab or set(classes) != set(range(INVALID + 1)):
        raise ValueError("The pinned tokenizer does not cover every grammar class")
    return bytes(classes)


def partition_sha(classes):
    return sha256(classes).hexdigest()


class Reducer:
    def __init__(self, classes):
        import torch

        if set(classes) != set(range(INVALID + 1)):
            raise ValueError("Expected all grammar classes")
        self.ids = torch.arange(len(classes), dtype=torch.int64).expand(4, -1)
        self.classes = torch.tensor(list(classes), dtype=torch.int64).expand(4, -1)

    def __call__(self, logits):
        """Keep the best token and logit in each class, with lowest-ID ties."""
        import torch

        vocab = self.ids.shape[1]
        if tuple(logits.shape) != (2, 2, vocab) or logits.device.type != "cpu":
            raise ValueError("Expected both physical CPU heads from a single native B2 prefill")
        scores = logits.reshape(4, vocab).to(torch.float64)
        if not scores.isfinite().all():
            raise ValueError("Cannot reduce a nonfinite vocabulary row")
        values = scores.new_full((4, INVALID + 1), -torch.inf)
        values.scatter_reduce_(1, self.classes, scores, reduce="amax")
        is_best = scores == values.gather(1, self.classes)
        leaders = self.ids.new_full((4, INVALID + 1), vocab)
        leaders.scatter_reduce_(1, self.classes, torch.where(is_best, self.ids, vocab), reduce="amin")
        return [[list(zip(tokens, maxima, strict=True)) for tokens, maxima in row]
                for row in zip_by_row(leaders.tolist(), values.tolist())]


def zip_by_row(leaders, values):
    return [[(leaders[i], values[i]), (leaders[i + 2], values[i + 2])] for i in range(2)]


def verify_head(maxima, native, classes, bare):
    """Reconcile the class partition, global winner and measured bare rows."""
    winner = audited_score(native, bare, len(classes))
    if len(maxima) != INVALID + 1:
        raise ValueError("Class maxima omit part of the grammar")
    for k, (token, score) in enumerate(maxima):
        if (type(token) is not int or not 0 <= token < len(classes) or classes[token] != k
                or type(score) not in (int, float) or not isfinite(score)):
            raise ValueError("A class leader has an invalid token or value")
        if token != winner and score > native["second_logit"]:
            raise ValueError("A class leader exceeds the native runner-up")
    top = min(maxima, key=lambda t: (-t[1], t[0]))
    if tuple(top) != (winner, native["peak"]):
        raise ValueError("Class leaders disagree with the complete native winner")
    for k, name in enumerate(LABELS):
        canonical = bare[name]
        leader, peak = maxima[k]
        value = native["labels"][name]
        if (classes[canonical] != k or value > peak or
                (value == peak and leader > canonical) or (leader == canonical and value != peak)):
            raise ValueError("A grammar leader disagrees with the canonical token")
    return classes[winner]


def margin(maxima, answer):
    """Reserve gold for the host and retain the identity of the strongest foil."""
    target = LABELS.index(answer)
    valid_foils = [pair for i, pair in enumerate(maxima[:-1]) if i != target]
    valid_token, valid_score = min(valid_foils, key=lambda t: (-t[1], t[0]))
    invalid_token, invalid_score = maxima[INVALID]
    gold_token, gold_score = maxima[target]
    foil, foil_score = min(((valid_token, valid_score), (invalid_token, invalid_score)),
                           key=lambda t: (-t[1], t[0]))
    return {"gold_token": gold_token, "gold_score": gold_score, "foil_token": foil,
            "foil_score": foil_score, "margin": gold_score - foil_score,
            "valid_margin": gold_score - valid_score, "invalid_margin": gold_score - invalid_score}
