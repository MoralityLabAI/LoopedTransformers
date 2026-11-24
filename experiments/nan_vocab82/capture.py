"""Use the frozen WRITER77 execution functions with a read-only head observer."""

from experiments.nan_premlp64.capture import same_bf16
from experiments.nan_radial76.geometry import ARMS
from experiments.nan_writer77.pipeline import batch_baseline, root_trial

from .heads import verify_head


class Observer:
    def __init__(self, capture, reducer, *, trial):
        self.engine = capture
        self.reducer = reducer
        self.trial = trial
        self.rows = []

    def capture(self, prompts, *, edit=None, rescue=None):
        result = self.engine.capture(prompts, edit=edit, rescue=rescue)
        if not self.trial:
            self.rows.extend(self.reducer(result["logits"]))
        elif edit is not None and not same_bf16(*edit):
            self.rows.append((rescue is not None, self.reducer(result["logits"])[1]))
        return result


def observed_baseline(capture, payload, arms, check, reduce, classes):
    observer = Observer(capture, reduce, trial=False)
    baseline, focus = batch_baseline(observer, payload, arms, check)
    bare = payload["tokens"]["bare_label_token_ids"]
    for row, maxima in zip(baseline, observer.rows, strict=True):
        for reading, native in zip(maxima, row["phases"], strict=True):
            verify_head(reading, native, classes, bare)
        row["class_maxima"] = maxima
    return baseline, focus


def observed_trial(capture, root, original, tokens, bare, fit, masks, np, check, reduce, classes):
    observer = Observer(capture, reduce, trial=True)
    result = root_trial(observer, root, original, tokens, bare, fit, masks, np, check)
    modes = (ARMS[0], "active_k_rescue", *ARMS[1:])
    if result["status"] != "closed":
        if observer.rows:
            raise ValueError("An unclosed root emitted edited vocabulary heads")
        return result
    if len(result["worlds"]) != 2 or len(observer.rows) != 2 * len(modes):
        raise ValueError("The edit schedule and its vocabulary observations disagree")
    rows = iter(observer.rows)
    for world in result["worlds"]:
        maxima = {}
        for mode in modes:
            rescued, reading = next(rows)
            if rescued != (mode == "active_k_rescue"):
                raise ValueError("The observed K-rescue position is incorrect")
            for phase, native in zip(reading, world["arms"][mode], strict=True):
                verify_head(phase, native, classes, bare)
            maxima[mode] = reading
        if maxima["active"][0] != maxima["active_k_rescue"][0]:
            raise ValueError("The K-rescue changed the observed first-pass grammar maxima")
        world["class_maxima"] = maxima
    return result
