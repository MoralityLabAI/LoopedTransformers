"""Bind paired-source closure and current-head telemetry to live signed edits."""
from hashlib import sha256

from experiments.nan_gated46.gated import margin


def fingerprint(packet, row):
    import torch

    tensors = [packet[k][phase, row] for k in ("inputs", "residual", "native_writes") for phase in (1, 2)]
    tensors.extend(packet["logits"][phase, row] for phase in (0, 1))
    result = []
    for value in tensors:
        value = value.detach().cpu().contiguous()
        result.append((str(value.dtype), tuple(value.shape), sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()))
    return tuple(result)


class ObservedCapture:
    """One duplicate baseline, four plans and four individually verified continuations."""

    def __init__(self, capture, expected, labels):
        self.inner = capture
        self.cache, self.head = capture.cache, capture.head
        self.expected = expected
        self.labels = labels
        self.plans, self.reached = {}, []
        self.vocab = None
        self.residual_l2 = None

    def planned_head(self, baseline, edit):
        import torch

        result = self.inner.planned_head(baseline, edit)
        origin = baseline["logits"][0, 1]
        delta = result[1].to(dtype=torch.float64) - origin.to(dtype=torch.float64)
        self.plans[id(result)] = (result, {
            "rms_logit_drift": float(delta.square().mean().sqrt()),
            "signed_bare_margin_delta": margin(result[1], self.labels) - margin(origin, self.labels)})
        return result

    def capture(self, prompts, edit=None, **options):
        packet = self.inner.capture(prompts, edit, **options)
        if edit is None:
            if self.vocab is not None or fingerprint(packet, 0) != self.expected or fingerprint(packet, 1) != self.expected:
                raise RuntimeError("Duplicate baseline differs from the independently qualified pair")
            self.vocab = packet["logits"].shape[-1]
            residual = packet["residual"][1, 1].double()
            self.residual_l2 = float(residual.norm())
        else:
            key = id(options["planned"])
            if key not in self.plans:
                raise RuntimeError("A continuation lacks its current-head telemetry")
            _, profile = self.plans.pop(key)
            future = packet["logits"][1, 1]
            self.reached.append((int(future.argmax()), margin(future, self.labels), profile))
        return packet

    def bind(self, result):
        if result["status"] != "closed":
            if self.reached or result["edited_pass2_runs"] != 0:
                raise ValueError("A refused candidate emitted a future observation")
            return result
        if self.plans or len(self.reached) != 4 or len(result["outcomes"]) != 4:
            raise ValueError("The signed continuations and full heads do not close")
        profile = {}
        for (key, row), (winner, bare, values) in zip(result["outcomes"].items(), self.reached, strict=True):
            if (winner, bare) != (row["winner"], row["bare_margin"]):
                raise ValueError("Signed future differs from its captured current-head record")
            profile[key] = values
        return result | {"phase1_profile": profile, "vocabulary_size": self.vocab,
                         "phase1_residual_l2": self.residual_l2}
