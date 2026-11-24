"""Compute paired layer-local readouts in the signed delayed ROUTE58 protocol."""
from math import fsum, isclose

from experiments.nan_premlp64.account import TERMS, account, row
from experiments.nan_premlp64.capture import same_bf16
from experiments.nan_readpath37.run import FixedHead
from experiments.nan_route58.capture import ObservedCapture
from experiments.nan_route58.host import ARMS, SIGNS, finite, score_world
from experiments.nan_sham56.native import continuation

from .capture import EARLY_SITES, FOCUS


KEYS = tuple(f"{arm}/{sign}" for arm in ARMS for sign in SIGNS)
LINEAR = TERMS + ("rms_scale", "head_rounding")
REPLAY_TOL = .125


class PairHead(FixedHead):
    """Use the existing CPU head arithmetic with only the registered two rows."""

    def __init__(self, engine, tokens, np):
        if len(tokens) != 2 or tokens[0] == tokens[1]:
            raise ValueError("Delayed attribution requires two distinct answer tokens")
        norm = engine.cache.loop.base.norm
        self.gamma = norm.weight.detach().float().cpu().numpy()
        self.eps = np.float32(norm.variance_epsilon)
        self.np = np
        self.cache = {token: engine.head.weight[token].detach().float().cpu().numpy() for token in tokens}

    def row(self, token):
        return self.cache[token]


def observe_change(reference, edited, layers, tokens, head, np):
    for layer in layers:
        before, after = reference["window"][layer], edited["window"][layer]
        for site in EARLY_SITES:
            if (not same_bf16(before[site][2, 0], after[site][2, 0])
                    or any(not same_bf16(before[site][1, r], after[site][1, r]) for r in (0, 1))):
                raise ValueError("A companion or pre-patch operand changed inside the delayed window")

    result = {}
    for layer in layers:
        before, after = reference["window"][layer], edited["window"][layer]
        result[str(layer)] = account(row(before, 2, 1), row(after, 2, 1), head, tokens, np)
    a, b = reference["logits"][1, 1], edited["logits"][1, 1]
    leaf = str(layers[-1])
    states = np.stack([r["residual"][2, 1].float().numpy()
                       for r in (reference["window"][layers[-1]], edited["window"][layers[-1]])])
    replay = head.logits(states, tokens)
    actual = np.stack([v[list(tokens)].float().numpy() for v in (a, b)])
    error = float(np.max(abs(replay - actual)))
    if error > REPLAY_TOL:
        raise ValueError("Final-layer selected replay diverged from the actual native head")
    result[leaf]["actual_selected_head_swing"] = (float(actual[1, 0] - actual[1, 1]) -
                                                  float(actual[0, 0] - actual[0, 1]))
    result[leaf]["max_selected_logit_error"] = error
    return result


class WindowTelemetry(ObservedCapture):
    def __init__(self, capture, expected, labels, np):
        super().__init__(capture, expected, labels)
        self.pair_head = PairHead(capture, labels, np)
        self.np = np
        self.window_baseline = None
        self.window_arms = []

    def capture(self, prompts, edit=None, **options):
        packet = super().capture(prompts, edit, **options)
        if edit is None:
            for sites in packet["window"].values():
                if any(not same_bf16(sites[site][phase, 0], sites[site][phase, 1])
                       for site in EARLY_SITES for phase in (1, 2)):
                    raise ValueError("Duplicate baseline rows differ inside the carry window")
            self.window_baseline = packet
        else:
            self.window_arms.append(observe_change(self.window_baseline, packet, self.inner.layers,
                                                  self.labels, self.pair_head, self.np))
        return packet

    def bind(self, result):
        record = super().bind(result)
        if record["status"] != "closed":
            if self.window_arms:
                raise ValueError("A refused arm emitted a delayed window")
            return record
        if tuple(record["outcomes"]) != KEYS or len(self.window_arms) != len(KEYS):
            raise ValueError("A delayed window lacks its registered signed ordering")
        return record | {"delayed_window": dict(zip(KEYS, self.window_arms, strict=True))}


def run_window_continuation(engine, expected, prompt, labels, tag, bank, policy,
                            cfg, round_cfg, dose, guard, masks, np):
    observed = WindowTelemetry(engine, expected, labels, np)
    return observed.bind(continuation(observed, prompt, labels, tag, bank, policy,
                                      cfg, round_cfg, dose, guard, masks, np))


def score_profile(record, winner, donor, round_cfg, dose, layers=FOCUS):
    """Independently gate and reconcile site-level signed dose-normalized effects."""
    original = score_world(record, winner, donor, round_cfg, dose)
    if original is None:
        if "delayed_window" in record:
            raise ValueError("Refused row carries an edited-layer result")
        return None
    arms = record["cases"]["arm_dose"]
    profile = record["delayed_window"]
    if set(profile) != set(KEYS):
        raise ValueError("Delayed window is missing a signed arm")
    stages = {str(layer) for layer in layers}
    summaries = {}
    base = record["baseline"]["bare_margin"]
    for key, values in profile.items():
        if set(values) != stages:
            raise ValueError("Delayed window does not contain the registered layers")
        for layer, cell in values.items():
            head = cell["native_swing"]
            finite_terms, linear = cell["finite"], cell["continuous"]
            if (set(finite_terms) != set(TERMS) or set(linear) != set(LINEAR)
                    or not finite(head, *finite_terms.values(), *linear.values())
                    or not isclose(fsum(finite_terms.values()), head, abs_tol=1e-9, rel_tol=0)
                    or not isclose(fsum(linear.values()), head, abs_tol=1e-9, rel_tol=0)):
                raise ValueError("Delayed site terms do not close against the rounded head")
            if layer == str(layers[-1]):
                native = record["outcomes"][key]["bare_margin"] - base
                error = cell["max_selected_logit_error"]
                if (not finite(error, cell["actual_selected_head_swing"])
                        or not 0 <= error <= REPLAY_TOL
                        or native != cell["actual_selected_head_swing"]
                        or abs(native - head) > 4 * error + 1e-12):
                    raise ValueError("Final-layer readout does not bind the native route score")

    def signed(get):
        active, control = (
            arms[plus]["requested_l2"] / (arms[plus]["realized_l2"] + arms[minus]["realized_l2"])
            * (get(plus) - get(minus))
            for plus, minus in (KEYS[:2], KEYS[2:])
        )
        return active - control

    for layer in sorted(stages, key=int):
        sites = {key: packet[layer] for key, packet in profile.items()}
        summaries[layer] = {
            "restricted_head_effect": signed(lambda key: sites[key]["native_swing"]),
            "rounded_head_allocation": {term: signed(lambda key: sites[key]["finite"][term]) for term in TERMS},
            "continuous_attribution": {term: signed(lambda key: sites[key]["continuous"][term]) for term in LINEAR},
        }
    return original | {"delayed_window_effects": summaries}
