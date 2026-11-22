"""Learn the one keep-or-abandon decision from paired Lean search receipts."""

from ..data import canonical
from ..recovery_progress import PROGRESS_RETURN_GAP, tied_failure_preference


def learn_recovery(policy, episodes, optimizer, *, imitate, progress_credit=False):
    import torch
    from torch.nn import functional as F

    left, right = episodes
    records = [[event for event in ep["events"] if event["kind"] == "recovery"] for ep in episodes]
    if any(len(items) != 1 for items in records) or not all(ep["within_cpu_budget"] for ep in episodes):
        return {"decisions": 0, "loss": None, "recovery_pairs": 0}
    a, b = (items[0] for items in records)
    choice = a["selected_index"]
    if choice != b["selected_index"] or a["input"]["parent_menu"] != b["input"]["parent_menu"]:
        raise ValueError("recovery training must expose both worlds to the same proposal")
    logits, values, returns = [], [], []
    for ep, record in zip((left, right), (a, b), strict=True):
        score, value = policy.recover(canonical(record["input"]), ep["depth"])
        logits.append(score)
        values.append(value)
        returns.append(float(ep["verified_closed"]) - 0.025 * ep["tactic_checks"])
    preference = tied_failure_preference(episodes) if progress_credit else 0
    progress = {"progress_preference": preference} if progress_credit else {}
    if imitate:
        terms = [(max(0.0, reward), -score.log_softmax(-1)[choice])
                 for score, reward in zip(logits, returns, strict=True) if reward > 0]
        if terms:
            loss = torch.stack([weight * term for weight, term in terms]).sum() / sum(weight for weight, _ in terms)
            decisions = len(terms)
        elif preference:
            left_choice = choice if preference > 0 else 1 - choice
            loss = -0.5 * PROGRESS_RETURN_GAP * (logits[0].log_softmax(-1)[left_choice]
                                                 + logits[1].log_softmax(-1)[1 - left_choice])
            decisions = 2
        else:
            return {"decisions": 0, "loss": None, "recovery_pairs": 1, **progress}
    else:
        probabilities = [score.softmax(-1)[choice] for score in logits]
        advantage = returns[0] - returns[1] + PROGRESS_RETURN_GAP * preference
        # The identical proposal in both worlds has probability one half.
        paired = -advantage * (probabilities[0] - probabilities[1]) / 2
        critic = torch.stack([F.smooth_l1_loss(value, value.new_tensor(reward))
                              for value, reward in zip(values, returns, strict=True)]).mean()
        loss = paired + 0.5 * critic
        decisions = 2
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite recovery loss")
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {"decisions": decisions, "loss": float(loss.detach()), "recovery_pairs": 1,
            "mean_return": sum(returns) / 2, **progress}
