"""Extract surviving goal discharges from a native Lean search trace."""

PROGRESS_RETURN_GAP = 0.125


def committed_goal_discharges(episode):
    accepted = []
    for event in episode["events"]:
        if event["kind"] == "rollback":
            restore = event["restore_script_length"]
            if not 0 <= restore <= len(accepted):
                raise ValueError("rollback does not identify a committed prefix")
            del accepted[restore:]
        if event["kind"] != "tactic" or event["status"] == "rejected":
            continue
        before = len(event["input"]["state"]["goals"])
        after = len(event["state_after"]["goals"])
        discharges = 0 if event["status"] == "certificate_rejected" else max(0, before - after)
        accepted.append((event["tactic"], discharges))
    if [tactic for tactic, _ in accepted] != episode["accepted_script"]:
        raise ValueError("committed progress differs from the recorded proof prefix")
    return sum(discharges for _, discharges in accepted)


def tied_failure_progress(episodes):
    left, right = episodes
    if (not all(ep["within_cpu_budget"] and not ep["verified_closed"] for ep in episodes)
            or left["tactic_checks"] != right["tactic_checks"]):
        return None
    return committed_goal_discharges(left), committed_goal_discharges(right)


def tied_failure_preference(episodes):
    progress = tied_failure_progress(episodes)
    if progress is None:
        return 0
    left, right = progress
    return int(left > right) - int(right > left)
