"""Refine the independent-box winner bound using the native token-ID tie rule."""

from experiments.nan_carrybound88.bounds import logit_box, norm_box
from experiments.nan_readpath37.run import bf16, decode_bf16


TILE = 768
MAX_RIVALS = 8


def tiles(head, check, np):
    head.stream.seek(head.base + head.weight_offset)
    for start in range(0, head.vocab, TILE):
        check()
        count = min(TILE, head.vocab - start)
        raw = head.stream.read(2 * head.width * count)
        if len(raw) != 2 * head.width * count:
            raise ValueError("Short vocabulary tile")
        yield start, decode_bf16(raw, np).reshape(count, head.width).astype(np.float64)


def outranks(logit, token, reference, original_id):
    return (logit > reference) | ((logit == reference) & (token < original_id))


def refine(head, lo, hi, winner, bound, check, np):
    lower, upper = norm_box(lo, hi, head.gamma, head.eps, np)
    center = (lower.astype(np.float64) + upper) / 2
    radius = (upper.astype(np.float64) - lower) / 2
    choices = [[] for _ in winner]
    threat_count = np.zeros(len(winner), dtype=np.int64)
    for start, weights in tiles(head, check, np):
        _, peak = logit_box(center, radius, weights, np)
        ids = np.arange(start, start + len(weights))
        active = (ids != winner[:, None]) & outranks(peak, ids, bound[:, None], winner[:, None])
        threat_count += active.sum(axis=1)
        for i, row in enumerate(active):
            indices = np.flatnonzero(row)
            if len(indices):
                order = np.lexsort((indices, -peak[i, indices]))[:MAX_RIVALS]
                additions = [(float(-peak[i, j]), start + int(j)) for j in indices[order]]
                choices[i] = sorted(choices[i] + additions)[:MAX_RIVALS]
    output = []
    cases, sources = [], []
    for i, (count, shortlist) in enumerate(zip(threat_count, choices, strict=True)):
        outcome = {"possible_outranking_tokens": int(count), "greedy_token_certified": int(count) == 0,
                   "witnesses_tested": 0, "different_greedy_token_witness": False,
                   "original_greedy_token_witness": False}
        output.append(outcome)
        if not count:
            continue
        anchor = head.row(int(winner[i])).astype(np.float64)
        unique = {lo[i].tobytes(): lo[i], hi[i].tobytes(): hi[i]}
        for _, token in shortlist:
            slope = (head.row(token).astype(np.float64) - anchor) * head.gamma
            proposal = np.where(slope >= 0, hi[i], lo[i])
            unique[proposal.tobytes()] = proposal
        for value in unique.values():
            cases.append(value)
            sources.append(i)
        outcome["witnesses_tested"] = len(unique)
    if cases:
        normalized = head.norm(np.stack(cases)).astype(np.float64)
        best = np.full(len(cases), -np.inf)
        leaders = np.full(len(cases), -1, dtype=np.int32)
        for start, weights in tiles(head, check, np):
            scores = bf16(normalized @ weights.T, np)
            local = scores.argmax(axis=1)
            peak = scores[np.arange(len(cases)), local]
            replace = peak > best
            best[replace] = peak[replace]
            leaders[replace] = start + local[replace]
        for i, token in zip(sources, leaders, strict=True):
            field = "original_greedy_token_witness" if token == winner[i] else "different_greedy_token_witness"
            output[i][field] = True
    return output


def fixture(np):
    values = np.asarray([2., 3., 3., 3., 4.])
    ids = np.asarray([1, 2, 3, 4, 5])
    np.testing.assert_array_equal(outranks(values, ids, 3., 3), [False, True, False, False, True])
    return {"ordered_tie_fixture_passed": True}
