"""Prospective production B2 capacity and B1 parity procedure."""
from time import monotonic


ABSOLUTE_LOGIT_TOLERANCE = 0.0625
RELATIVE_LOGIT_TOLERANCE = 0.01


def compare(observed, baseline):
    if observed.shape != baseline.shape:
        raise RuntimeError("B2 and B1 readout shapes differ")
    delta = (observed - baseline).abs()
    bound = ABSOLUTE_LOGIT_TOLERANCE + RELATIVE_LOGIT_TOLERANCE * baseline.abs()
    if (delta > bound).any():
        raise RuntimeError("B2 full-vocabulary logits exceed the frozen B1 parity envelope")
    if observed.argmax(dim=-1).tolist() != baseline.argmax(dim=-1).tolist():
        raise RuntimeError("B2 and B1 greedy choices differ on a frozen preflight row")
    return {"max_abs_logit_delta": float(delta.max()), "greedy_tokens_equal": True,
            "vocabulary_rows_compared": int(delta.numel())}


def repeated(tokens, count):
    return (tokens * ((count + len(tokens) - 1) // len(tokens)))[:count]


def preflight(engine, inputs, maximum_position, guard):
    """Use real prompt pairs plus explicit padded pruning and capacity fixtures."""
    if len(inputs) != 2 or any(len(ids) < 3 for ids in inputs):
        raise ValueError("Both CAL prompts require at least three tokens")
    if max(map(len, inputs)) + 2 > maximum_position:
        raise ValueError("Admission position is smaller than the CAL capture")
    ragged = [inputs[0], inputs[1][:min(len(inputs[1]), len(inputs[0]) - 1)]]
    report = {}
    for depth in (1, 2):
        checks = {}
        began = monotonic()
        try:
            for label, pair in (("actual_CAL", inputs), ("padded_CAL", ragged)):
                physical = max(map(len, pair))
                engine.reset(depth)
                guard(began, 2, physical)
                batch = engine.prefill_logits(pair)
                step_ids = batch.argmax(dim=-1).tolist()
                canvas_prefill = engine.cache.attention.tolist()
                expected = [[1] * len(ids) + [0] * (physical - len(ids)) for ids in pair]
                if canvas_prefill != expected:
                    raise RuntimeError("Initial B2 attention mask differs from the planned padding")
                guard(began, 2, physical + 1)
                stepped = engine.advance_logits([0, 1], step_ids)
                next_id = int(stepped[1].argmax())
                guard(began, 1, physical + 2)
                pruned = engine.advance_logits([1], [next_id])
                if engine.cache.attention.tolist() != [expected[1] + [1, 1]]:
                    raise RuntimeError("The B2 survivor lost its physical padding hole")
                one_initial = []
                one_step = []
                for index, ids in enumerate(pair):
                    engine.reset(depth)
                    guard(began, 1, len(ids))
                    one_initial.append(engine.prefill_logits([ids]))
                    guard(began, 1, len(ids) + 1)
                    one_step.append(engine.advance_logits([0], [step_ids[index]]))
                guard(began, 1, len(pair[1]) + 2)
                one_pruned = engine.advance_logits([0], [next_id])
                checks[label] = {
                    "prefill": [compare(batch[i:i + 1], one_initial[i]) for i in range(2)],
                    "increment": [compare(stepped[i:i + 1], one_step[i]) for i in range(2)],
                    "pruned_increment": compare(pruned, one_pruned),
                    "padded_tokens_at_prefill": 2 * physical - sum(map(len, pair)),
                    "final_survivor_physical_position": physical + 2,
                }

            engine.reset(depth)
            physical = maximum_position - 1
            long_pair = [repeated(inputs[0], physical), repeated(inputs[1], physical - 1)]
            guard(began, 2, physical)
            capacity = engine.prefill_logits(long_pair)
            guard(began, 2, maximum_position)
            engine.advance_logits([0, 1], capacity.argmax(dim=-1).tolist())
            if engine.cache.physical_length != maximum_position:
                raise RuntimeError("The B2 maximum-position fixture did not fill its bound")
            guard(began, 2, maximum_position)
            report[str(depth)] = {"checks": checks, "tested_max_position": maximum_position,
                                  "KV_bytes_at_max_position": engine.cache.cache_bytes(),
                                  "wall_seconds_including_all_checks": monotonic() - began}
        finally:
            engine.reset(depth)
    return report
