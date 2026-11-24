"""Stream exact BF16 head logits and clean-referenced full-vocabulary statistics."""
from experiments.nan_readpath37.run import bf16, decode_bf16


def summarize_blocks(blocks, rows, ref, labels, np):
    columns = np.arange(rows)
    winners = np.zeros(rows, dtype=np.int64)
    top = np.full(rows, -np.inf)
    logz = np.full(rows, -np.inf)
    moment = np.zeros(rows)
    squared_error = np.zeros(rows)
    maximum_delta = np.zeros(rows)
    bare = np.full((rows, labels.shape[1]), np.nan)
    offset = 0
    for logits in blocks:
        if logits.ndim != 2 or logits.shape[0] != rows or not np.isfinite(logits).all():
            raise ValueError("Full-head chunk is incomplete or nonfinite")
        logits = logits.astype(np.float64, copy=False)
        highest = logits.argmax(axis=1)
        local_top = logits[columns, highest]
        replace = local_top > top
        winners[replace] = (highest + offset)[replace]
        top[replace] = local_top[replace]
        old_z = logz
        chunk_z = local_top + np.log(np.exp(logits - local_top[:, None]).sum(axis=1))
        logz = np.logaddexp(old_z, chunk_z)
        native = logits[ref]
        delta = logits - native
        moment = (np.exp(old_z[ref] - logz[ref]) * moment
                  + (np.exp(native - logz[ref, None]) * delta).sum(axis=1))
        squared_error += (delta**2).sum(axis=1)
        maximum_delta = np.maximum(maximum_delta, np.max(abs(delta), axis=1))
        stop = offset + logits.shape[1]
        i, j = np.where((labels >= offset) & (labels < stop))
        bare[i, j] = logits[i, labels[i, j] - offset]
        offset = stop
    kl = logz - logz[ref] - moment
    if not offset or not np.isfinite(bare).all() or not np.isfinite(kl).all() or kl.min() < -1e-10:
        raise ValueError("Full-head normalization did not close")
    return {"vocab": offset, "winner": winners, "top_logit": top, "logz": logz, "bare": bare,
            "max_abs_delta": maximum_delta, "rms_delta": np.sqrt(squared_error / offset),
            "KL_clean_to_edit": np.maximum(kl, 0)}


def full_head(head, states, references, labels, block_rows, check, np):
    hidden = head.norm(states).astype(np.float64)
    stride = 2 * head.width

    def chunks():
        head.stream.seek(head.base + head.weight_offset)
        for start in range(0, head.vocab, block_rows):
            check()
            rows = min(block_rows, head.vocab - start)
            raw = head.stream.read(rows * stride)
            if len(raw) != rows * stride:
                raise ValueError("Incomplete native vocabulary weight block")
            weights = decode_bf16(raw, np).reshape(rows, head.width).astype(np.float64)
            yield bf16(hidden @ weights.T, np)

    result = summarize_blocks(chunks(), len(hidden), references, labels, np)
    if result["vocab"] != head.vocab:
        raise ValueError("Did not consume all checkpoint vocabulary rows")
    return result


def verify_reference(np):
    rng = np.random.default_rng(44)
    dense = rng.normal(size=(5, 17))
    refs = np.array([0, 0, 0, 3, 3])
    labels = np.array([[0, 16], [3, 6], [8, 9], [0, 10], [5, 2]])
    dense[0, [0, 16]] = 100
    result = summarize_blocks((dense[:, i:i+4] for i in range(0, 17, 4)), len(dense), refs, labels, np)
    norm = np.logaddexp.reduce(dense, axis=1)
    normalized = dense - norm[:, None]
    kl = (np.exp(normalized[refs]) * (normalized[refs] - normalized)).sum(axis=1)
    delta = dense - dense[refs]
    np.testing.assert_allclose(result["KL_clean_to_edit"], kl, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(result["logz"], norm, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(result["winner"], dense.argmax(axis=1))
    np.testing.assert_array_equal(result["bare"], np.take_along_axis(dense, labels, axis=1))
    np.testing.assert_allclose(result["rms_delta"], np.sqrt((delta**2).mean(axis=1)), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(result["max_abs_delta"], abs(delta).max(axis=1), atol=1e-12, rtol=1e-12)
    return len(dense)
