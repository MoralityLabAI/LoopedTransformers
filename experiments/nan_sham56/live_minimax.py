"""Build the registered peak-minimizing sham from live staged head weights."""
from types import SimpleNamespace

from experiments.nan_delay39.contrasts import numpy_state

from .minimax import WIDTH, derivative_columns, form_basis, optimize


def designed_sham(engine, h, original, gradient, active, radius, tag, cfg, dose, check, np):
    import torch

    norm = engine.cache.loop.base.norm
    source = SimpleNamespace(gamma=numpy_state(norm.weight), eps=norm.variance_epsilon)
    axes, support = form_basis(h, original, gradient, active, radius, tag, cfg, np)
    design = derivative_columns(source, h, axes, radius, np)
    weights = engine.head.weight
    if weights.shape[0] < 64:
        raise ValueError("Peak design requires at least 64 vocabulary rows")
    projected = np.empty((weights.shape[0], WIDTH), dtype=np.float32)
    block_rows = dose["block_rows"]
    for start in range(0, weights.shape[0], block_rows):
        check()
        stop = min(start + block_rows, weights.shape[0])
        block = weights[start:stop].detach().to(device="cpu", dtype=torch.float32).numpy()
        projected[start:stop] = block @ design
    q, before, after = optimize(projected, np)
    return axes @ q.astype(np.float64), support, {"predicted_peak_before": before, "predicted_peak_after": after}
