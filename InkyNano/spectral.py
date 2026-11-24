"""Reference SVD on one *already dequantized* float matrix, out x in.

Do not feed packed NVFP4/MXFP8 codes here. Extraction/parity against the
pinned teacher is a prerequisite. Run a single matrix at a time.
"""
import torch


@torch.no_grad()
def nested_svd(weight, max_rank, *, randomized=False, seed=0):
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("expected a dequantized floating 2D matrix")
    if not 1 <= max_rank <= min(weight.shape):
        raise ValueError("invalid rank")
    w = weight.detach().float()
    if not torch.isfinite(w).all():
        raise ValueError("nonfinite weight")
    if randomized:
        q = min(min(w.shape), max_rank + 16)
        devices = [w.device.index if w.device.index is not None else torch.cuda.current_device()] if w.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            u, s, v = torch.svd_lowrank(w, q=q, niter=3)
        order = torch.argsort(s, descending=True)
        u, s, vh = u[:, order], s[order], v[:, order].T
    else:
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
    left = (u[:, :max_rank] * s[:max_rank]).contiguous()
    return left, vh[:max_rank].contiguous(), s.contiguous()


def apply_factors(x, left, right, rank):
    if not 1 <= rank <= min(left.shape[1], right.shape[0]):
        raise ValueError("invalid prefix rank")
    return (x @ right[:rank].T) @ left[:, :rank].T


@torch.no_grad()
def relative_error(weight, left, right, rank, x=None):
    if not 1 <= rank <= min(left.shape[1], right.shape[0]):
        raise ValueError("invalid prefix rank")
    if x is None:
        target = weight.float()
        pred = left[:, :rank].float() @ right[:rank].float()
    else:
        target = x.float() @ weight.float().T
        pred = apply_factors(x.float(), left.float(), right.float(), rank)
    denom = torch.linalg.vector_norm(target)
    error = torch.linalg.vector_norm(pred - target)
    return float(error / denom) if denom > 0 else (0.0 if error == 0 else float("inf"))
