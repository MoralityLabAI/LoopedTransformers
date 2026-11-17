"""Prospective, depth-matched control for an ARG-03 readout-blind fiber.

This is a NEW time-local arm, separate from the original one-shot t=1
random-context continuation. At each specified depth t, start with the
same clean h_t and inject independent random content into the unscored
context positions, scaled to the *measured full-state displacement*
of the structured swap h_t^swap-h_t^clean.

The next readout q(F(h_t)) should be compared across matched controls.
The model caller must verify exact pre-continuation readout equality,
per-case norm parity and that all continuation arms use the same model
weights and context. No model forward pass is performed in this module.
"""
from __future__ import annotations

def match_random_direction(clean, structured, random_direction, *,
                           prefix_width:int=8, scored_tail:int=3,
                           eps:float=1e-10):
    """Return clean + norm-matched random prefix perturbation, Torch tensors.

    Args: clean, structured and random_direction have [batch, sequence, width].
    The random source is a DIRECTION, not an already propagated random branch.
    Norm matching is per case over ALL sequence/state coordinates at that t.
    Only the prefix_width tokens immediately before scored_tail are edited.

    This implementation deliberately refuses near-zero or NaN random directions
    rather than silently introducing an arbitrary unit perturbation.
    """
    import torch
    if any(x.ndim!=3 for x in (clean,structured,random_direction)):
        raise ValueError("State shapes must be [B,S,D]")
    if clean.shape!=structured.shape or clean.shape!=random_direction.shape:
        raise ValueError("States and direction must share shape")
    if (type(prefix_width) is not int or type(scored_tail) is not int
            or prefix_width<=0 or scored_tail<1
            or prefix_width+scored_tail>clean.shape[1]):
        raise ValueError("Invalid context window/tail")
    if not (torch.isfinite(clean).all() and torch.isfinite(structured).all()
            and torch.isfinite(random_direction).all()):
        raise ValueError("Nonfinite states")
    start=clean.shape[1]-scored_tail-prefix_width
    stop=clean.shape[1]-scored_tail
    structured_diff=(structured-clean).float()
    reference_norm=structured_diff.reshape(clean.shape[0],-1).norm(dim=-1)
    v=torch.zeros_like(clean,dtype=torch.float32)
    v[:,start:stop]=random_direction[:,start:stop].float()
    random_norm=v.reshape(clean.shape[0],-1).norm(dim=-1)
    if bool((random_norm<=eps).any()) or bool((reference_norm<=eps).any()):
        raise ValueError("Zero structured or random displacement cannot be calibrated")
    scale=(reference_norm/random_norm).reshape(-1,1,1)
    adjusted=(clean.float()+v*scale).to(clean.dtype)
    # Explicit null-space parity at the current readout-position indices.
    if not torch.equal(adjusted[:,stop:],clean[:,stop:]):
        raise AssertionError("Scored token positions altered")
    actual=(adjusted.float()-clean.float()).reshape(clean.shape[0],-1).norm(dim=-1)
    relative=(actual-reference_norm).abs()/reference_norm.clamp_min(eps)
    if float(relative.max().item())>0.005:
        raise AssertionError("Per-case displacement matching failed (possibly low precision)")
    return adjusted, {
        "schema":"arg03.depth_state_matched_random.v1",
        "window":prefix_width, "scored_tail":scored_tail,
        "reference_l2":reference_norm.tolist(),
        "adjusted_l2":actual.tolist(),
        "max_relative_norm_error":float(relative.max().item()),
        "intervention": "time_local_depth_t_new_arm",
        "original_H2a_unchanged":True
    }
