"""PPT-A/PPT-B checked-pool research gyms; run python -m ppt_gym --help.

These gyms train and evaluate policies against previously kernel-checked tactic
menus. New Lean theorem certification belongs to the independent Lean-CGRT
collector and DataGoblin verifier.
"""
from .core import Config, PlasticProofTransformer, FastHead, RetrievalControl

__all__ = ("Config", "PlasticProofTransformer", "FastHead", "RetrievalControl")
