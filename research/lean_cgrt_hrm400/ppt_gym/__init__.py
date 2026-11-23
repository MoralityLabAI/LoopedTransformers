"""PPT-A/PPT-B checked-pool research gyms; run python -m ppt_gym --help.

These gyms train and evaluate policies against previously kernel-checked tactic
menus. New Lean theorem certification belongs to the independent Lean-CGRT
collector and DataGoblin verifier.
"""
__all__ = ("Config", "PlasticProofTransformer", "FastHead", "RetrievalControl")


def __getattr__(name):
    if name in __all__:
        from . import core

        return getattr(core, name)
    raise AttributeError(name)
