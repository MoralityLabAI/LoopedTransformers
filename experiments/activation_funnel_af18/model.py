"""AF-18 shared-parameter native TypedLoop with 2/4/6/8 recurrent calls."""
from __future__ import annotations
from experiments.activation_funnel_af17.model import BellmanTypedPolicy

LOOPS=(2,4,6,8)
class SweepPolicy(BellmanTypedPolicy):
    def __init__(self,loops):
        if loops not in LOOPS:raise ValueError('unregistered recurrence depth')
        super().__init__()
        self.loops=loops
