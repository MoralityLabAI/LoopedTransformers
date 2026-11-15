"""RAM-06 frozen stronger non-neural TRAIN-only empirical tactic controls.

Source-agnostic: only compiler-observed source-state text, printed pending
goals, and genuine train-only candidate outcomes. No test family labels
or original proof commands.
"""
from __future__ import annotations
from collections import defaultdict
import re

def goal_shape(state):
    goals=state["goals"]
    if not goals:return ("empty",False)
    target=goals[0].split("⊢",1)[-1]
    if "↔" in target:
        kind="iff"
    elif "→" in target or "∀" in target:
        kind="implication"
    elif "∧" in target:
        kind="conjunction"
    elif "=" in target:
        kind="equality"
    elif any(c in target for c in ("≤","≥","<",">")):
        kind="inequality"
    else:
        kind="other"
    return kind,len(goals)>1

def reward(row):
    return {0:-.6,1:1.5,2:.3,3:3}[row["label"]]+.4*row["delta"]

class Frequency:
    def __init__(self,rows):
        groups=defaultdict(list)
        for row in rows:
            if row["development"]:
                raise AssertionError("train-only frequency received development labels")
            groups[(len(row["before"]["goals"])>1,row["family"])].append(reward(row))
        self.table={key:sum(v)/len(v) for key,v in groups.items()}
    def rank(self,state,pool):
        multi=len(state["goals"])>1
        return [item for i,item in sorted(enumerate(pool),key=lambda x:
            (-self.table.get((multi,x[1][0]),-.6),x[0]))]

class ShapeFrequency(Frequency):
    def __init__(self,rows,prior_weight=8):
        super().__init__(rows)
        if prior_weight!=8:
            raise AssertionError("RAM06 prior weight was changed")
        self.prior_weight=prior_weight
        groups=defaultdict(list)
        for row in rows:
            key=(goal_shape(row["before"]),row["family"])
            groups[key].append(reward(row))
        self.group={key:(sum(v),len(v)) for key,v in groups.items()}
    def rank(self,state,pool):
        shape=goal_shape(state)
        multi=shape[1]
        def score(family):
            baseline=self.table.get((multi,family),-.6)
            total,n=self.group.get((shape,family),(0.,0))
            return (total+self.prior_weight*baseline)/(n+self.prior_weight)
        return [item for i,item in sorted(enumerate(pool),key=lambda x:
            (-score(x[1][0]),x[0]))]
