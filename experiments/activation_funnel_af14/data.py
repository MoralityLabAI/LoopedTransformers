"""AF-14: prospective graph layouts outside AF-05..10's enumerated layout roster.

The Boolean-Horn rule semantics are kept from the pinned AF-05 oracle. Physical permutations 20..25 use distinct fixed seeds and were not used by AF-01 through AF-12. This changes graph layout, not logical rule
families. All downstream labels are derived independently from the Horn oracle.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
from functools import lru_cache
from itertools import product
import hashlib, json, random
import torch
from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05.experiment import Dataset as ParentDataset

FRESH = ('fresh_depth3','fresh_comp','fresh_depth4','fresh_depth5')
LAYOUTS = {'fresh_depth3':(30,), 'fresh_comp':(31,), 'fresh_depth4':(26,27), 'fresh_depth5':(28,29)}
DEPTH = {'fresh_depth3':3,'fresh_comp':3,'fresh_depth4':4,'fresh_depth5':5}


def physical_permutations(index:int):
    if index < 8:
        return gym.permutation_maps(index)
    if index not in range(26,32):raise ValueError('AF-14 unregistered layout')
    rng=random.Random(201411 + index*7933)
    atoms=list(range(gym.ATOMS));rules=list(range(gym.RULES))
    rng.shuffle(atoms);rng.shuffle(rules)
    return tuple(atoms),tuple(rules)


def encode(case:gym.Case)->gym.GraphBatch:
    if case.permutation < 8:
        return gym.encode_case(case)
    up,down=gym.logical_rules(case)
    atom_pos,rule_pos=physical_permutations(case.permutation)
    af=torch.zeros(gym.ATOMS,5);rf=torch.zeros(gym.RULES,3)
    pu=torch.zeros(gym.RULES,gym.ATOMS);hu=torch.zeros_like(pu)
    pd=torch.zeros_like(pu);hd=torch.zeros_like(pu)
    facts=gym.initial_facts(case)
    for logical in range(gym.ATOMS):
        i=atom_pos[logical]
        af[i]=torch.tensor([float(logical in facts),float(logical==gym.SUBGOALS[0]),
             float(logical==gym.SUBGOALS[1]),float(logical==gym.GOAL),1.])
    for k,(premises,head) in enumerate(up+down):
        i=rule_pos[k]
        rf[i]=torch.tensor([len(premises)/2.,float(k>=len(up)),1.])
        p=pu if k<len(up) else pd
        h=hu if k<len(up) else hd
        for node in premises:p[i,atom_pos[node]]=1.
        h[i,atom_pos[head]]=1.
    return gym.GraphBatch(af,rf,pu,hu,pd,hd,
      torch.tensor([atom_pos[n] for n in (*gym.SUBGOALS,gym.GOAL)]),
      torch.tensor(gym.oracle(case)[0],dtype=torch.long))


def stack(cases):
    rows=[encode(c) for c in cases]
    if not rows:raise ValueError('cannot stack empty fresh cases')
    return gym.GraphBatch(*(torch.stack([getattr(x,f.name) for x in rows])
                            for f in fields(gym.GraphBatch)))


@lru_cache(maxsize=4)
def cases_for(split:str):
    if split not in FRESH:raise ValueError('AF-14 private held-out roster')
    output=[]
    for layout in LAYOUTS[split]:
        for k0,k1,gate,facts in product(gym.KINDS,gym.KINDS,gym.GATES,
                                       product((0,1),repeat=4)):
            # A fresh-layout version of AF-05's held-out OR,OR -> AND case.
            special=(k0,k1,gate)==gym.STRESS_COMPOSITION
            if split=='fresh_comp' and not special:continue
            output.append(gym.Case(split,DEPTH[split],(k0,k1),gate,tuple(facts),layout))
    return tuple(output)


def sample_hash(case):
    b=encode(case)
    payload=[getattr(b,x).tolist() for x in ('atom_features','rule_features',
             'prem_up','head_up','prem_down','head_down','read_slots')]
    return gym.digest(payload)


def fresh_manifest():
    seen=set();rows={}
    # No committed holdout target/labels, only feature-hash receipts.
    old=set()
    for split in gym.SPLITS:
        for c in gym.cases_for(split):
            old.add(sample_hash(c))
    from experiments.activation_funnel_af11 import data as af11data
    for split in af11data.FRESH:
        for c in af11data.cases_for(split):
            b=af11data.encode(c)
            old.add(gym.digest([getattr(b,x).tolist() for x in
                ('atom_features','rule_features','prem_up','head_up','prem_down','head_down','read_slots')]))
    from experiments.activation_funnel_af12 import data as af12data
    for split in af12data.FRESH:
        for c in af12data.cases_for(split):
            b=af12data.encode(c)
            old.add(gym.digest([getattr(b,x).tolist() for x in
                ('atom_features','rule_features','prem_up','head_up','prem_down','head_down','read_slots')]))
    from experiments.activation_funnel_af13 import data as af13data
    for split in af13data.FRESH:
        for c in af13data.cases_for(split):
            b=af13data.encode(c)
            old.add(gym.digest([getattr(b,x).tolist() for x in
                ('atom_features','rule_features','prem_up','head_up','prem_down','head_down','read_slots')]))
    for split in FRESH:
        cases=cases_for(split)
        hashes=[sample_hash(c) for c in cases]
        if any(x in old or x in seen for x in hashes) or len(hashes)!=len(set(hashes)):
            raise ValueError('AF-14 fresh corpus overlaps prior encoded graph')
        seen.update(hashes)
        rows[split]={'cases':len(cases),'layouts':list(LAYOUTS[split]),
                     'depth':DEPTH[split],
                     'input_hashes_sha256':gym.digest(hashes),
                     'case_ids_sha256':gym.digest([c.id for c in cases])}
    result={'experiment':'AF-14','basis':'AF-05 closed finite Horn semantics',
            'novelty_scope':'new physical graph layouts 26-31, same Horn logical templates, prospective after AF12',
            'fresh_splits':rows,
            'previous_layouts':list(range(26)),
            'old_frozen_manifest_sha256':hashlib.sha256(
                ( __import__('pathlib').Path(__file__).resolve().parents[1]/
                 'activation_funnel_af05/corpus_manifest.json').read_bytes()).hexdigest()}
    result['manifest_sha256']=gym.digest(result)
    return result


class Dataset:
    def __init__(self,split):
        if split not in FRESH:
            self._old=ParentDataset(split)
            self.cases=self._old.cases
            self.batch=self._old.batch
            self.donors=self._old.donors
            return
        self._old=None
        self.cases=cases_for(split)
        self.batch=stack(self.cases)
        groups={}
        for i,c in enumerate(self.cases):groups.setdefault(c.skeleton,[]).append(i)
        self.donors={}
        for i,c in enumerate(self.cases):
            for j in (0,1):
                other=1-j
                pool=[k for k in groups[c.skeleton]
                  if self.cases[k].facts[2*other:2*other+2]==c.facts[2*other:2*other+2]
                  and gym.oracle(self.cases[k])[0][j]!=gym.oracle(c)[0][j]]
                if not pool:raise ValueError('donor coverage failure')
                donor=min(pool,key=lambda k:(sum(a!=b for a,b in zip(
                    self.cases[k].facts,c.facts)),self.cases[k].facts))
                self.donors[i,j]=(donor,gym.oracle_interchange(c,self.cases[donor],j))

    def get(self,indices):
        ix=torch.as_tensor(indices,dtype=torch.long)
        return gym.GraphBatch(*(getattr(self.batch,f.name).index_select(0,ix)
                          for f in fields(gym.GraphBatch)))

    def pair(self,indices,choices):
        if len(indices)!=len(choices):raise ValueError('mismatched donor choices')
        ix=[];labels=[]
        for i,j in zip(indices,choices):
            k,v=self.donors[int(i),int(j)];ix.append(k);labels.append(v)
        return self.get(ix),torch.tensor(labels,dtype=torch.long)
