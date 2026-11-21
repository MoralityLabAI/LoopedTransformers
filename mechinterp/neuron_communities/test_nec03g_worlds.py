"""NEC-03G Boolean world registry tests: independent tasks and sealed cohort."""
from __future__ import annotations

import hashlib

import pytest
import torch

from mechinterp.neuron_communities.nec03g_worlds import (
    SCHEMA, COUNTS, DEV_COUNT, CONFIRM_COUNT, additive_control_truth,
    context_catalog, digest, essential, gate, program_truth, registry,
    target_bits, teacher_functions, worlds, write_registry,
)


def test_128_distinct_essential_teacher_functions_and_position_sets():
    items=worlds()
    assert len(items)==DEV_COUNT+CONFIRM_COUNT==128
    assert len({x["truth_table_16"] for x in items})==128
    assert len({tuple(x["focal_positions"]) for x in items})==128
    assert len({x["trunk_seed"] for x in items})==128
    assert {x["cohort"] for x in items[:64]}=={"development"}
    assert {x["cohort"] for x in items[64:]}=={"confirmation"}
    assert all(5<=x["truth_table_16"].bit_count()<=11 for x in items)
    assert all(essential(x["truth_table_16"]) for x in items)
    assert all(x["focal_positions"][0]<=4 and x["focal_positions"][-1]>=7
               for x in items)
    assert all(x["world_sha256"]==digest({k:v for k,v in x.items()
                                         if k!="world_sha256"}) for x in items)


def test_program_truth_digest_and_matched_additive_control():
    for w in worlds()[:8]:
        formula=w["program"]
        assert program_truth(formula["structure"],*formula["gates"],
                             formula["negated_inputs_mask"])==w["truth_table_16"]
        control=additive_control_truth(w["truth_table_16"])
        assert control.bit_count()==w["truth_table_16"].bit_count()
        assert control!=w["truth_table_16"]  # disjoint teacher functions
    with pytest.raises(ValueError):
        gate(0,1,"unsupported")
    with pytest.raises(ValueError):
        essential(-1)


def test_4096_full_context_closed_histories_and_disjoint_source_labels():
    world=worlds()[0]
    catalog,manifest=context_catalog(world)
    assert manifest["world_sha256"]==world["world_sha256"]
    assert manifest["sha256"]==digest({k:v for k,v in manifest.items()
                                       if k!="sha256"})
    assert {k:len(rows) for k,rows in catalog.items()}=={
        "train":2560,"development":768,"reserved":768}
    assert set.union(*(set(rows) for rows in catalog.values()))=={
        tuple((n>>j)&1 for j in range(12)) for n in range(4096)}
    for name,rows in catalog.items():
        expected=COUNTS[name]
        assert len(rows)==16*expected
        for k in range(0,len(rows),16):
            quartet=rows[k:k+16]
            remaining=manifest["context_positions"]
            assert len({tuple(x[i] for i in remaining) for x in quartet})==1
            assert len({tuple(x[i] for i in world["focal_positions"])
                        for x in quartet})==16


def test_truth_table_target_matches_all_focal_interventions_and_class_prior():
    world=worlds()[0]
    catalog,_=context_catalog(world)
    sample=torch.tensor(catalog["development"][:16],dtype=torch.long)
    p=target_bits(sample,world)
    a=target_bits(sample,world,control=True)
    assert p.tolist()==[(world["truth_table_16"]>>k)&1 for k in range(16)]
    control=additive_control_truth(world["truth_table_16"])
    assert a.tolist()==[(control>>k)&1 for k in range(16)]
    assert int(p.sum())==int(a.sum())
    with pytest.raises(ValueError):
        target_bits(sample[:,:11],world)


def test_registry_exposes_confirm_commitments_only(tmp_path):
    a=registry()
    b=registry()
    assert a==b
    assert a["registry_sha256"]==digest({k:v for k,v in a.items()
                                        if k!="registry_sha256"})
    assert len(a["dev_full_world_specs"])==64
    assert len(a["confirmation_commitments"])==64
    assert all(set(v)=={"world_id","world_sha256"}
               for v in a["confirmation_commitments"])
    assert not a["confirmation_outcomes_exposed"]
    assert a["confirmation_requires_separate_preflight_authorization"]
    out=tmp_path/"frozen_registry.json"
    assert write_registry(out)==a
    assert hashlib.sha256(out.read_bytes()).hexdigest()==hashlib.sha256(
        out.read_bytes()).hexdigest()
