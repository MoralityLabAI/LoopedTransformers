"""NEC-03L source-registry tests, no native model training."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    SCHEMA,DEVELOPMENT_ROOT_COUNT,EMBARGOED_ROOT_COUNT,
    additive_control_truth,context_catalog,digest,essential,labels,
    program_truth,registration,worlds,write_registration,
)


def test_128_structurally_fresh_5_essential_bit_worlds_and_sealed_embargo():
    root=worlds()
    assert len(root)==128
    assert len({w["truth_table_32"] for w in root})==128
    assert len({tuple(w["focal_positions"]) for w in root})==128
    assert len({w["native_model_seed"] for w in root})==128
    assert all(w["cohort"]=="development" for w in root[:96])
    assert all(w["cohort"]=="embargoed" for w in root[96:])
    assert all(essential(w["truth_table_32"]) for w in root)
    assert all(10<=w["truth_table_32"].bit_count()<=22 for w in root)
    assert all(len(w["focal_positions"])==5 and
               len(set(w["focal_positions"]))==5 for w in root)
    assert all(w["world_sha256"]==digest({
        k:v for k,v in w.items() if k!="world_sha256"}) for w in root)


def test_grammar_reconstructs_exact_32_bit_teacher_and_prior_control():
    for w in worlds()[:32]:
        p=w["program"]
        assert program_truth(
            p["structure"],p["gates"],p["literal_inversion_mask"])==w["truth_table_32"]
        assert additive_control_truth(w["truth_table_32"]).bit_count()==w["truth_table_32"].bit_count()
        assert additive_control_truth(w["truth_table_32"])!=w["truth_table_32"]
    with pytest.raises(ValueError):
        essential(-1)
    with pytest.raises(ValueError):
        program_truth("unknown",["and"]*4,0)


def test_5_input_context_source_closure_and_80_24_24_partition():
    w=worlds()[0]
    catalog,manifest=context_catalog(w)
    assert {k:len(v) for k,v in catalog.items()}=={
        "train":2560,"development":768,"reserved":768}
    assert len({row for group in catalog.values() for row in group})==4096
    assert manifest["source_sha256"]==digest({
        k:v for k,v in manifest.items() if k!="source_sha256"})
    assert manifest["source_fit_context_count"]==40
    assert manifest["source_rank_context_count"]==40
    for group in catalog.values():
        for i in range(0,len(group),32):
            block=group[i:i+32]
            assert len(block)==32
            assert len({tuple(row[p] for p in manifest["context_positions"])
                        for row in block})==1
            assert len({tuple(row[p] for p in w["focal_positions"])
                        for row in block})==32


def test_tensor_teacher_targets_match_entire_full_32_focal_mask():
    w=worlds()[1]
    catalog,_=context_catalog(w)
    bits=torch.tensor(catalog["development"][:32],dtype=torch.long)
    y=labels(bits,w)
    z=labels(bits,w,control=True)
    expected=[(w["truth_table_32"]>>m)&1 for m in range(32)]
    alternative=[(additive_control_truth(w["truth_table_32"])>>m)&1 for m in range(32)]
    assert y.tolist()==expected
    assert z.tolist()==alternative
    assert sum(expected)==sum(alternative)
    with pytest.raises(ValueError):
        labels(bits[:,:11],w)


def test_embargoed_world_cannot_be_scored_and_registry_hash_is_repeatable(tmp_path):
    with pytest.raises(PermissionError,match="embargoed"):
        context_catalog(worlds()[96])
    with pytest.raises(PermissionError,match="embargoed"):
        context_catalog(worlds()[127])
    reg=registration()
    assert reg==registration()
    assert reg["registry_sha256"]==digest({
        k:v for k,v in reg.items() if k!="registry_sha256"})
    assert reg["five_input_development_worlds"]==DEVELOPMENT_ROOT_COUNT==96
    assert reg["five_input_embargoed_worlds"]==EMBARGOED_ROOT_COUNT==32
    assert len(reg["new_five_input_embargoed_world_commitments"])==32
    assert all(set(z)=={"world_id","world_sha256"}
               for z in reg["new_five_input_embargoed_world_commitments"])
    assert reg["embargoed_models_trained"]==0
    assert reg["original_four_input_confirmation_contexts_scored"]==0
    p=tmp_path/"five_input_registry.json"
    assert write_registration(p)==reg
    assert json.loads(p.read_text())==reg
