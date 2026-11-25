"""Compile actual MC-03 clean and counterfactual trace witnesses.

This is a post-measurement compiler-integration test, selected for valid examples.
It does not change the complete scientific panel or its reported success rates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from research_gym.analysis.mechanistic_certificate import Statement, relation, encode
from research_gym.scripts.mechanistic_certificate import export_relation, file_hash
from research_gym.scripts.mc01_compiler_parity import read_wtns, satisfies

COMPILER_SHA256 = '85342c7ff332d948df7c0c50ecf201e6129349aef550ce873f3c811b79fe53a3'


def candidates(root: Path, seed: int):
    run = root/f'seed_{seed}'/'trained'
    metadata = json.loads((run/'metrics.json').read_text())
    for name in ('heldout.npz','replays.npz','selection.json','clean_trace.npz'):
        if file_hash(run/name) != metadata['file_sha256'][name]:
            raise ValueError(f'capture file pin failed: {name}')
    checkpoint = root/f'seed_{seed}'/'final.pt'
    if file_hash(checkpoint) != metadata['checkpoint_sha256']:
        raise ValueError('trained checkpoint pin failed')
    with np.load(run/'heldout.npz',allow_pickle=False) as z:
        predicted=z['logits'].argmax(-1)
        if not np.array_equal(predicted,z['predictions']):
            raise ValueError('heldout label/logit mismatch')
        valid=np.flatnonzero((predicted==z['targets']).all(0))
        if not len(valid):
            raise ValueError('no valid natural trace; compiler demo unavailable')
        item=int(valid[0]); graph=tuple(map(int,z['graphs'][item]));start=int(z['starts'][item])
        labels=(start,)+tuple(map(int,predicted[:,item]))
        natural=(Statement(graph,start,len(labels)-1,int(z['targets'][-1,item])),labels,
                 {'kind':'natural','item':item,'capture_sha256':metadata['file_sha256']['heldout.npz']})
    eligible=[i for i,r in enumerate(metadata['pairs']) if r['arms'][0]['counterfactual_trace_match']]
    if not eligible:
        raise ValueError('no valid intervention trace; compiler demo unavailable')
    index=eligible[0];row=metadata['pairs'][index];claim=row['claim']
    with np.load(run/'replays.npz',allow_pickle=False) as z:
        labels=(int(z[f'pair_{index}_current_logits'][0].argmax()),)+tuple(
            map(int,z[f'pair_{index}_future_logits'][:,0].argmax(-1)))
    observed=row['arms'][0]
    if labels!=(observed['post_patch_label'],*observed['future_labels']):
        raise ValueError('intervention label/logit mismatch')
    target=claim['paths']['counterfactual']
    patched=(Statement(tuple(claim['receiver_graph']),claim['donor_cursor'],claim['horizon'],target[-1]),
             labels,{'kind':'cursor_transfer','pair_index':index,'visit':row['visit'],
                     'capture_sha256':metadata['file_sha256']['replays.npz']})
    return [(s,l,{**origin,'seed':seed,'checkpoint_sha256':metadata['checkpoint_sha256']})
            for s,l,origin in (natural,patched)]


def compile_one(circom: Path, directory: Path, statement: Statement, labels, origin) -> dict:
    if file_hash(circom)!=COMPILER_SHA256:
        raise ValueError('Circom binary differs from pinned official release')
    emitted=export_relation(statement,labels,directory)
    if not emitted['witness_valid']:
        raise ValueError('captured trace fails reference relation')
    command=[str(circom),str(directory/'trace.circom'),'--O0','--r1cs','--wasm','--sym','--json','-o',str(directory)]
    p=subprocess.run(command,capture_output=True,text=True,timeout=90)
    (directory/'compile.log').write_text(p.stdout+p.stderr)
    if p.returncode:raise RuntimeError(p.stdout+p.stderr)
    p=subprocess.run(['node',str(directory/'trace_js/generate_witness.js'),str(directory/'trace_js/trace.wasm'),
                      str(directory/'witness.json'),str(directory/'observed.wtns')],
                      capture_output=True,text=True,timeout=30)
    if p.returncode:raise RuntimeError(p.stdout+p.stderr)
    prime,values=read_wtns(directory/'observed.wtns')
    constraints=json.loads((directory/'trace_constraints.json').read_text())['constraints']
    symbols={}
    for line in (directory/'trace.sym').read_text().splitlines():
        _,wire,_,name=line.split(',',3)
        if int(wire)<1 or name in symbols:raise ValueError('unexpected O0 symbol layout')
        symbols[name]=int(wire)
    input_data=json.loads((directory/'witness.json').read_text())
    seen=set()
    for name,array in input_data.items():
        for i,value in enumerate(array):
            index=symbols[f'main.{name}[{i}]']
            if index in seen or values[index]!=value:raise ValueError('witness/input mapping differs')
            seen.add(index)
    if seen!=set(range(1,len(values))) or not satisfies(constraints,values,prime):
        raise ValueError('compiled witness mapping or validity failed')
    rejected=0
    for index in sorted(seen):
        changed=values.copy();changed[index]=(changed[index]+1)%prime
        if satisfies(constraints,changed,prime):raise ValueError(f'mutation {index} accepted')
        rejected+=1
    receipt={**origin,'status':'COMPILED_OBSERVED_TRACE','statement':statement.identity(),
             'observed_labels':list(labels),'compiler_sha256':COMPILER_SHA256,
             'constraints':len(constraints),'witness_wires':len(values),'accepted_observed_witnesses':1,
             'single_wire_mutations_rejected':rejected,'proof_generated':False,
             'cryptographic_neural_source_binding':False,
             'selection':'first valid example in the already reported complete panel',
             'artifact_sha256':{name:file_hash(directory/name) for name in (
                 'trace.circom','trace.r1cs','trace_constraints.json','trace.sym','observed.wtns')}}
    (directory/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def run(circom: Path, root: Path, out: Path):
    if out.exists() and any(out.iterdir()):raise ValueError('empty output directory required')
    out.mkdir(parents=True,exist_ok=True)
    receipts=[]
    for seed in (201,202):
        for statement,labels,origin in candidates(root,seed):
            receipts.append(compile_one(circom,out/f"seed_{seed}_{origin['kind']}",statement,labels,origin))
    summary={'status':'COMPILED_OBSERVED_TRACES','receipts':receipts,
             'valid_witnesses':sum(r['accepted_observed_witnesses'] for r in receipts),
             'mutations_rejected':sum(r['single_wire_mutations_rejected'] for r in receipts),
             'proof_generated':False,'cryptographic_neural_source_binding':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--circom',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();run(a.circom.resolve(),a.run.resolve(),a.out.resolve())
