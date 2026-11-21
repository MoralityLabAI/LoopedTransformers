"""One CPU lane streams immutable closed episodes while the GPU generates."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import time
from .common import ROOT,THREADS,append,read,rows,sha,verify,write,load_sources,parse

def labels():
    result={}
    for stage in ('nan_gym05','nan_comp14','nan_ctx08','nan_ctx21','nan_story23'):
        file=Path('runs/nan_gym05/cohort/host_labels.jsonl') if stage=='nan_gym05' else Path('experiments')/stage/'fixtures/host_labels.jsonl'
        for r in rows(file):result[stage+'/'+r['id']]=r
    return result

def decode(array):
    import numpy as np
    assert array.dtype==np.uint16
    # Exact BF16 -> float32 expansion, independent of Torch's cast path.
    return (array.astype(np.uint32)<<16).view(np.float32)

def run(a):
    for key in THREADS:os.environ[key]='1'
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();verify()
    import numpy as np
    from mechinterp.nanbeige_bpd_gym.probe_towers import encoding,load_npz
    from experiments.nan_story23.common import fit_asset
    from experiments.nan_fit07.pointwise import masks
    from experiments.nan_story23.host import local_replay
    import torch
    torch.set_num_threads(1)
    artifact=Path(a.artifact_root);out=artifact/'host';out.mkdir(parents=True,exist_ok=False)
    source={r['id']:r for r in load_sources()[0]};gold=labels();seen=set();dictionaries={};banks={};assets={}
    inventory=read('experiments/nan_xref19/CACHE_INVENTORY.json')['backends']
    for backend in ('nf4','bf16'):
        dictionaries[backend]={}
        for layer in (5,11,17,21):
            descriptions=inventory[backend]['dictionaries'][str(layer)]
            selected=[]
            for name in (f'layer{layer}.npz',f'layer{layer}_random_initial.npz'):
                pin=descriptions[name];assert sha(pin['path'])==pin['sha256'];selected.append(load_npz(pin['path']))
            dictionaries[backend][layer]=selected
        assets[backend]=fit_asset(backend,2);assert assets[backend]
        banks[backend]=load_npz(assets[backend]['checkpoint'])
    def process(backend,entry):
        base=artifact/backend;closed=base/entry['file'];assert sha(closed)==entry['sha256']
        record=read(closed);r=source[record['source_id']];g=gold[r['id']]
        parsed=parse(record['text'],r);valid=parsed is not None
        if r['stage']=='nan_story23':
            final=bool(valid and parsed['final_room']==g['final_room'])
            checkpoints=bool(valid and parsed['intermediate_rooms']==g['intermediate_rooms'])
            path=bool(valid and parsed['path']==g['path'])
            grade=dict(format_valid=valid,final_correct=final,checkpoints_correct=checkpoints,path_correct=path,joint_correct=final and checkpoints and path)
        elif r['stage']=='nan_gym05':
            from experiments.nan_gym05.grade import score_value
            grade=dict(format_valid=valid,final_correct=bool(valid and score_value(parsed['answer'],g)['correct']))
        else:grade=dict(format_valid=valid,final_correct=bool(valid and parsed['answer']==g['answer']))
        aggregates=defaultdict(list);codes=[];bpd=[];mask_errors=[];write_errors=[];prefix_keys=[]
        for fragment in record['fragments']:
            file=base/fragment['file'];assert sha(file)==fragment['sha256'];p=read(file)
            state=base/p['state_file'];assert sha(state)==p['state_sha256']
            with np.load(state,allow_pickle=False) as z:
                coords={(i['loop'],i['layer'],i['site']):decode(z[i['key']]) for i in p['state_index']}
                assert len(coords)==40 and all(np.isfinite(v).all() for v in coords.values())
                for phase in (1,2):
                    for layer in (5,11,17,21):
                        H=coords[phase,layer,'residual_after_layer'][None].astype(np.float64)
                        sae,random=dictionaries[backend][layer]
                        Z,mse,l2,error=encoding(H,sae,4);_,rmse,_,re=encoding(H,random,4)
                        aggregates[layer,phase].append([float(mse[0]),float(rmse[0]),float(l2[0])])
                        if layer==21:codes.append(Z[0])
                    x=coords[phase,21,'down_proj_input'][None].astype(np.float64)
                    bank=banks[backend];shadow=masks(x,bank,assets[backend]['receipt']['final_checkpoint']['CI_config'])[0]
                    bpd.append(shadow)
                    if record['mode']!='native':
                        actual=z['actual_masks'][phase-1];native=decode(z['native_at_reached_input'])[phase-1]
                        expected=shadow if record['mode']=='reached_state' else np.asarray(assets[backend]['static_CAL_mask'])[phase-1] if record['mode']=='static_CAL_mean' else np.ones(8)
                        mask_errors.append(float(np.max(np.abs(actual-expected))))
                        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=1e-5)
                        predicted=local_replay(x[0],native,actual,bank,assets[backend]['receipt']['training_contract']['ablation'],phase-1)
                        y=coords[phase,21,'down_proj_output'];rel=float(np.linalg.norm(predicted-y)/max(float(np.linalg.norm(y)),1e-12))
                        assert rel<=.03;write_errors.append(rel)
                # Record recurrence at each physical residual site, no token n.
                for layer in (5,11,17,21):
                    A=coords[1,layer,'residual_after_layer'].astype(float);B=coords[2,layer,'residual_after_layer'].astype(float)
                    aggregates[layer,'recurrence'].append([float(np.linalg.norm(B-A)),float(A@B/max(float(np.linalg.norm(A)*np.linalg.norm(B)),1e-12))])
                prefix_keys.append(dict(file=fragment['file'],history_sha256=p['history_sha256'],prefix_ids_sha256=p['prefix_ids_sha256'],
                    batch_size=p['batch_size'],padded_length=p['padded_length']))
        codefile=out/(backend+'_'+closed.stem+'_codes.npz')
        with codefile.open('xb') as f:np.savez(f,codes=np.asarray(codes),shadow_masks=np.asarray(bpd))
        summary=dict(backend=backend,source_id=r['id'],root=r['root'],split=r['split'],stage=r['stage'],mode=record['mode'],lane=record['lane'],
            grade=grade,prefixes=len(record['fragments']),physical_vectors=40*len(record['fragments']),
            metrics=[dict(layer=key[0],loop=key[1],mean=np.mean(v,axis=0).tolist()) for key,v in aggregates.items()],
            maximum_mask_abs_error=max(mask_errors,default=None),maximum_local_write_relative_L2_error=max(write_errors,default=None),
            code_file=codefile.name,code_file_sha256=sha(codefile),closed_file=closed.name,closed_file_sha256=sha(closed),
            prefix_keys=prefix_keys,CPU_threads=1,sealed_test_accessed=False)
        name=backend+'_'+closed.stem+'_audit.json';write(out/name,summary)
        append(out/'events.jsonl',dict(event='episode_audited',backend=backend,file=name,sha256=sha(out/name)))
    while True:
        changed=False
        for backend in ('nf4','bf16'):
            journal=artifact/backend/'closed.jsonl'
            if not journal.exists():continue
            # Append-only writer, only consume complete newline-terminated
            # records. Every referenced episode/state is write-once closed.
            with journal.open('rb') as f:lines=f.read().splitlines(keepends=True)
            for line in lines:
                if not line.endswith(b'\n'):continue
                import json
                entry=json.loads(line);key=(backend,entry['file'])
                if key in seen:continue
                process(backend,entry);seen.add(key);changed=True
        if (artifact/'model_lane_terminal.json').exists():
            # Marker is written only after actual worker processes exit.
            break
        if not changed:time.sleep(10)
    from .paired import run as compare
    compare(artifact)
    write(out/'receipt.json',dict(status='completed',closed_episodes_audited=len(seen),new_model_calls=0,
        fixed_SAE_dictionaries=8,random_controls=8,qualified_BPD_T2_checkpoints=2,sealed_test_accessed=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact-root',required=True);run(p.parse_args())
