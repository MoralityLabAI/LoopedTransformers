"""Prospective fixed SAE/BPD projections after independent cached-v2 audits.

Cold import has no numerical dependencies. Numerical work needs only NumPy;
the future frozen host must admit RAM/thread budgets, wait for GPU termination,
pin every asset, and verify actual model/stage/terminal receipts first. No
training, alignment fitting, donor transfer, hypothesis test or model launch.
"""
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from .chunked_audit import checked,file_hash,read_json

TARGETS=dict(bf16='cf3db880c052137e7ff7a0dac1e498174f78e32fd3ae9a384aca341fb1d09186',
    nf4='8e80e7d748d5daf858a0b37dd7843d38952554ce3aabea1104e52cf1ef7663b3')


@dataclass
class FixedAssets:
    backend:str
    target_digest:str
    dictionaries:dict
    bank:dict
    asset:dict
    pins:dict
    MOCK_ONLY:bool=False


def load_fixed_assets(backend,expected_target_digest,guard):
    """Load the original CAL-only towers/random controls and qualified IO18 T2.

    This is not admission: the caller must already hold its frozen serialized
    host lease. Receipt/dictionary bindings must also be in that source seal.
    """
    from experiments.nan_ram28.common import read,sha
    from experiments.nan_story23.common import fit_asset
    from .probe_towers import load_npz
    checked(backend in TARGETS and expected_target_digest==TARGETS[backend],'Wrong backend/target condition')
    guard();asset=fit_asset(backend,2);checked(asset is not None,'Qualified fixed IO18 T2 asset required')
    checked(asset['receipt']['target_parameter_sha256_unchanged']==expected_target_digest,'BPD target-byte binding changed')
    inventory=read('experiments/nan_xref19/CACHE_INVENTORY.json')['backends'][backend]
    dictionaries={};pins={}
    for layer in (5,11,17,21):
        dictionaries[layer]=[]
        for name in (f'layer{layer}.npz',f'layer{layer}_random_initial.npz'):
            guard();pin=inventory['dictionaries'][str(layer)][name]
            checked(sha(pin['path'])==pin['sha256'],'Fixed SAE/random asset changed')
            dictionaries[layer].append(load_npz(pin['path']));pins[f'layer{layer}/{name}']=pin['sha256'];guard()
    guard();bank=load_npz(asset['checkpoint']);guard()
    pins.update(BPD_checkpoint=asset['checkpoint_sha256'],BPD_receipt=asset['receipt_sha256'],BPD_qualification=asset['audit_sha256'])
    return FixedAssets(backend,expected_target_digest,dictionaries,bank,asset,pins)


def grade_packet(text,r,g):
    """Original complete-packet endpoints; the trace cannot rescue an answer."""
    from experiments.nan_full26.common import parse
    parsed=parse(text,r);valid=parsed is not None
    if r['stage']=='nan_story23':
        final=bool(valid and parsed['final_room']==g['final_room'])
        checkpoints=bool(valid and parsed['intermediate_rooms']==g['intermediate_rooms'])
        path=bool(valid and parsed['path']==g['path'])
        grade=dict(format_valid=valid,final_correct=final,checkpoints_correct=checkpoints,path_correct=path,joint_correct=final and checkpoints and path)
    elif r['stage']=='nan_gym05':
        from experiments.nan_gym05.grade import score_value
        grade=dict(format_valid=valid,final_correct=bool(valid and score_value(parsed['answer'],g)['correct']))
    else:grade=dict(format_valid=valid,final_correct=bool(valid and parsed['answer']==g['answer']))
    return grade


def project_episode(base,audit,source,gold,assets,out,*,geometry_keys,expected_target_digest,guard):
    """One audited episode at a time; full codes stay in a private output file.

    Same frozen encoding, state-only controller, BF16 local-write rounding,
    mask tolerances (rtol2e-5/atol1e-5), write relative L2 <=.03, recurrence
    metrics and grade endpoints as FULL26. Target/terminal proof is external.
    """
    import numpy as np
    from experiments.nan_full26.host import decode
    from .probe_towers import encoding
    from experiments.nan_fit07.pointwise import masks
    from experiments.nan_story23.host import local_replay
    checked(audit['status']=='episode_audited' and source['id']==audit['source_id'],'Independent episode audit required')
    checked(assets.target_digest==expected_target_digest,'Asset/model target bytes differ')
    if not assets.MOCK_ONLY:
        checked(expected_target_digest==TARGETS[assets.backend],'Unrecognized real target bytes')
        checked(assets.asset['receipt']['target_parameter_sha256_unchanged']==expected_target_digest,'Fixed BPD target binding changed')
    base=Path(base);out=Path(out);checked(out.is_dir(),'Private host output directory required')
    record=read_json(base/audit['closed_file'],audit['closed_sha256'])
    checked(record['source_id']==source['id'] and record['mode']==audit['mode'],'Audited row changed')
    checked(len(record['fragments'])==len(audit['prefixes']),'Audited prefix extent changed')
    aggregates=defaultdict(list);codes=[];bpd=[];mask_errors=[];write_errors=[];prefix_keys=[]
    for fragment,proof in zip(record['fragments'],audit['prefixes']):
        guard();checked(fragment['file']==proof['file'] and fragment['sha256']==proof['sha256'],'Audited fragment changed')
        p=read_json(base/fragment['file'],fragment['sha256']);state=base/p['state_file']
        checked(file_hash(state)==p['state_sha256'],'Audited physical states changed')
        with np.load(state,allow_pickle=False) as z:
            coords={(i['loop'],i['layer'],i['site']):decode(z[i['key']]) for i in p['state_index']}
            checked(len(coords)==40 and all(np.isfinite(v).all() for v in coords.values()),'Incomplete numerical states')
            for phase in (1,2):
                for layer in (5,11,17,21):
                    H=coords[phase,layer,'residual_after_layer'][None].astype(np.float64)
                    sae,random=assets.dictionaries[layer]
                    Z,mse,l2,error=encoding(H,sae,4);_,rmse,_,re=encoding(H,random,4)
                    aggregates[layer,phase].append([float(mse[0]),float(rmse[0]),float(l2[0])])
                    if layer==21:codes.append(Z[0])
                x=coords[phase,21,'down_proj_input'][None].astype(np.float64)
                shadow=masks(x,assets.bank,assets.asset['receipt']['final_checkpoint']['CI_config'])[0];bpd.append(shadow)
                if record['mode']!='native':
                    actual=z['actual_masks'][phase-1];native=decode(z['native_at_reached_input'])[phase-1]
                    expected=shadow if record['mode']=='reached_state' else np.asarray(assets.asset['static_CAL_mask'])[phase-1] if record['mode']=='static_CAL_mean' else np.ones(8)
                    mask_errors.append(float(np.max(np.abs(actual-expected))))
                    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=1e-5)
                    predicted=local_replay(x[0],native,actual,assets.bank,assets.asset['receipt']['training_contract']['ablation'],phase-1)
                    y=coords[phase,21,'down_proj_output'];rel=float(np.linalg.norm(predicted-y)/max(float(np.linalg.norm(y)),1e-12))
                    checked(rel<=.03,'Independent local-write replay failed');write_errors.append(rel)
            for layer in (5,11,17,21):
                A=coords[1,layer,'residual_after_layer'].astype(float);B=coords[2,layer,'residual_after_layer'].astype(float)
                aggregates[layer,'recurrence'].append([float(np.linalg.norm(B-A)),float(A@B/max(float(np.linalg.norm(A)*np.linalg.norm(B)),1e-12))])
        key=geometry_keys.get(fragment['file'],dict(signature=None,reason='missing_complete_batch_audit'))
        if key['signature'] is not None:checked(key['signature']==proof['context_sha256'],'Geometry proof differs from audited context')
        prefix_keys.append(dict(file=fragment['file'],history_sha256=p['history_sha256'],prefix_ids_sha256=p['prefix_ids_sha256'],
            complete_cache_lineage_signature=key['signature'],geometry_exclusion=key['reason']))
        guard()
    codefile=out/(assets.backend+'_'+Path(audit['closed_file']).stem+'_codes.npz')
    with codefile.open('xb') as stream:np.savez(stream,codes=np.asarray(codes),shadow_masks=np.asarray(bpd))
    guard()
    return dict(status='MOCK_ONLY' if assets.MOCK_ONLY else 'fixed_assets_projected',backend=assets.backend,
        source_id=source['id'],root=source['root'],split=source['split'],stage=source['stage'],mode=record['mode'],lane=record['lane'],
        grade=grade_packet(record['text'],source,gold),prefixes=len(record['fragments']),physical_vectors=40*len(record['fragments']),
        metrics=[dict(layer=key[0],loop=key[1],mean=np.mean(v,axis=0).tolist()) for key,v in aggregates.items()],
        maximum_mask_abs_error=max(mask_errors,default=None),maximum_local_write_relative_L2_error=max(write_errors,default=None),
        code_file=codefile.name,code_file_sha256=file_hash(codefile),closed_file=audit['closed_file'],closed_file_sha256=audit['closed_sha256'],
        prefix_keys=prefix_keys,fixed_asset_pins=assets.pins,CPU_threads=1,formal_comparisons=0,sealed_test_accessed=False,
        target_stage_terminal_receipt_verification_still_required=True)
