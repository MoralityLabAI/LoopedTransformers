"""PEF-09: ungated SVD feature ablation in frozen PEF-07 loopers.

Protocol: research/pef09/PROTOCOL.md, committed 2bbd53e BEFORE heldout runs.
The 48 probe contexts alone determine pair directions and random energy
comparators. All 16 focal masks receive the same fixed affine projection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from research.pef08 import pef08 as p8
from research.pef07 import pef07 as p7

SEEDS = (103, 131, 167)
TASKS = ("coalition", "threshold")
PAIRS = ((0, 1), (2, 3))
LOOPS = (4, 8, 12)
RANDOM_COUNT = 8
CANDIDATES = 256
DIM = 32
SIGNATURE = '849d039e47dec472b0b4835d24928e9a547e262bf84fd7dfed16741df9911ac1'
PROTO_SHA = '2bbd53ec098ed5c256fe9b015599cc076c877a4f'


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def tensor_digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def verify_original_evidence(folder, catalog, signature, generated_manifest):
    ledger = json.loads((folder/'sealed_training_ledger.json').read_text())
    stored = json.loads((folder/'source_split.json').read_text())
    if (ledger.get('source_sha256', ledger.get('source_split_sha256')) != SIGNATURE or
        stored.get('sha256', stored.get('manifest_sha256')) != SIGNATURE):
        raise AssertionError('PEF-07 source manifest identifier mismatch')
    encoded = p7.encoded if hasattr(p7,'encoded') else p7.canonical
    data = {k: v for k, v in stored.items() if k not in ('sha256','manifest_sha256')}
    if hashlib.sha256(encoded(data)).hexdigest() != SIGNATURE:
        raise AssertionError('sealed manifest corrupted')
    if stored.get('contexts_in_sha256_order',stored.get('sorted_contexts')) != generated_manifest.get('contexts_in_sha256_order',generated_manifest.get('sorted_contexts')):
        raise AssertionError('context ordering differs')
    if stored.get('context_assignment',stored.get('split_assignment')) != generated_manifest.get('context_assignment',generated_manifest.get('split_assignment')):
        raise AssertionError('context split differs')
    if not ledger.get('all_six_trunks_sealed_before_scoring',ledger.get('all_checkpoints_frozen_before_heldout',False)):
        raise AssertionError('missing freeze provenance')
    recs = p8.checkpoint_rows(ledger)
    if len(recs)!=6 or {(v['seed'],v['task']) for v in recs} != {(seed,task) for seed in SEEDS for task in TASKS}:
        raise AssertionError('expected the six original PEF-07 trunks')
    if {k:len(v) for k,v in catalog.items()} != {'train':2560,'probe':768,'heldout':768}:
        raise AssertionError('source sizes changed')
    for rec in recs:
        if rec['steps']!=450 or rec.get('source_sha256',rec.get('source_manifest_sha256'))!=SIGNATURE:
            raise AssertionError('nonregistered checkpoint metadata')
    return ledger, recs


def all_base_differences(state, pair):
    """Full 4-corner pair effects at all 4 assignments of other focal bits.

    The 16 source masks are grouped by context, in natural ascending order.
    Shape [48*4,32] for original PEF-07 probe or heldout cohorts.
    """
    if state.ndim!=2 or state.shape[1]!=DIM or len(state)%16:
        raise ValueError('source must be 16-cell, D=32 context cubes')
    i,j=pair
    if i>=j or i not in range(4) or j not in range(4):
        raise ValueError('invalid distinct focal pair')
    base_bits=[k for k in range(4) if k not in (i,j)]
    cube=state.reshape(-1,16,DIM)
    result=[]
    for b in range(4):
        base=((b&1)<<base_bits[0])+(((b>>1)&1)<<base_bits[1])
        a,c=1<<i,1<<j
        result.append(cube[:,base+a+c]-cube[:,base+a]-cube[:,base+c]+cube[:,base])
    return torch.stack(result,dim=1).reshape(-1,DIM)


def fixed_sign(unit):
    unit = unit.clone()
    loc=int(unit.abs().argmax())
    if unit[loc]<0:
        unit=-unit
    return unit


def leading_direction(matrix, *, fallback=True):
    if matrix.ndim!=2 or matrix.shape[1]!=DIM:
        raise ValueError('expected [N, 32] signal')
    data=matrix.to(dtype=torch.float64)
    # SVD is over the uncentered 192x32 pair-difference stack.
    _, sv, right=torch.linalg.svd(data, full_matrices=False)
    degenerate=bool(sv[0]<1e-10)
    if degenerate and fallback:
        unit=torch.zeros(DIM,dtype=torch.float64)
        unit[0]=1.0
    elif degenerate:
        raise AssertionError('unexpected degenerate matrix')
    else:
        unit=fixed_sign(right[0])
    return unit, float(sv[0]), degenerate


def projection_energy(probe, mu, direction):
    x=probe.double()-mu.double()
    u=direction.double()
    return float(((x@u)**2).mean())


def random_energy_matches(probe, mu, selected, *, seed, task, loop, pair_idx):
    init=9009+1000*seed+10*loop+2*pair_idx+(100000 if task=='threshold' else 0)
    generator=torch.Generator(device='cpu').manual_seed(init)
    vectors=torch.randn((CANDIDATES,DIM),generator=generator,dtype=torch.float64)
    vectors=torch.nn.functional.normalize(vectors,dim=1)
    energies=((probe.double()-mu.double())@vectors.T).square().mean(0)
    target=projection_energy(probe,mu,selected)
    differences=(torch.log(energies+1e-10)-torch.log(torch.tensor(target+1e-10,dtype=torch.float64))).abs()
    ranked=sorted(range(CANDIDATES),key=lambda idx:(float(differences[idx]),idx))
    chosen=ranked[:RANDOM_COUNT]
    return [{
        'name':f'matched_random_{i}',
        'candidate_index':int(idx),
        'unit':vectors[idx].tolist(),
        'probe_projection_energy':float(energies[idx]),
        'target_probe_energy_ratio':float(energies[idx]/max(target,1e-15)),
    } for i,idx in enumerate(chosen)]


def projector(state, mu, direction):
    """Apply a single global affine rank-one projector to every input row."""
    if state.ndim!=2 or state.shape[1]!=DIM:
        raise ValueError('expected [batch,32]')
    v=direction.to(dtype=state.dtype,device=state.device)
    center=mu.to(dtype=state.dtype,device=state.device)
    if v.shape!=(DIM,) or center.shape!=(DIM,):
        raise ValueError('expected a 32-vector and center')
    if not torch.allclose(v@v,v.new_tensor(1.),rtol=1e-5,atol=1e-5):
        raise AssertionError('unit vector required')
    return state-((state-center)@v)[:,None]*v[None,:]


def projector_many(state, mu, directions):
    """[K,N,D] global affine ablations, independent of input masks."""
    x=state.float()
    u=directions.to(dtype=x.dtype,device=x.device)
    center=mu.to(dtype=x.dtype,device=x.device)
    if u.ndim!=2 or u.shape[1]!=DIM:
        raise ValueError('directions must be [K,32]')
    if not torch.allclose(u.square().sum(-1),torch.ones(len(u)),rtol=1e-5,atol=1e-5):
        raise AssertionError('all directions must be unit length')
    projected=(x-center)@u.T
    return x[None,:,:]-projected.T[:,:,None]*u[:,None,:]


def select_probe_features(model, rec, probe_bits):
    seed,task=rec['seed'],rec['task']
    trace=p8.states_by_loop(model,probe_bits)
    fit=p8.ridge_fit(model,trace[14],p8.labels(probe_bits,task))
    centers={loop:trace[loop].double().mean(0) for loop in LOOPS}
    rows=[]
    for t in LOOPS:
        probe=trace[t]
        mu=centers[t]
        pair_directions={}
        for pair in PAIRS:
            v,sv,degenerate=leading_direction(all_base_differences(probe,pair))
            pair_directions[str(pair)]=v
        global_pc,global_sv,_=leading_direction(probe.double()-mu)
        for idx,pair in enumerate(PAIRS):
            chosen=pair_directions[str(pair)]
            other=pair_directions[str(PAIRS[1-idx])]
            # All 8 random nulls are selected by PROBE projection energy alone.
            randoms=random_energy_matches(probe,mu,chosen,seed=seed,task=task,loop=t,pair_idx=idx)
            array=[
                {'name':'target_pair_svd','unit':chosen.tolist(),
                 'probe_projection_energy':projection_energy(probe,mu,chosen)},
                {'name':'other_pair_svd','unit':other.tolist(),
                 'probe_projection_energy':projection_energy(probe,mu,other)},
                {'name':'global_pc','unit':global_pc.tolist(),
                 'probe_projection_energy':projection_energy(probe,mu,global_pc)},
            ]+randoms
            for condition in array:
                condition['unit_sha256']=tensor_digest(torch.tensor(condition['unit'],dtype=torch.float64))
            rows.append({
                'loop':t,'pair':list(pair),'center':mu.tolist(),
                'pair_top_singular_value':float(torch.linalg.svdvals(
                    all_base_differences(probe,pair).double())[0]),
                'pair_degenerate': bool(torch.linalg.svdvals(
                    all_base_differences(probe,pair).double())[0]<1e-10),
                'global_pc_singular_value':global_sv,
                'conditions':array,
            })
    return {'seed':seed,'task':task,'checkpoint_sha256':rec['checkpoint_sha256'],
            'fitted_h14_ridge_sha256':p7.ridge_digest(fit) if hasattr(p7,'ridge_digest')
                                     else hashlib.sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes() for t in fit)).hexdigest(),
            'ridge_fit':[x.tolist() for x in fit],
            'rows':rows}


def native_score(model, state):
    logits=model.head(state)
    return logits[:,1]-logits[:,0]


def pair_magnitudes(margins):
    return {f'{p[0]}{p[1]}':float(all_base_differences(margins.reshape(-1,1).repeat(1,DIM),p)[:,0].abs().mean()) for p in PAIRS}


def pair_magnitudes_scalar(margins):
    return {f'{p[0]}{p[1]}':float(all_base_differences(margins[:,None].expand(-1,DIM),p)[:,0].abs().mean()) for p in PAIRS}


def errors_by_mask(margin,truth):
    wrong=((margin>=0).long()!=truth).reshape(-1,16)
    return [int(x) for x in wrong.sum(0)]


def class_breakdown(margin,truth):
    predicted=(margin>=0).long()
    return {'correct':int((predicted==truth).sum()),'total':int(len(truth)),
            'recall_0':float((predicted[truth==0]==0).float().mean()),
            'recall_1':float((predicted[truth==1]==1).float().mean()),
            'errors_by_focal_mask':errors_by_mask(margin,truth)}


@torch.no_grad()
def replay_batched(model, bits, state_by_condition, start):
    if start not in LOOPS:
        raise ValueError('not a registered visit')
    conditions,n,d=state_by_condition.shape
    expanded_bits=bits.repeat(conditions,1)
    memory=state_by_condition.reshape(-1,d)
    native=memory if start==12 else None
    for visit in range(start,14):
        if visit<12:
            injected=expanded_bits[:,visit]
        else:
            injected=torch.full((len(expanded_bits),),p8.CONTINUATION[visit-12],dtype=torch.long)
        memory=model.step(memory,model.bit_embedding(injected))
        if visit==11:
            native=memory
    if native is None:
        raise AssertionError('no h12 state')
    return native.reshape(conditions,n,d),memory.reshape(conditions,n,d)


def ridge_from_selection(selection):
    return tuple(torch.tensor(x,dtype=torch.float64) for x in selection['ridge_fit'])


@torch.no_grad()
def score_model(model, selection, held_bits, trace):
    seed,task=selection['seed'],selection['task']
    truth=p8.labels(held_bits,task)
    original_h12=native_score(model,trace[12])
    fit=ridge_from_selection(selection)
    original_h14=p8.ridge_margin(trace[14],fit)
    if class_breakdown(original_h12,truth)['correct']!=768 or class_breakdown(original_h14,truth)['correct']!=768:
        raise AssertionError('PEF-07 original 768/768 did not reproduce')
    original_magnitudes=pair_magnitudes_scalar(original_h12)
    original_late=pair_magnitudes_scalar(original_h14)
    data=[]
    for row in selection['rows']:
        t=row['loop']
        pair=tuple(row['pair'])
        other=PAIRS[1-PAIRS.index(pair)]
        target_key=f'{pair[0]}{pair[1]}'
        other_key=f'{other[0]}{other[1]}'
        mu=torch.tensor(row['center'],dtype=torch.float64)
        directions=torch.tensor([c['unit'] for c in row['conditions']],dtype=torch.float64)
        states=projector_many(trace[t],mu,directions)
        final,late=replay_batched(model,held_bits,states,t)
        scored=[]
        for k,cond in enumerate(row['conditions']):
            native=native_score(model,final[k])
            later=p8.ridge_margin(late[k],fit)
            values=pair_magnitudes_scalar(native)
            late_values=pair_magnitudes_scalar(later)
            att={p:1-values[p]/max(original_magnitudes[p],1e-10) for p in original_magnitudes}
            late_att={p:1-late_values[p]/max(original_late[p],1e-10) for p in original_late}
            vec=directions[k].float()
            mean_energy=float((((trace[t]-mu.float())@vec)**2).mean())
            scored.append({
                'condition':cond['name'],'unit_sha256':cond['unit_sha256'],
                'candidate_index':cond.get('candidate_index'),
                'probe_projection_energy':cond['probe_projection_energy'],
                'heldout_projection_energy':mean_energy,
                'native_h12':class_breakdown(native,truth),
                'h14_fixed_ridge':class_breakdown(later,truth),
                'mean_abs_h12_margin_change':float((native-original_h12).abs().mean()),
                'h12_pair_magnitudes':values, 'h12_pair_attenuation':att,
                'h12_pair_selectivity':att[target_key]-att[other_key],
                'h14_pair_magnitudes':late_values,'h14_pair_attenuation':late_att,
                'h14_pair_selectivity':late_att[target_key]-late_att[other_key],
            })
        random_controls=[x['h12_pair_selectivity'] for x in scored if x['condition'].startswith('matched_random_')]
        # For an even count, median of two middle values (conventional sample median).
        random_controls_sorted=sorted(random_controls)
        median=(random_controls_sorted[3]+random_controls_sorted[4])/2
        excess=scored[0]['h12_pair_selectivity']-median
        data.append({
            'loop':t,'pair':list(pair),'original_h12_pair_magnitudes':original_magnitudes,
            'original_h14_pair_magnitudes':original_late,
            'target_excess_over_matched_random_median':excess,
            'random_median_selectivity':median,
            'registered_h12_error_condition':scored[0]['native_h12']['correct']<=752,
            'conditions':scored,
        })
    return {'seed':seed,'task':task,'original_h12_correct':768,
            'original_h14_correct':768,'frozen_checkpoint_sha256':selection['checkpoint_sha256'],
            'frozen_probe_selection_sha256':digest(selection),'rows':data}


def run(evidence, output):
    os.environ.setdefault('DNNL_MAX_CPU_ISA','AVX2')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    evidence,output=Path(evidence),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    catalog,sig,manifest=p8.source_data()
    ledger,recs=verify_original_evidence(evidence,catalog,sig,manifest)
    modelsha={}
    selections=[]
    # Freeze all probe-selected directions and ridge decoders BEFORE heldout inference.
    for rec in recs:
        model=p8.checked_model(evidence,rec)
        modelsha[(rec['seed'],rec['task'])]=p8.weights_digest(model)
        row=select_probe_features(model,rec,catalog['probe'])
        selections.append(row)
        print('PEF09_PROBE_FROZEN',row['seed'],row['task'],digest(row),flush=True)
    seal={'campaign':'PEF-09','protocol_commit':PROTO_SHA,
          'source_sha256':sig,'checkpoint_ledger_sha256':hashlib.sha256(
              (evidence/'sealed_training_ledger.json').read_bytes()).hexdigest(),
          'all_selections_probe_only':True,'all_selections_frozen_before_heldout':True,
          'selections':selections}
    (output/'frozen_selection_ledger.json').write_bytes(canonical(seal))
    seal_hash=hashlib.sha256((output/'frozen_selection_ledger.json').read_bytes()).hexdigest()
    print('PEF09_SELECTION_LEDGER_SHA256',seal_hash,flush=True)
    models=[]
    for rec,sel in zip(recs,selections,strict=True):
        model=p8.checked_model(evidence,rec)
        trace=p8.states_by_loop(model,catalog['heldout'])
        r=score_model(model,sel,catalog['heldout'],trace)
        if p8.weights_digest(model)!=modelsha[(rec['seed'],rec['task'])]:
            raise AssertionError('frozen model changed during ablation')
        models.append(r)
        print('PEF09_SCORED',r['seed'],r['task'],flush=True)
    decision=[]
    for seed in SEEDS:
        c=next(r for r in models if r['seed']==seed and r['task']=='coalition')
        pairrows=[r for r in c['rows'] if r['loop']==8]
        passed=all(r['target_excess_over_matched_random_median']>0.15
                   and r['registered_h12_error_condition'] for r in pairrows)
        decision.append({'seed':seed,'both_pairs_pass':passed,'per_pair_excess':[
            {'pair':r['pair'],'excess':r['target_excess_over_matched_random_median'],
             'target_h12_correct':r['conditions'][0]['native_h12']['correct']}
            for r in pairrows]})
    positive=sum(d['both_pairs_pass'] for d in decision)
    results={'campaign':'PEF-09','protocol_commit':PROTO_SHA,'input_source_sha256':sig,
             'frozen_selection_ledger_sha256':seal_hash,
             'registered_diagnostic_decision':'criterion_met' if positive>=2 else 'criterion_not_met',
             'registered_seed_count':positive,'registered_per_seed':decision,
             'original_checkpoints_and_models_unchanged':True,
             'task_list':list(TASKS),'seeds':list(SEEDS),'loops':list(LOOPS),
             'selection_method':'probe_full_base_uncentered_rank1_SVD_ungated',
             'torch_version':str(torch.__version__),'models':models}
    (output/'summary.json').write_bytes(canonical(results))
    compact={'campaign':results['campaign'],'diagnostic':results['registered_diagnostic_decision'],
             'registered_seed_count':positive,'by_seed':decision,
             'table':[{'seed':r['seed'],'task':r['task'],'loop':i['loop'],'pair':i['pair'],
                       'target_h12_correct':i['conditions'][0]['native_h12']['correct'],
                       'target_h14_correct':i['conditions'][0]['h14_fixed_ridge']['correct'],
                       'target_selectivity':i['conditions'][0]['h12_pair_selectivity'],
                       'random_median_selectivity':i['random_median_selectivity'],
                       'excess':i['target_excess_over_matched_random_median'],
                       'other_pair_correct':i['conditions'][1]['native_h12']['correct'],
                       'global_pc_correct':i['conditions'][2]['native_h12']['correct'],
                       'probe_energy':i['conditions'][0]['probe_projection_energy']}
                      for r in models for i in r['rows']]}
    (output/'compact_receipt.json').write_bytes(canonical(compact))
    print(json.dumps({'diagnostic':results['registered_diagnostic_decision'],'by_seed':decision},indent=2),flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.evidence,args.out)


if __name__=='__main__':
    main()
