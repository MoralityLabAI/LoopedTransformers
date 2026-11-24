"""Source/metadata-only work accounting; never changes model admission or order."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import sys
from .bounded_cpu_engineering import CPUBudget


def token_cap(row):
    return 64 if row['stage']=='nan_story23' else 256 if row['schema']['type']=='grid' else 128


def capture_layout(spans,layers,phases=2):
    """Physical module dimensions, not one residual width for all five sites."""
    layout={}
    for layer in layers:
        prefix=f'model.layers.{layer}.'
        down=spans[prefix+'mlp.down_proj.weight'];output=spans[prefix+'self_attn.o_proj.weight']
        norm=spans[prefix+'post_attention_layernorm.weight']
        if any(p.dtype!='BF16' for p in (down,output,norm)):raise ValueError('The registered stored states are BF16')
        if len(down.shape)!=2 or len(output.shape)!=2:raise ValueError('Physical projections require matrix weights')
        hidden=down.shape[0]
        if output.shape[0]!=hidden or norm.shape!=(hidden,):raise ValueError('Incompatible residual widths')
        layout[layer]=dict(residual_after_layer=hidden,down_proj_input=down.shape[1],down_proj_output=hidden,
            o_proj_input=output.shape[1],o_proj_output=hidden)
    return layout,phases*2*sum(sum(sites.values()) for sites in layout.values())


def run(artifact,snapshot,out):
    from experiments.nan_ram28.common import ROOT,read,sha,write,verify
    from experiments.nan_mlp27.common import load_sources,intervention_rows
    from experiments.nan_mlp27.engine import Engine
    seal=verify();cfg=read(ROOT/'config.json');public,pins=load_sources();assert pins==seal['source_pins']
    bindings=read('experiments/nan_full26/tokenizer_receipt.json')['contexts']
    preflight_file=Path(artifact)/'nf4/preflight.json';preflight=read(preflight_file);assert preflight['status']=='passed'
    engine=Engine.__new__(Engine);engine.capacity=preflight['capacity']
    from mechinterp.nanbeige_bpd_gym.lazy_bf16_storage import CheckpointStore
    from experiments.nan_storyaudit30.audit import WIDTHS
    snapshot=Path(snapshot);metadata=CheckpointStore(snapshot);model_config=read(snapshot/'config.json')
    layout,payload=capture_layout(metadata.spans,cfg['layers_zero_based'],phases=model_config['num_loops'])
    assert model_config['num_loops']==2 and all(widths==WIDTHS for widths in layout.values())
    assert all(w['down_proj_input']==model_config['intermediate_size'] and w['o_proj_input']==model_config['num_attention_heads']*model_config['head_dim'] for w in layout.values())
    headers={}
    import hashlib
    for file in {v.path for v in metadata.spans.values()}:
        with file.open('rb') as stream:
            prefix=stream.read(8);size=int.from_bytes(prefix,'little');assert 0<size<=64*1024**2
            headers[file.name]=hashlib.sha256(prefix+stream.read(size)).hexdigest()
    def tokens(r):return bindings[r['id']]['tokens']
    def probe(B,P):
        candidates=[p for p in engine.capacity if p['status']=='passed' and p['batch']==B and p['length']>=P
            and p['includes_reached_state_route'] and (B==1 or p['right_padding']>0)]
        assert candidates,'No registered probe admits a source-only shape'
        return min(candidates,key=lambda p:p['length'])
    # Exactly the frozen baseline scheduler's +256 capacity margin, followed
    # by all three routes and all 72 source worlds. No reply is consulted.
    batches=[];index=0
    while index<len(public):
        rows=[public[index]]
        if index+1<len(public) and engine.admitted(2,max(tokens(public[index]),tokens(public[index+1]))+256):rows.append(public[index+1])
        batches.append(('baseline','native',rows));index+=len(rows)
    baseline_batches=len(batches)
    ab=intervention_rows(public);assert len(public)==1549 and len(ab)==72
    for route in cfg['AB_routes']:
        for row in ab:
            assert engine.admitted(2,tokens(row)+256)
            batches.append(('AB',route,[row,row]))
    groups=defaultdict(lambda:Counter());counts=Counter();row_slots=prefixes=captures=0
    for lane,route,rows in batches:
        row_slots+=len(rows);caps=[token_cap(r) for r in rows];prefixes+=sum(caps)
        for r,cap in zip(rows,caps):
            g=groups[lane,route,r['stage']];g['trajectory_slots']+=1;g['row_prefixes_at_full_cap']+=cap
        for step in range(max(caps)):
            active=[i for i,c in enumerate(caps) if step<c];B=len(active);P=max(tokens(rows[i])+step for i in active)
            p=probe(B,P);counts[B,p['length']]+=1;captures+=1
    direct=sum(token_cap(r) for r in public)+sum(2*token_cap(r) for _ in cfg['AB_routes'] for r in ab)
    assert sum(counts.values())==captures
    costs={key:count*probe(*key)['cuda_elapsed_ms']/1000 for key,count in counts.items()}
    assert prefixes==direct and row_slots==1549+2*216
    assert sum(v['trajectory_slots'] for v in groups.values())==row_slots
    assert not any(k=='torch' or k.startswith('torch.') for k in sys.modules)
    result=dict(experiment='NAN-RAM-28',evidence_class='Source-only accounting and closed engineering preflight; no generated text/state/grade read',
        source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),preflight_sha256=sha(preflight_file),
        token_bindings_sha256=sha('experiments/nan_full26/tokenizer_receipt.json'),
        source_roots=len({r['root'] for r in public}),source_roots_are_identifiers_not_independent_checkpoint_replications=True,
        baseline_trajectories=1549,baseline_initial_batches=baseline_batches,baseline_initial_batch_sizes=dict(Counter(len(rows) for lane,_,rows in batches if lane=='baseline')),
        AB_pairs=216,total_trajectory_slots=row_slots,immediate_EOS_scenario_capture_batches=len(batches),
        full_token_cap_scenario_capture_batches=captures,full_token_cap_scenario_row_prefixes=prefixes,
        full_token_cap_scenario_physical_vectors=prefixes*40,
        physical_vector_site_widths=layout,
        checkpoint_shape_headers_sha256=headers,checkpoint_config_sha256=sha(snapshot/'config.json'),
        checkpoint_index_sha256=sha(snapshot/'model.safetensors.index.json'),
        shape_metadata_only_no_pretrained_tensor_materialized=True,
        physical_vector_payload_bytes_per_row_prefix=payload,
        full_token_cap_scenario_physical_vector_payload_bytes=prefixes*payload,
        payload_scope='Only forty BF16 vectors per row/prefix; excludes boundary/native/mask arrays, zip/JSON/filesystem overhead and other artifacts. Not a total storage guarantee.',
        fixed_artifact_budget_bytes=cfg['maximum_artifact_bytes_per_backend'],fixed_loaded_model_seconds=cfg['maximum_backend_seconds'],
        scenario_probe_equivalent_seconds=sum(costs.values()),
        scenario_scope='Conditional arithmetic: each full-cap prefix is charged the measured enclosing B1/B2 reached-state capacity probe time. This is neither a runtime prediction nor a bound; native vs edited/padded kernels, EOS/packet stops, GPU clocks, I/O, allocation and thermal changes differ.',
        probe_buckets=[dict(batch=B,enclosing_probe_tokens=P,capture_batches_at_full_cap=counts[B,P],full_cap_probe_equivalent_seconds=t) for (B,P),t in sorted(costs.items())],
        passed_capacity_probes=[{k:r[k] for k in ('batch','tokens','right_padding','cuda_elapsed_ms')} for r in engine.capacity if r['status']=='passed'],
        groups=[dict(lane=k[0],route=k[1],stage=k[2],**v) for k,v in sorted(groups.items())],
        execution_order='All 1549 baselines first, then all 216 A/B pairs; a fixed-budget stop leaves later work unknown, without reordering or retry.',
        criteria_or_worker_changed=False,pretrained_forwards=0,Torch_imported=False,sealed_test_accessed=False,formal_comparisons=0)
    write(out/'accounting.json',result);return result


def main():
    p=argparse.ArgumentParser()
    for name in ('artifact-root','snapshot','out-dir'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.out_dir)
    with CPUBudget(out):result=run(a.artifact_root,a.snapshot,out)
    from experiments.nan_ram28.common import read,write,sha
    r=read(out/'resources.json');assert r['status']=='passed'
    result['resources']=dict(samples=len(r['samples']),maximum_sampled_private_bytes=max(x['private_bytes'] for x in r['samples']),
        minimum_sampled_available_physical_bytes=min(x['available_physical_bytes'] for x in r['samples']))
    result['tool_source_sha256']=sha(__file__)
    write(out/'receipt.json',result)
    print({k:result[k] for k in ('baseline_initial_batches','total_trajectory_slots','full_token_cap_scenario_capture_batches',
        'full_token_cap_scenario_row_prefixes','full_token_cap_scenario_physical_vectors','full_token_cap_scenario_physical_vector_payload_bytes',
        'scenario_probe_equivalent_seconds','resources')},flush=True)


if __name__=='__main__':main()
