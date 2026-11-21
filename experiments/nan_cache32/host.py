"""All planned slots: independent bytes/tokens, fixed probes, then full batch proof."""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
from .common import ROOT,RUNS,read,write,append,sha,verify,load_sources,source_group,assert_no_torch
from .plan import build,digest


def journal(path):
    path=Path(path)
    if not path.exists():return {}
    result={}
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):continue
            item=json.loads(line);assert item['file'] not in result,'Duplicate closed entry'
            result[item['file']]=item
    return result


def terminal_integrity(base,manifest_sha256,plan_sha256,expected_target):
    """Raw fragments never substitute for a missing normal integrity receipt."""
    base=Path(base)
    if not all((base/name).exists() for name in ('manifest.json','receipt.json','preflight.json')):return False
    manifest=read(base/'manifest.json');receipt=read(base/'receipt.json');preflight=read(base/'preflight.json')
    return (manifest['source_manifest_sha256']==manifest_sha256 and manifest['plan_sha256']==plan_sha256
        and receipt['source_manifest_sha256']==manifest_sha256 and receipt['plan_sha256']==plan_sha256
        and preflight['status']=='passed' and receipt.get('normal_terminal_integrity_verified') is True
        and receipt.get('checkpoint_files_verified_after') is True and receipt.get('quant_state_unchanged') is True
        and receipt.get('target_before')==receipt.get('target_after')==expected_target)


def run(artifact,snapshot):
    from .host_budget import Budget,HostResourceBudgetReached
    budget=Budget();seal=verify('host');artifact=Path(artifact)
    terminal=read(artifact/'model_lane_terminal.json')
    assert terminal['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST.json') and terminal['all_actual_workers_ended']
    from .predecessors import ensure_inventory_clear
    ensure_inventory_clear(exclude_pid=__import__('os').getpid())
    from importlib.metadata import version
    cfg=read(ROOT/'config.json')
    for name,expected in cfg['host_packages'].items():assert version(name)==expected,name
    from transformers import PreTrainedTokenizerFast
    from experiments.nan_gym05.common import prompt_ids
    from experiments.nan_full26.host import labels
    from mechinterp.nanbeige_bpd_gym.chunked_audit import pinned_layout,audit_episode,complete_batch_signatures
    from mechinterp.nanbeige_bpd_gym.chunked_probes import load_fixed_assets,project_episode
    from .loader import TARGETS
    source,pins=load_sources();assert pins==seal['public_source_pins'];byid={r['id']:r for r in source}
    tokens=read('experiments/nan_full26/tokenizer_receipt.json');plan=build(source,tokens['contexts']);assert digest(plan)==seal['plan_sha256']
    for name,expected in tokens['tokenizer_files_sha256'].items():assert sha(Path(snapshot)/name)==expected,name
    tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(snapshot)/'tokenizer.json'));wrapped=SimpleNamespace(tok=tok)
    layout=pinned_layout(snapshot);gold=labels();out=artifact/'host';out.mkdir(exist_ok=False)
    events=out/'events.jsonl';events.touch(exist_ok=False)
    registered={f"{job['lane']}_{job['ordinal']:04d}_r{i}_closed.json" for job in plan['jobs'] for i in range(len(job['members']))}
    counts={}
    try:
        for backend in cfg['backends']:
            base=artifact/backend;entries=journal(base/'closed.jsonl');assert set(entries)<=registered,'Unregistered closed neural row'
            qualified=terminal_integrity(base,sha(ROOT/'SOURCE_MANIFEST.json'),digest(plan),TARGETS[backend]);fixed=None
            counts[backend]=dict(registered_rows=1981,closed_raw_rows=len(entries),byte_audited_rows=0,numerically_projected_rows=0,primary_verified_rows=0,
                normal_terminal_integrity_verified=qualified,failed_rows=0)
            if qualified and entries:fixed=load_fixed_assets(backend,TARGETS[backend],budget.check)
            for job in plan['jobs']:
                budget.check();members=[]
                for m in job['members']:
                    r=byid[m['source_id']];ids=prompt_ids(wrapped,snapshot,r['prompt']);pin=tokens['contexts'][r['id']]
                    from mechinterp.nanbeige_bpd_gym.chunked_audit import ids_hash
                    assert len(ids)==pin['tokens'] and ids_hash(ids)==pin['ids_sha256']
                    members.append(dict(source_id=r['id'],mode=m['mode'],initial_ids=ids))
                audited={};projected={};slots=[]
                for i,m in enumerate(job['members']):
                    name=f"{job['lane']}_{job['ordinal']:04d}_r{i}_closed.json";entry=entries.get(name);r=byid[m['source_id']]
                    condition=cfg['AB_routes'][job['ordinal']//72] if job['lane']=='AB' else None
                    slot=dict(backend=backend,lane=job['lane'],ordinal=job['ordinal'],original_row=i,source_id=r['id'],root=r['root'],stage=r['stage'],split=r['split'],
                        source_group=source_group(r,gold[r['id']],condition),status='missing_closed_row',primary_verified=False,grade=None)
                    slots.append(slot)
                    if entry is None:continue
                    try:
                        from .common import parse
                        audit=audit_episode(base,entry,r,members=members,ordinal=job['ordinal'],lane=job['lane'],original_row=i,
                            decode=lambda ids:tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),parse=parse,
                            eos_token_id=tok.eos_token_id,layout=layout,token_bound=tokens['contexts'][r['id']])
                        rawfile=out/(backend+'_'+name.removesuffix('.json')+'_raw_audit.json');write(rawfile,audit)
                        audited[i]=audit;counts[backend]['byte_audited_rows']+=1;slot.update(status='target_unverified',byte_audit_file=rawfile.name,byte_audit_sha256=sha(rawfile))
                        if not qualified:continue
                        summary=project_episode(base,audit,r,gold[r['id']],fixed,out,geometry_keys={},expected_target_digest=TARGETS[backend],guard=budget.check)
                        summaryfile=out/(backend+'_'+name.removesuffix('.json')+'_projection.json');write(summaryfile,summary)
                        projected[i]=summary;counts[backend]['numerically_projected_rows']+=1
                        slot.update(status='complete_context_pending',projection_file=summaryfile.name,projection_sha256=sha(summaryfile))
                    except Exception as error:
                        if isinstance(error,HostResourceBudgetReached):raise
                        counts[backend]['failed_rows']+=1;slot.update(status='audit_or_projection_failed',failure_type=type(error).__name__)
                        # Private detail, never copied into the public aggregate.
                        failurefile=out/(backend+'_'+name.removesuffix('.json')+'_failure.json')
                        write(failurefile,dict(type=type(error).__name__,message=str(error),closed_entry=entry,sealed_test_accessed=False))
                complete=complete_batch_signatures([audited[i] for i in projected]) if projected else {}
                proofs=[]
                for i in projected:
                    proof=audited[i];eligible=all(complete[p['file']]['signature'] is not None for p in proof['prefixes'])
                    slots[i].update(status='verified' if eligible else 'incomplete_cache_context',primary_verified=eligible,
                        grade=projected[i]['grade'] if eligible else None,own_packet_grade_private=projected[i]['grade'])
                    counts[backend]['primary_verified_rows']+=int(eligible)
                    proofs.append(dict(original_row=i,closed_file=proof['closed_file'],closed_sha256=proof['closed_sha256'],
                        prefix_proofs=proof['prefixes'],projection_file=slots[i]['projection_file'],projection_sha256=slots[i]['projection_sha256']))
                side=out/f"{backend}_{job['lane']}_{job['ordinal']:04d}_batch.json"
                write(side,dict(backend=backend,lane=job['lane'],ordinal=job['ordinal'],execution_index=job['execution_index'],
                    source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),slots=slots,complete_cache_prefixes=complete,proofs=proofs))
                append(events,dict(event='scheduled_batch_audited',backend=backend,file=side.name,sha256=sha(side)))
            del fixed;gc.collect();assert_no_torch()
        from .paired import run as pair
        pair(artifact,plan,byid,budget)
        write(out/'receipt.json',dict(status='completed',source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),counts=counts,
            registered_rows_both_backends=3962,new_model_calls=0,Torch_imported=False,formal_comparisons=0,sealed_test_accessed=False))
    finally:budget.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact-root',required=True);p.add_argument('--snapshot',required=True)
    a=p.parse_args();run(a.artifact_root,a.snapshot)
