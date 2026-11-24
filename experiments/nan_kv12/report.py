"""Independent closed-artifact cache/uncached comparison; no task reward claims."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import math
from pathlib import Path
import numpy as np
from .common import read,sha,write,public_rows,verify


def historical_prefixes(runs, public, *, historical_root=Path('runs/nan_kv11')):
    """Compare only common initial histories, never diverged free-running tails."""
    old=historical_root;new=runs/'nf4';groups=defaultdict(list);bindings={}
    for backend in ('nf4','bf16'):
        folder=old/backend;receipt=folder/'receipt.json'
        if receipt.exists():
            r=read(receipt);assert r['source_manifest_sha256']=='bf9cb5d06ffa94d57de11f54dc4d4d4b2ab473c4ccf9f22715fb8d6d7960de67'
            bindings[backend]=dict(status=r['status'],receipt_sha256=sha(receipt))
        for file in sorted(new.glob('case_*.json')):
            prior=folder/file.name
            if not prior.exists():continue
            a,b=read(file),read(prior);ordinal=int(file.stem.split('_')[1]);source=public[ordinal]
            assert a['id']==b['id']==source['id'] and a['initial_prompt_ids']==b['initial_prompt_ids']
            if not a['steps'] or not b['steps']:continue
            x,y=a['steps'][0],b['steps'][0];xp,yp=new/x['artifact_file'],folder/y['artifact_file']
            assert sha(xp)==x['artifact_sha256'] and sha(yp)==y['artifact_sha256']
            with np.load(xp,allow_pickle=False) as z,np.load(yp,allow_pickle=False) as q:
                delta_new=z['reference_boundary_2'].astype(np.float64)-z['reference_boundary_1']
                delta_old=q['reference_boundary_2'].astype(np.float64)-q['reference_boundary_1']
                delta_error=float(np.linalg.norm(delta_new-delta_old)/max(np.linalg.norm(delta_old),1e-12))
                for t in (1,2):
                    h,c=q[f'reference_boundary_{t}'].astype(np.float64),z[f'reference_boundary_{t}'].astype(np.float64)
                    logits,old_logits=z[f'reference_logits_{t}'],q[f'reference_logits_{t}']
                    row=dict(root=source['root'],boundary_relative_l2=float(np.linalg.norm(c-h)/max(np.linalg.norm(h),1e-12)),
                        boundary_cosine=float(c@h/max(np.linalg.norm(c)*np.linalg.norm(h),1e-12)),
                        logit_max_abs=float(np.max(np.abs(logits-old_logits))),
                        free_argmax_equal=int(logits.argmax()==old_logits.argmax()),normalized_loop_delta_relative_l2=delta_error)
                    for layer in (5,11,17,21):
                        key=f'reference_{t}_{layer}_residual_after_layer';v,w=z[key].astype(np.float64),q[key].astype(np.float64)
                        row[f'layer_{layer}_residual_relative_l2']=float(np.linalg.norm(v-w)/max(np.linalg.norm(w),1e-12))
                        row[f'layer_{layer}_residual_cosine']=float(v@w/max(np.linalg.norm(v)*np.linalg.norm(w),1e-12))
                    groups[backend,source['suite'],a['copy_control'],t].append(row)
    summary=[]
    for (backend,suite,copy,t),values in sorted(groups.items()):
        roots=defaultdict(list)
        for r in values:roots[r['root']].append(r)
        metrics={field:float(np.mean([np.mean([r[field] for r in records]) for records in roots.values()])) for field in values[0] if field!='root'}
        summary.append(dict(reference_backend=backend,suite=suite,copy_control=copy,loop=t,contexts=len(values),source_families=len(roots),root_macro=metrics))
    return dict(scope='Only initial full-prefix histories, before any policy-dependent generated-token divergence; no complete-task accuracy',
        KV11_receipt_bindings=bindings,groups=summary)


def report(runs,out):
    root=Path(__file__).parent;seal=verify(root);cfg=read(root/'config.json');_,public=public_rows();result={}
    for name,h in seal['host_only_files_sha256'].items():assert sha(root/name)==h,name
    for backend in cfg['backends']:
        folder=runs/backend;groups=defaultdict(list);seen=set();prefill=0;readouts=0;vectors=0;disagreements=0;steps_total=0;maximum_error=0.;max_rel=0.
        files=sorted(folder.glob('case_*.json'));failed_cases=0
        for file in files:
            r=read(file);ordinal=int(file.stem.split('_')[1]);source=public[ordinal];assert source['id']==r['id'] and r['id'] not in seen;seen.add(r['id'])
            assert all(r[k]==source[k] for k in ('root','suite','split')) and r['copy_control']==source['id'].endswith(':copy')
            assert 0<=len(r['steps'])<=8;failed_cases+=int(r['status']!='completed')
            trajectory=[];prefill_actual=True
            for j,step in enumerate(r['steps']):
                assert step['step']==j and step['history_tokens']==len(r['initial_prompt_ids'])+j
                assert step['cache_sequence_length']==step['history_tokens'] and step['cache_bytes']==180224*step['history_tokens']
                path=folder/step['artifact_file'];assert sha(path)==step['artifact_sha256']
                with np.load(path,allow_pickle=False) as z:
                    expected=set()
                    for t,p in enumerate(step['phases'],1):
                        assert p['loop']==t
                        rk=f'reference_logits_{t}';ck=f'cached_logits_{t}';rh=f'reference_boundary_{t}';ch=f'cached_boundary_{t}';expected.update((rk,ck,rh,ch))
                        a,b=z[rk],z[ck];h,c=z[rh],z[ch];assert a.shape==b.shape==(166144,) and h.shape==c.shape==(3072,)
                        assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in (a,b,h,c));error=float(np.max(np.abs(a-b)))
                        rel=float(np.linalg.norm(c.astype(np.float64)-h)/max(np.linalg.norm(h.astype(np.float64)),1e-12))
                        assert math.isclose(error,p['logit_max_abs'],rel_tol=1e-7,abs_tol=1e-8) and math.isclose(rel,p['normalized_boundary_relative_l2'],rel_tol=1e-4,abs_tol=1e-8)
                        assert p['free_argmax_equal']==(int(a.argmax())==int(b.argmax())) and p['logits_bit_exact']==(a.tobytes()==b.tobytes())
                        if j==0:prefill_actual &= a.tobytes()==b.tobytes() and h.tobytes()==c.tobytes()
                        maximum_error=max(maximum_error,error);max_rel=max(max_rel,rel);readouts+=1
                        if t==2:
                            assert step['reference_token_id']==int(a.argmax()) and step['cached_free_token_id_on_reference_history']==int(b.argmax())
                    for t in (1,2):
                        for layer in (5,11,17,21):
                            for site in ('residual_after_layer','down_proj_input','down_proj_output','o_proj_input','o_proj_output'):
                                rk=f'reference_{t}_{layer}_{site}';ck=f'cached_{t}_{layer}_{site}';expected.update((rk,ck));a,b=z[rk],z[ck]
                                width={'down_proj_input':10752,'o_proj_input':6144}.get(site,3072)
                                assert a.shape==b.shape==(width,) and a.dtype==b.dtype==np.float32 and np.isfinite(a).all() and np.isfinite(b).all()
                                if j==0:prefill_actual &= a.tobytes()==b.tobytes()
                                vectors+=2
                    assert set(z.files)==expected
                equal=step['reference_token_id']==step['cached_free_token_id_on_reference_history'];disagreements+=int(not equal);steps_total+=1
                trajectory.append(step['reference_token_id'])
                groups[source['suite'],r['copy_control']].append(dict(root=source['root'],argmax_equal=int(equal),reference_seconds=step['reference_seconds'],cached_seconds=step['cached_seconds'],step=j))
            assert trajectory==r['reference_generated_ids']
            if r['steps']:
                assert r['steps'][0]['prefill_all_sites_and_logits_bit_exact']==prefill_actual;prefill+=int(prefill_actual)
            if r['status']=='completed':assert prefill_actual and 1<=len(trajectory)<=8 and (r['reference_ended_EOS'] or len(trajectory)==8)
        receipt=read(folder/'receipt.json') if (folder/'receipt.json').exists() else None
        if receipt:
            assert receipt['source_manifest_sha256']==sha(root/'SOURCE_MANIFEST.json') and receipt['attempted_contexts']==len(files)
            assert receipt['completed_contexts']==len(files)-failed_cases
            if receipt['status']=='completed':
                assert len(files)==130 and not failed_cases
                manifest=read(folder/'manifest.json');assert receipt['target_parameter_sha256_unchanged']==manifest['target_parameter_sha256']
        summary=[]
        for (suite,copy),values in sorted(groups.items()):
            roots=defaultdict(list)
            for r in values:roots[r['root']].append(r)
            macro=lambda field:float(np.mean([np.mean([r[field] for r in records]) for records in roots.values()]))
            summary.append(dict(suite=suite,copy_control=copy,source_families=len(roots),readouts=len(values),root_macro_free_argmax_agreement=macro('argmax_equal'),
                root_macro_reference_seconds=macro('reference_seconds'),root_macro_cached_seconds=macro('cached_seconds')))
        driver=[]
        for name in ('driver_after_warmup.json','driver_terminal.json'):
            if (folder/name).exists():
                d=read(folder/name)
                driver.append(dict(stage=name,status=d['status'],dedicated_bytes=sum(v['DedicatedUsage'] for v in d.get('WDDM_worker_rows',[])),
                    shared_bytes=sum(v['SharedUsage'] for v in d.get('WDDM_worker_rows',[])),whole_device_used_MiB=d.get('nvidia_device_used_MiB'),file_sha256=sha(folder/name)))
        result[backend]=dict(status=receipt['status'] if receipt else 'partial_evidence' if files else 'queued',registered_contexts=130,closed_context_records=len(files),driver_samples=driver,
            prefill_contexts_byte_exact=prefill,paired_vocabulary_readouts_checked=readouts,physical_vectors_checked=vectors,reference_token_positions=steps_total,
            free_argmax_disagreements=disagreements,full_vocabulary_logit_max_abs=maximum_error,normalized_boundary_relative_l2_max=max_rel,groups=summary,
            worker_receipt_sha256=sha(folder/'receipt.json') if receipt else None,CUDA_peak_allocated_bytes=receipt['CUDA_peak_allocated_bytes'] if receipt else None,
            CUDA_peak_reserved_bytes=receipt['CUDA_peak_reserved_bytes'] if receipt else None,failure_type=(receipt.get('failure') or {}).get('type') if receipt else None)
    payload=dict(experiment='NAN-KV-12',updated_UTC=datetime.now(timezone.utc).isoformat(),source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'),backends=result,
        unit='41 heterogeneous DEV source families; 130 contexts, copy arms/tokens/loops/backends are repeated measures; one checkpoint',
        primary_contrast='cached versus full-prefix execution on identical natural reference histories, capped at eight tokens/EOS',
        n_units=41,n_cases=130,physical_dose=0,scientific_causal_floor=None,matched_sham_floor=None,estimated_causal_power='not calibrated',
        numerical_policy=cfg['numerical_policy'],historical_common_prefixes=historical_prefixes(runs,public),
        readout_selection_exposure='GYM05, later outcomes and partial KV11 exposed; this is a separately registered execution diagnostic',
        planned_formal_comparisons=0,actual_formal_comparisons=0,BPD_transfer_attempted=False,sealed_test_accessed=False)
    write(out/'nan_kv12_progress.json',payload)
    lines=['# NAN-KV-12: explicit-effective NF4 cached execution','',f"Updated {payload['updated_UTC']}",'',
        'Every NF4 decoder linear explicitly dequantizes its unchanged packed weight to BF16 before F.linear. '+
        'This changes full-prefix arithmetic too; default NF4 KV11 and BF16 KV11 remain separate historical conditions. '+
        'They are compared only at common initial full-prefix histories, with all numerical and physical recurrence differences retained.','',
        'All 130 public GYM05 DEV contexts are registered, across 41 source families: pointer, ARC1/ARC2, Sudoku '+
        'and storyworld, with every public copy control. Private routing and the protected original test are not read. '+
        'The full-prefix model supplies a natural reference history of up to eight tokens/EOS; both paths receive '+
        'exactly that history. This compares numerical execution and measured probe/readout latency, not complete task solving.','',
        '| Backend | Status | Closed contexts | Prefills byte exact | Token argmax disagreements | Max logit difference |',
        '|---|---|---:|---:|---:|---:|']
    for b,r in result.items():lines.append(f"| {b.upper()} | {r['status']} | {r['closed_context_records']}/130 | {r['prefill_contexts_byte_exact']} | {r['free_argmax_disagreements']} | {r['full_vocabulary_logit_max_abs']:.6g} |")
    lines+=['','The aggregate JSON includes all suite/copy/loop groups for matched initial-prefix vocabulary, '+
        'normalized boundary/loop-delta and physical four-layer residual geometry against both KV11 conditions. '+
        'Generated histories can diverge across policies and are not treated as paired later readouts.']
    lines+=['','Independent CPU checks recompute paired differences from every archived vocabulary readout and physical site/boundary array. '+
        'Prefill must be byte exact; a failure stops that backend with partial artifacts preserved. Incremental '+
        'GEMM/GEMV/SDPA differences, argmax changes and all source groups remain visible. Approximate agreement does '+
        'not authorize replacing any frozen full-prefix experiment. A cache argmax on a reference history is not an '+
        'independent cached free-running reply after a divergence. No task accuracy or transfer score is claimed.','',
        'Latency includes both-loop observations and full-vocabulary readout, in fixed reference-then-cache order '+
        'on a machine shared with the CPU math workload. It is not an isolated throughput benchmark. One CPU thread, '+
        'Below Normal priority and an .85 CUDA cap are retained. All raw tokens, states and readouts stay local.','',
        'The original algebraic competence gates remain failed; this execution diagnostic does not repair them. '+
        'No causal/equivalence claim or donor intervention. The original 168-case test stays sealed.','',
        '[Protocol](../../experiments/nan_kv12/PROTOCOL.md) · [Source seal](../../experiments/nan_kv12/SOURCE_MANIFEST.json) · [Aggregate evidence](nan_kv12_progress.json)','']
    (out/'NAN_KV12_PROGRESS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n');print({b:r['status'] for b,r in result.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_kv12');p.add_argument('--out',default='reports/nanbeige_bpd_gym');a=p.parse_args();report(Path(a.runs),Path(a.out))


if __name__=='__main__':main()
