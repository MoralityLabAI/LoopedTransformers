"""Compact provenance-checked index of COMMITTED reports; no model/run reads.

This indexes existing published audits. It is not an independent regrading,
scientific selection rule, live process monitor or inference launcher.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT='reports/nanbeige_bpd_gym/'
BRANCH='research/nanbeige-bpd-gym-suite-towers'
OUTPUTS=(ROOT+'BENCHMARK_INDEX.md',ROOT+'benchmark_index.json')
FLAGS=getattr(subprocess,'CREATE_NO_WINDOW',0)
STUDIES=(
    ('GYM05','nan_gym05','native bounded-prefix-stop wider suite','NAN_GYM05_PROGRESS.md'),
    ('PRIME06','nan_prime06','official SDK complete-final-reply wider suite','NAN_PRIME06_PROGRESS.md'),
    ('FIT07','nan_fit07','native free-first-token screen and eight-step fitting','NAN_FIT07_PROGRESS.md'),
    ('CTX08','nan_ctx08','native free-first-token 1K/2K/4K contexts','NAN_CTX08_PROGRESS.md'),
    ('RECOMP09','nan_recomp09','exposed-source BF16 T2 learned BPD execution amendment','NAN_RECOMP09_PROGRESS.md'),
    ('PRIME-CTX10','nan_prime_ctx10','official SDK complete one-token final reply, context taskset','NAN_PRIME_CTX10_PROGRESS.md'),
    ('KV11','nan_kv11','same-reference-history cache/full-prefix numerical diagnostic','NAN_KV11_PROGRESS.md'),
    ('KV12','nan_kv12','explicit-effective NF4 arithmetic/cache numerical diagnostic','NAN_KV12_PROGRESS.md'),
    ('RANKSITE13','nan_ranksite13','six exposed-source BF16 learned rank/site cells and local finite-write probes','NAN_RANKSITE13_PROGRESS.md'),
)


def digest(data):return hashlib.sha256(data).hexdigest()


def read_git(commit,path,*,optional=False):
    try:return subprocess.check_output(['git','show',commit+':'+path],stderr=subprocess.DEVNULL,creationflags=FLAGS)
    except subprocess.CalledProcessError:
        if optional:return None
        raise ValueError('Missing committed evidence: '+path) from None


def pick(value,fields):return {k:value[k] for k in fields if k in value}


def below_normal():
    if os.name=='nt':
        import ctypes
        k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p;k.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint]
        if not k.SetPriorityClass(k.GetCurrentProcess(),0x4000):raise ctypes.WinError(ctypes.get_last_error())


def score(rows,count):
    return dict(correct=sum(r['correct'] for r in rows),scored_cases=sum(r[count] for r in rows))


def task_scores(rows,count,copy_field,*,development=False,loop2=False):
    selected=[r for r in rows if (not development or r['split']=='development') and (not loop2 or r['loop']==2)]
    return {name:score([r for r in selected if (r[copy_field] in (True,'copy'))==copy],count) for name,copy in (('main',False),('copy',True))}


def extract(identifier,d):
    """Whitelist aggregates. Extra fields, including raw text, are never copied."""
    result=[]
    if identifier=='GYM05':
        for b,c in d['coverage'].items():
            groups=d['summaries'][b]
            result.append(dict(backend=b,status=c['status'],source_root_groups=c['roots'],published_contexts=sum(r['records'] for r in groups),
                scores=task_scores(groups,'records','condition',development=True),fitted_SAE_layers=len(d['tower_diagnostics'][b]['layers'])))
    elif identifier=='FIT07':
        for b,r in d['backends'].items():
            audit=r.get('independent_audit') or {};baseline=audit.get('baseline') or {}
            cells=[]
            for c in r['cells']:
                cells.append(dict(fit_passes=c['passes'],status=c['status'],optimizer_steps=(c.get('training') or {}).get('optimizer_steps',c.get('optimizer_steps_completed')),
                    **pick(c,('factor_update_l2','controller_update_l2','failure_type','checkpoint_sha256'))))
            result.append(dict(backend=b,status=r['status'],allow_fits=baseline.get('allow_fits'),cells=cells))
    elif identifier=='RECOMP09':
        a=d.get('independent_fit_audit') or {};p=d.get('SAE_cross_reference') or {}
        result.append(dict(backend='bf16',status=d['status'],optimizer_steps=d['optimizer_steps_completed'],fit_audit_status=a.get('status'),
            SAE_probe_status=p.get('status'),SAE_probe_groups=len(p.get('groups',[])),
            **pick(a,('factor_update_l2','controller_update_l2','checkpoint_sha256','CUDA_peak_allocated_bytes','CUDA_peak_reserved_bytes'))))
    elif identifier=='RANKSITE13':
        for c in d['cells']:
            a=c.get('fit_independent_audit') or {};p=c.get('physical_write_SAE_probe') or {};w=c.get('cached_suite_probe') or {}
            result.append(dict(backend='bf16',condition=c['condition'],module=c['module'],rank=c['rank'],fit_passes=c['fit_passes'],status=c['status'],
                baseline_contexts_verified=c['baseline_contexts_verified'],optimizer_steps=c['optimizer_steps_completed'],fit_audit_status=a.get('status'),
                finite_writes_verified=p.get('signed_finite_writes_verified'),cached_probe_groups=len(w.get('groups',[])),
                **pick(c,('failure_type','CUDA_peak_allocated_bytes','CUDA_peak_reserved_bytes'))))
    else:
        for b,r in d['backends'].items():
            if identifier=='PRIME06':
                a=r.get('independent_attempt_audit') or {}
                result.append(dict(backend=b,status=r['worker_terminal_status'],
                    graded_episodes=a['unique_successfully_graded_contexts'],registered_contexts=a['registered_contexts'],missing_graded_contexts=a['missing_graded_contexts'],
                    fresh_neural_requests=r['fresh_worker_requests_recorded'],scores=task_scores(r['groups'],'contexts','condition')))
            elif identifier=='CTX08':
                result.append(dict(backend=b,**pick(r,('status','completed_contexts','registered_contexts','physical_vectors_checked','CUDA_peak_allocated_bytes','CUDA_peak_reserved_bytes','failure_type')),
                    scores=task_scores(r['groups'],'contexts','copy_control',development=True,loop2=True)))
            elif identifier=='PRIME-CTX10':
                result.append(dict(backend=b,**pick(r,('status','registered_contexts','neural_requests','successful_official_episodes','missing_graded_contexts',
                    'orphan_or_unbound_neural_replies','physical_prefix_vectors_byte_exact')),
                    scores=task_scores(r['groups'],'graded_contexts','copy_control')))
            else:
                result.append(dict(backend=b,**pick(r,('status','registered_contexts','closed_context_records','prefill_contexts_byte_exact',
                    'paired_vocabulary_readouts_checked','physical_vectors_checked','reference_token_positions','free_argmax_disagreements',
                    'full_vocabulary_logit_max_abs','normalized_boundary_relative_l2_max','CUDA_peak_allocated_bytes','CUDA_peak_reserved_bytes','failure_type'))))
    for r in result:
        for s in r.get('scores',{}).values():
            if not 0<=s['correct']<=s['scored_cases']:raise ValueError('Invalid correctness denominator: '+identifier)
        if identifier in ('PRIME06','PRIME-CTX10'):
            graded=r['graded_episodes'] if identifier=='PRIME06' else r['successful_official_episodes']
            if graded+r['missing_graded_contexts']!=r['registered_contexts']:raise ValueError('Invalid graded/missing coverage: '+identifier)
            if sum(s['scored_cases'] for s in r['scores'].values())!=graded:raise ValueError('SDK group denominator mismatch: '+identifier)
        if identifier in ('KV11','KV12'):
            if not 0<=r['prefill_contexts_byte_exact']<=r['closed_context_records']<=r['registered_contexts']:raise ValueError('Invalid cache context coverage')
            if not 0<=r['free_argmax_disagreements']<=r['reference_token_positions']:raise ValueError('Invalid numerical diagnostic denominator')
    return result


def study(spec,loader):
    identifier,folder,endpoint,markdown=spec;manifest='experiments/'+folder+'/SOURCE_MANIFEST.json';report=ROOT+folder+'_progress.json'
    seal_raw=loader(manifest,False);seal=json.loads(seal_raw)
    raw=loader(report,identifier=='KV12')
    result=dict(study=identifier,endpoint=endpoint,source_manifest_path=manifest,source_manifest_sha256=digest(seal_raw),
        report_path=report,report_markdown=ROOT+markdown)
    if raw is None:
        result.update(published_report=False,report_sha256=None,rows=[],status='registered_no_published_execution_report')
        return result
    d=json.loads(raw)
    if d['source_manifest_sha256']!=digest(seal_raw):raise ValueError('Published report/source seal mismatch: '+identifier)
    if d.get('sealed_test_accessed') is not False:raise ValueError('Original-test seal assertion missing/changed: '+identifier)
    for k in ('BPD_transfer_attempted','BPD_donor_transfer_attempted','BPD_donor_transfer_enabled'):
        if d.get(k,False):raise ValueError('This development index does not classify donor transfer: '+identifier)
    for k in ('formal_comparisons','formal_inferential_comparisons','planned_formal_comparisons','actual_formal_comparisons'):
        if d.get(k,0)!=0:raise ValueError('New inference family requires explicit index review: '+identifier)
    result.update(published_report=True,report_sha256=digest(raw),updated_UTC=d.get('updated_UTC'),rows=extract(identifier,d))
    return result


def build(commit,*,loader=None,producer_sha256=None):
    if loader is None:loader=lambda path,optional:read_git(commit,path,optional=optional)
    studies=[study(s,loader) for s in STUDIES]
    signature=digest(json.dumps(dict(producer=producer_sha256,evidence=[pick(r,('study','report_sha256','source_manifest_sha256')) for r in studies]),sort_keys=True).encode())
    return dict(schema_version=1,source_commit=commit,producer_sha256=producer_sha256,evidence_signature=signature,studies=studies,
        evidence_class='provenance-checked index of committed aggregate audits, not independent regrading or a live process claim',
        original_algebraic_competence_gate='failed and unchanged; pointer retrieval and engineering do not repair it',
        endpoint_policy='native bounded-prefix, free-first-token, complete SDK final-reply and same-history numerical diagnostics remain separate; no pooled accuracy',
        units='source roots/families and checkpoint counts remain study-specific; tokens, loops, variants, features and rank/site arms are correlated measures',
        formal_inferential_comparisons=0,BPD_donor_transfer_attempted=False,sealed_test_accessed=False,
        privacy='named committed aggregate reports/source seals only; no local runs, prompts, replies, tokens, states or per-record codes are read')


def render(data):
    lines=['# Nanbeige benchmark evidence index','',
        'This page indexes committed aggregate reports and their source seals. It is not a fresh grading pass or a live-process claim. '+
        'The linked reports retain complete groups, controls, failures and missing coverage.','',
        f"Evidence source commit: `{data['source_commit']}`. Signature: `{data['evidence_signature']}`.",'',
        '| Study / endpoint | Backend or fit | Published status | Evidence |','|---|---|---|---|']
    for s in data['studies']:
        link=s['report_markdown'].split('/')[-1] if s['published_report'] else '../../'+s['source_manifest_path']
        label=f"[{s['study']}]({link})"
        if not s['rows']:
            lines.append(f"| {label}: {s['endpoint']} | registered | {s['status']} | No published execution score |")
            continue
        for r in s['rows']:
            who=r.get('condition',r['backend']);status=r.get('status','unknown');parts=[]
            if 'scores' in r:
                for name,v in r['scores'].items():parts.append(f"{name} {v['correct']}/{v['scored_cases']}")
            if 'published_contexts' in r:parts.append(f"{r['published_contexts']} contexts; {r['source_root_groups']} root groups; {r['fitted_SAE_layers']} SAE layers")
            for field,title in (('graded_episodes','graded SDK episodes'),('successful_official_episodes','graded SDK episodes'),('missing_graded_contexts','missing grades'),
                ('completed_contexts','closed contexts'),('closed_context_records','closed contexts'),('prefill_contexts_byte_exact','exact prefills'),
                ('free_argmax_disagreements','loop-2 argmax disagreements'),('optimizer_steps','optimizer steps'),('finite_writes_verified','audited finite writes')):
                if r.get(field) is not None:parts.append(f"{r[field]} {title}")
            if 'cells' in r:
                for c in r['cells']:parts.append(f"T{c['fit_passes']} {c['status']}"+(f" ({c['optimizer_steps']} steps)" if c['optimizer_steps'] is not None else ''))
            lines.append(f"| {label}: {s['endpoint']} | {who} | {status} | {'; '.join(parts)} |")
    lines+=['','Accuracy columns use each endpoint\'s own scored denominator; partial SDK coverage remains partial. '+
        'Successful SDK episodes denote valid recorded grades, not correct model answers. CTX08 accuracy is DEV loop-2 free-first-token; '+
        'its 288 correlated DEV contexts arise from four source families. GYM05 uses native prefix stopping; Prime uses complete final replies. '+
        'PRIME-CTX10 reuses exposed CTX08 DEV sources; agreement adds transport/endpoint evidence, not independent source roots. '+
        'Cache fragments receive no complete-task accuracy score, and cache argmaxes use the full-prefix reference history rather than an independent reply after divergence.','',
        'BF16/NF4 stay distinct; explicit-effective NF4 changes full-prefix arithmetic as well as incremental execution. '+
        'Both original cache backends have measured token-choice disagreements despite exact prefills. '+
        'The original failed NF4 fitting screen, BF16 saved-tensor failure and incomplete Prime attempts remain preserved. '+
        'There is no automatic retry, orphan-response rescue or reinterpretation of missing grades as wrong answers.','',
        'Eight CAL-only SAE dictionaries and actual learned BF16 VPD/BPD assets are separate from weight-SVD/random controls. '+
        'Static CAL-mean routes retaining the same small-DEV correctness limit behavioral specificity. '+
        'Rank/site finite-write probes include the real o_proj residual/norm/MLP path; its derivative is not assumed equivalent to rounded finite writes. '+
        'No geometry/correlation or random-model engineering result establishes donor-specific mediation.','',
        'All studies are exposed development work on one checkpoint. Roots/families, cases, loops, tokens and features are not interchangeable sample sizes; '+
        'no accuracy is pooled across studies and no formal confirmation/equivalence is claimed. Original algebraic gates remain failed. '+
        'The original 168-case test stays sealed.','',
        'Resource policy remains one large model, one numerical CPU thread and Below Normal priority, with the separate math process untouched. '+
        'NF4 decoder layers/head are GPU resident; embedding lookup stays on CPU to retain context/cache capacity. '+
        'BF16 streams decoder layers. [Dedicated/shared device counters](NAN_PRIME_CTX10_BF16_MEMORY.md) remain separate from CUDA allocator peaks.','',
        '[Exact aggregate source hashes and fields](benchmark_index.json) · [Requirements and remaining work](ROADMAP.md) · '+
        '[Original gated causal campaign](../../papers/shared_weights_dark_fibers/NAN_BPD01_ADDENDUM.md)','']
    return '\n'.join(lines)


def refresh():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,creationflags=FLAGS).strip()
    source=Path(__file__).read_bytes();path=Path(__file__).resolve().relative_to(Path.cwd().resolve()).as_posix()
    if digest(read_git(commit,path))!=digest(source):raise ValueError('Commit the index implementation before publishing its output')
    data=build(commit,producer_sha256=digest(source));old=read_git(commit,OUTPUTS[1],optional=True)
    if old and json.loads(old)['evidence_signature']==data['evidence_signature']:
        return dict(changed=False,evidence_signature=data['evidence_signature'],source_commit=commit)
    data['generated_UTC']=datetime.now(timezone.utc).isoformat()
    Path(OUTPUTS[0]).write_text(render(data),encoding='utf-8',newline='\n')
    Path(OUTPUTS[1]).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
    return dict(changed=True,evidence_signature=data['evidence_signature'],source_commit=commit)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    below_normal();print(refresh())


if __name__=='__main__':main()
