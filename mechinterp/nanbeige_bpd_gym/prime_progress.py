"""Aggregate-only real Prime episode progress; raw local routing stays local."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
from experiments.nan_gym05.common import read,write,sha
from experiments.nan_prime06.wire import prompt_sha
from .prime_evidence import raw_reply_bound,successful_episode


def progress(runs,out):
    source=Path('experiments/nan_prime06/SOURCE_MANIFEST.json');backends={}
    for backend in ('nf4','bf16'):
        episodes_file=runs/'prime'/('nan-prime06-'+backend)/'traces.jsonl'
        episodes=[]
        if episodes_file.exists():
            for line in episodes_file.read_text(encoding='utf-8').splitlines():
                if not line:continue
                try:episodes.append(json.loads(line))
                except json.JSONDecodeError:break  # the official writer may be appending its final line
        groups=defaultdict(list);failed=defaultdict(int);bindings=0;errors=0;sdk_trimmed=0
        for i,e in enumerate(episodes):
            data=e['task']['data'];label=data['private_label']
            condition='copy' if label['copy_control'] else ('hop'+str(label['hops']) if label.get('hops') else 'main')
            if len(e['traces'])!=1:
                errors+=1;failed[data['suite'],condition]+=1;continue
            t=e['traces'][0]
            if t['verifiers']['commit']!='ed70053b6a7a4079d6976e03bdcd7d420356d594':raise ValueError('Wrong official SDK revision')
            if t['agent']['runtime']['type']!='subprocess' or t['agent']['config']['harness']['id']!='null':raise ValueError('Only registered local null-harness rollouts')
            if not successful_episode(e):
                errors+=1;failed[data['suite'],condition]+=1;continue
            worker=runs/backend/f'request_{i:04d}.json'
            if worker.exists():
                r=read(worker)
                if r['prompt_sha256']==prompt_sha(data['prompt']) and raw_reply_bound(r,t):
                    bindings+=1;sdk_trimmed+=int(r['text']!=t['last_reply'])
            if t['reward'] not in (0.,1.):raise ValueError('Unexpected registered task reward')
            groups[data['suite'],condition].append((data['root_id'],int(t['reward']),bool(t['info'].get('format_valid',False))))
        observed=read(runs/backend/'progress.json')['requests'] if (runs/backend/'progress.json').exists() else 0
        receipt=read(runs/backend/'receipt.json') if (runs/backend/'receipt.json').exists() else None
        backends[backend]={'completed_official_episodes':len(episodes),'fresh_worker_requests_recorded':observed,
            'episode_worker_reply_bindings_verified':bindings,'recorded_errors':errors,
            'SDK_last_reply_outer_whitespace_trimmed_episodes':sdk_trimmed,
            'successful_official_episodes':len(episodes)-errors,
            'failed_or_incomplete_episode_groups':[{'suite':k[0],'condition':k[1],'episodes':v} for k,v in sorted(failed.items())],
            'worker_terminal_status':receipt['status'] if receipt else None,'complete':len(episodes)==bindings==162 and bool(receipt and receipt['status']=='completed') and not errors,
            'groups':[{'suite':k[0],'condition':k[1],'contexts':len(rr),'source_root_groups':len({r[0] for r in rr}),
                'correct':sum(r[1] for r in rr),'format_valid':sum(r[2] for r in rr)} for k,rr in sorted(groups.items())]}
        audit=runs/'audit'/(backend+'_independent.json')
        if audit.exists():
            verified=read(audit)
            if verified['status']!='passed' or verified['worker_receipt_sha256']!=sha(runs/backend/'receipt.json') or verified['source_manifest_sha256']!=sha(source):raise ValueError('Unbound Prime audit')
            backends[backend]['independent_audit']=verified
            backends[backend]['independent_audit_sha256']=sha(audit)
    result={'experiment':'NAN-PRIME-06','updated_UTC':datetime.now(timezone.utc).isoformat(),'source_manifest_sha256':sha(source),
        'endpoint':'complete final reply, distinct from native prefix-stop; no pooling','backends':backends,
        'evidence_class':'exposed development; official SDK rewards pending independent final audit',
        'formal_inferential_comparisons':0,'BPD_transfer_attempted':False,'sealed_test_accessed':False,
        'privacy':'aggregate episode/reply counts only; no routing text, raw predictions, source IDs, tokens or vectors'}
    write(out/'nan_prime06_progress.json',result)
    lines=['# NAN-PRIME-06 local model progress','',f"Updated {result['updated_UTC']}",'',
        'Real pinned Nanbeige replies pass through the official Prime vf-run / subprocess / null-harness / trace / task-reward path on this machine. No cloud runtime or hosted inference is used. The fixed mock engineering episodes are excluded.','',
        'Partial source-order counts are not final accuracy estimates. This condition scores complete final replies through EOS or the registered schema cap; its results are separate from native first-complete-prefix scores.','',
        '| Backend | Official episodes recorded | Fresh requests recorded | Worker/reply bindings | Failed/incomplete episodes |','|---|---:|---:|---:|---:|']
    for b,r in backends.items():lines.append(f"| {b.upper()} | {r['completed_official_episodes']}/162 | {r['fresh_worker_requests_recorded']} | {r['episode_worker_reply_bindings_verified']} | {r['recorded_errors']} |")
    lines+=['','Bindings require the saved raw assistant message to match the worker reply exactly. The pinned SDK `Trace.last_reply` accessor separately strips outer whitespace; that behavior is counted explicitly in the JSON. The registered anchored grammar already strips outer whitespace, so this representation check changes neither model output nor task grades.']
    lines+=['','Correctness tables include successful official episodes only. Failed or incomplete episodes remain explicit missing graded coverage, with per-suite counts in the JSON; they are neither fabricated zero-reward answers nor new independent roots. Final completeness also requires all 162 raw reply bindings.']
    lines+=['','| Backend | DEV suite | Condition | Correct | Format valid | Source roots represented |','|---|---|---|---:|---:|---:|']
    for b,data in backends.items():
        for r in data['groups']:lines.append(f"| {b.upper()} | {r['suite']} | {r['condition']} | {r['correct']}/{r['contexts']} | {r['format_valid']}/{r['contexts']} | {r['source_root_groups']} |")
    for b,data in backends.items():
        if 'independent_audit' in data:
            audit=data['independent_audit'];main=[r for r in audit['summaries'] if r['condition']!='copy'];copy=[r for r in audit['summaries'] if r['condition']=='copy']
            lines+=['',f"{b.upper()} independent audit passed: {audit['episodes']} episodes, {audit['fresh_worker_requests']} fresh replies and {audit['state_vectors']} physical vectors. Final-reply main correctness {sum(r['correct'] for r in main)}/105; copy {sum(r['correct'] for r in copy)}/57. Native generated-prefix tokens agree in {audit['native_prefix_token_agreements']}/162 contexts. These are endpoint diagnostics, not new independent model replications."]
            if 'native_prefix_geometry' in audit:
                geometry=audit['native_prefix_geometry']
                lines+=['',f"{b.upper()} prefix-state check: {geometry['bit_exact_vectors']}/{geometry['matched_vectors']} physical vectors match native inference byte for byte; maximum absolute change {geometry['maximum_absolute_change']:.6g}. The JSON retains all source-root-weighted site/loop groups and loop-displacement comparisons. Residuals are measured before the final loop RMSNorm. This checks transport/endpoint consistency within a backend, not BF16/NF4 equivalence."]
            if b=='nf4':lines+=['','For NF4, native prefix-stop scores were 26/105 main and 57/57 copy; complete-reply scores are 22/105 and 55/57. All 162 prefixes agree exactly, so these six lost correct answers follow the shared prefixes. Whole-output format validity falls from 160/162 to 131/162; no parser repair or inference change was introduced.']
    lines+=['','[Frozen protocol](../../experiments/nan_prime06/PROTOCOL.md); registration commit `f3859240d5a7e1aa77c650f431fd8d5b2a6a099c`. The final independent audit checks SDK/worker/source bindings, tokenizer decoding, scores, full replies and all captured physical-vector dimensions. The one-checkpoint, reused-development design supports descriptive findings only. No VPD/BPD transfer is run and the original 168-case test stays sealed.','']
    (out/'NAN_PRIME06_PROGRESS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print({b:{k:r[k] for k in ('completed_official_episodes','fresh_worker_requests_recorded','episode_worker_reply_bindings_verified','recorded_errors')} for b,r in backends.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_prime06');p.add_argument('--out',default='reports/nanbeige_bpd_gym');a=p.parse_args()
    progress(Path(a.runs),Path(a.out))


if __name__=='__main__':main()
